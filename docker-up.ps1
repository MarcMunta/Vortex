param(
    [switch]$Gpu,
    [switch]$Build,
    [int]$FrontendPort = 0,
    [int]$ApiPort = 0,
    [int]$ControlPort = 0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Test-PortUsed {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Resolve-Port {
    param(
        [string]$Name,
        [int]$Requested,
        [int]$Default,
        [int]$Fallback,
        [string]$EnvName
    )
    $envItem = Get-Item -Path "Env:$EnvName" -ErrorAction SilentlyContinue
    if ($Requested -gt 0) {
        return $Requested
    }
    if ($envItem -and $envItem.Value) {
        return [int]$envItem.Value
    }
    if (-not (Test-PortUsed $Default)) {
        return $Default
    }
    if (-not (Test-PortUsed $Fallback)) {
        Write-Host "[docker] $Name port $Default ocupado; usando $Fallback."
        return $Fallback
    }
    Write-Host "[docker] $Name ports $Default y $Fallback ocupados; reusando $Fallback si pertenece a este compose."
    return $Fallback
}

$resolvedFrontend = Resolve-Port -Name "Frontend" -Requested $FrontendPort -Default 5173 -Fallback 15173 -EnvName "VORTEX_FRONTEND_PORT"
$resolvedApi = Resolve-Port -Name "Api" -Requested $ApiPort -Default 8000 -Fallback 18000 -EnvName "VORTEX_API_PORT"
$resolvedControl = Resolve-Port -Name "Control" -Requested $ControlPort -Default 8765 -Fallback 18765 -EnvName "VORTEX_CONTROL_PORT"

$env:VORTEX_FRONTEND_PORT = [string]$resolvedFrontend
$env:VORTEX_API_PORT = [string]$resolvedApi
$env:VORTEX_CONTROL_PORT = [string]$resolvedControl

$files = @("-f", "docker-compose.yml")
if ($Gpu) {
    $files += @("-f", "docker-compose.gpu.yml")
}

$args = @("compose") + $files + @("up", "-d")
if ($Build) {
    $args += "--build"
}

& docker @args
$code = $LASTEXITCODE
if ($code -eq 0) {
    Write-Host "[docker] frontend http://localhost:$resolvedFrontend"
    Write-Host "[docker] api      http://localhost:$resolvedApi"
    Write-Host "[docker] control  http://localhost:$resolvedControl"
}
exit $code
