param(
  [switch]$Build,
  [switch]$NoBrowser,
  [switch]$Logs,
  [string]$ApiProfile = "rtx4080_16gb_llama2_7b_q4_local"
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[docker-run] " + $Message)
}

function Wait-HttpOk([string]$Uri, [int]$TimeoutSec = 30) {
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
  while ([DateTime]::UtcNow -lt $deadline) {
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 3
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 1000
    }
  }
  return $false
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $root "c3_rnt2_ai\docker-compose.yml"

if (-not (Test-Path -LiteralPath $composeFile)) {
  throw "Compose file not found: $composeFile"
}

$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if (-not $dockerCmd) {
  throw "docker was not found in PATH."
}

$env:VORTEX_API_PROFILE = $ApiProfile
if (-not $env:C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT) {
  $env:C3RNT2_HOST_WORKSPACE_WINDOWS_ROOT = Split-Path -Parent $root
}

$spatialModelPath = Join-Path $root "vortex-chat\public\models\hand_landmarker.task"
if (-not (Test-Path -LiteralPath $spatialModelPath)) {
  Write-Step "Spatial assets missing. Run .\\scripts\\setup_spatial_assets.ps1 for full local hand tracking."
}

$cleanupArgs = @("compose", "-f", $composeFile, "--profile", "manual", "--profile", "qwen-sglang", "stop", "sglang-runtime", "vortex-api-sglang", "model-init", "eval")
Write-Step "Stopping legacy SGLang/Qwen services if they are still running..."
& docker @cleanupArgs *> $null

$args = @("compose", "-f", $composeFile, "up", "-d")
if ($Build) {
  $args += "--build"
}
$args += @("vortex-api", "vortex-control", "vortex-frontend")

Write-Step ("Running: docker " + ($args -join " "))
Write-Step ("API profile: " + $ApiProfile)
& docker @args
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$frontendUrl = "http://127.0.0.1:4173"
$controlUrl = "http://127.0.0.1:8765/control/status"
$apiUrl = "http://127.0.0.1:8000/readyz"

Write-Step "Waiting for frontend and control plane..."
$frontendReady = Wait-HttpOk -Uri $frontendUrl -TimeoutSec 90
$controlReady = Wait-HttpOk -Uri $controlUrl -TimeoutSec 90
$apiReady = Wait-HttpOk -Uri $apiUrl -TimeoutSec 300

Write-Host ""
Write-Step "Summary"
Write-Host ("  Frontend: " + $frontendUrl + " (ready=" + $frontendReady + ")")
Write-Host ("  Control:  http://127.0.0.1:8765/control/status (ready=" + $controlReady + ")")
Write-Host ("  Backend:  http://127.0.0.1:8000/readyz (ready=" + $apiReady + ")")
Write-Host ("  Compose:  " + $composeFile)
Write-Host ("  Profile:  " + $ApiProfile)
Write-Host ""
Write-Host ("Logs: docker compose -f `"$composeFile`" logs -f vortex-api vortex-control vortex-frontend")

if ($Logs) {
  & docker compose -f $composeFile logs -f vortex-api vortex-control vortex-frontend
  exit $LASTEXITCODE
}

if (-not $NoBrowser) {
  Start-Process -FilePath ($frontendUrl + "/") | Out-Null
}
