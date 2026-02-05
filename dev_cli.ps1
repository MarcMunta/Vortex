param(
  [string]$ModelProfile = $env:C3RNT2_PROFILE,
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

if (-not $ModelProfile) { $ModelProfile = "dev_small" }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "c3_rnt2_ai"
$frontendDir = Join-Path $root "vortex-chat"

Write-Host "[dev] backend: python -m vortex serve --profile $ModelProfile --host 0.0.0.0 --port $BackendPort"
Write-Host "[dev] frontend: npm run dev -- --host 0.0.0.0 --port $FrontendPort"

$backend = Start-Process `
  -FilePath "python" `
  -WorkingDirectory $backendDir `
  -ArgumentList @("-m","vortex","serve","--profile",$ModelProfile,"--host","0.0.0.0","--port",$BackendPort) `
  -PassThru

try {
  Push-Location $frontendDir
  npm run dev -- --host 0.0.0.0 --port $FrontendPort
}
finally {
  Pop-Location
  if ($backend -and -not $backend.HasExited) {
    Stop-Process -Id $backend.Id -Force
  }
}
