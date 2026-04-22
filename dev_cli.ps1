param(
  [string]$ModelProfile = $env:C3RNT2_PROFILE,
  [int]$BackendPort = 8000,
  [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"

if (-not $ModelProfile) { $ModelProfile = "rtx4080_16gb_gemma4_26b_a4b_hf" }

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $root "c3_rnt2_ai"
$frontendDir = Join-Path $root "vortex-chat"
$pythonExe = Join-Path $root ".venv\\Scripts\\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
  $pythonExe = "python"
}

Write-Host "[dev] backend: $pythonExe -m vortex serve --profile $ModelProfile --host 0.0.0.0 --port $BackendPort"
Write-Host "[dev] frontend: npm run dev -- --host 0.0.0.0 --port $FrontendPort"

$backend = Start-Process `
  -FilePath $pythonExe `
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
