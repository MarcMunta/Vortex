param()

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[spatial-assets] " + $Message)
}

$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "vortex-chat"
$nodeModules = Join-Path $frontend "node_modules\@mediapipe\tasks-vision\wasm"
$publicWasm = Join-Path $frontend "public\mediapipe\wasm"
$publicModels = Join-Path $frontend "public\models"
$modelTarget = Join-Path $publicModels "hand_landmarker.task"
$modelUrl = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if (-not (Test-Path -LiteralPath $nodeModules)) {
  throw "MediaPipe wasm folder not found. Run npm install in vortex-chat first."
}

New-Item -ItemType Directory -Force -Path $publicWasm | Out-Null
New-Item -ItemType Directory -Force -Path $publicModels | Out-Null

Write-Step "Copying local MediaPipe WASM runtime..."
Copy-Item -Path (Join-Path $nodeModules "*") -Destination $publicWasm -Recurse -Force

if (-not (Test-Path -LiteralPath $modelTarget)) {
  Write-Step "Downloading local hand landmarker model..."
  Invoke-WebRequest -UseBasicParsing -Uri $modelUrl -OutFile $modelTarget
} else {
  Write-Step "Model already present."
}

Write-Step "Done."
Write-Host ("  WASM:  " + $publicWasm)
Write-Host ("  Model: " + $modelTarget)
