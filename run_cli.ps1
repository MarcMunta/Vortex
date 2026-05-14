param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CliArgs = @()
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[run] " + $Message)
}

function Fail([string]$Message, [int]$Code = 1) {
  Write-Host ("[run] ERROR: " + $Message) -ForegroundColor Red
  exit $Code
}

function Test-CommandAvailable([string]$Name) {
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Resolve-CommandPath([string]$Name) {
  $cmd = Get-Command $Name -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) {
    return $cmd.Source
  }
  return $null
}

function Test-PythonExecutable([string]$Path) {
  if (-not $Path -or -not (Test-Path -LiteralPath $Path)) {
    return $false
  }
  try {
    & $Path -c "import sys" *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Test-VortexProfile([string]$PythonExe, [string]$Profile) {
  if (-not $Profile) {
    return $false
  }
  try {
    & $PythonExe -c "import sys; from c3rnt2.config import load_settings; load_settings(sys.argv[1])" $Profile *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Test-TransformersProfileSupport([string]$PythonExe, [string]$Profile) {
  if (-not $Profile) {
    return $true
  }
  try {
    & $PythonExe -c @"
import sys
from c3rnt2.config import load_settings

settings = load_settings(sys.argv[1])
core = settings.get("core", {}) or {}
model = str(core.get("hf_model") or "")
if not model.startswith("google/gemma-4"):
    raise SystemExit(0)

from transformers.models.auto.configuration_auto import CONFIG_MAPPING
raise SystemExit(0 if "gemma4" in CONFIG_MAPPING else 1)
"@ $Profile *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Test-HfProfileCached([string]$PythonExe, [string]$Profile, [string]$BackendDir) {
  if (-not $Profile) {
    return $false
  }
  try {
    & $PythonExe -c @"
import sys
from pathlib import Path
from c3rnt2.config import load_settings
from c3rnt2.model_init import model_cache_status, resolve_cache_dir

settings = load_settings(sys.argv[1])
core = settings.get("core", {}) or {}
model = str(core.get("hf_model") or "").strip()
if not model:
    raise SystemExit(1)
cache_dir = resolve_cache_dir(str(Path(sys.argv[2]) / "data" / "models" / "hf-cache"))
status = model_cache_status(model, cache_dir)
raise SystemExit(0 if bool(status.get("cached")) else 1)
"@ $Profile $BackendDir *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Test-TorchCudaAvailable([string]$PythonExe) {
  try {
    & $PythonExe -c @"
import sys
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
"@ *> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Ensure-TorchCudaBuild([string]$PythonExe) {
  if (-not (Test-CommandAvailable "nvidia-smi")) {
    return
  }
  if (Test-TorchCudaAvailable -PythonExe $PythonExe) {
    return
  }
  Write-Step "Installing CUDA-enabled PyTorch for the detected NVIDIA GPU..."
  & $PythonExe -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch torchvision
  if (-not (Test-TorchCudaAvailable -PythonExe $PythonExe)) {
    Fail "CUDA-enabled PyTorch install did not expose torch.cuda. Check the NVIDIA driver and wheel compatibility."
  }
}

function New-DirectoryIfMissing([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Wait-Port(
  [string]$TargetHost,
  [int]$Port,
  [int]$TimeoutSec = 15
) {
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
  while ([DateTime]::UtcNow -lt $deadline) {
    $client = $null
    try {
      $client = New-Object System.Net.Sockets.TcpClient
      $async = $client.BeginConnect($TargetHost, $Port, $null, $null)
      if ($async.AsyncWaitHandle.WaitOne(500)) {
        $client.EndConnect($async)
        return $true
      }
    } catch {
      # ignore
    } finally {
      try { if ($client) { $client.Close() } } catch {}
    }
    Start-Sleep -Milliseconds 250
  }
  return $false
}

function Get-ListeningPid([int]$Port) {
  try {
    $netstatMatches = netstat -ano | Select-String -Pattern (":$Port\s")
    foreach ($m in $netstatMatches) {
      $line = $m.Line
      if (-not $line) { continue }
      if ($line -notmatch "\sLISTENING\s") { continue }
      $parts = ($line -split "\s+") | Where-Object { $_ }
      if ($parts.Count -lt 2) { continue }
      $pidStr = $parts[-1]
      $outPid = 0
      if ([int]::TryParse($pidStr, [ref]$outPid) -and $outPid -gt 0) {
        return $outPid
      }
    }
  } catch {
    return $null
  }
  return $null
}

function Test-PortFree([string]$Name, [int]$Port) {
  $existingPid = Get-ListeningPid -Port $Port
  if ($existingPid) {
    Fail "$Name port $Port is already in use (pid=$existingPid). Run .\\stop.bat or set VORTEX_*_PORT."
  }
}

function Find-ChromeExe() {
  $cmd = Get-Command "chrome.exe" -ErrorAction SilentlyContinue
  if ($cmd -and $cmd.Source) { return $cmd.Source }

  $candidates = @(
    (Join-Path ($env:ProgramFiles) "Google\\Chrome\\Application\\chrome.exe"),
    (Join-Path (${env:ProgramFiles(x86)}) "Google\\Chrome\\Application\\chrome.exe"),
    (Join-Path ($env:LocalAppData) "Google\\Chrome\\Application\\chrome.exe")
  ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

  return ($candidates | Select-Object -First 1)
}

function Open-UrlInBrowser([string]$Url) {
  $chrome = Find-ChromeExe
  if ($chrome) {
    Start-Process -FilePath $chrome -ArgumentList @($Url) | Out-Null
    return
  }
  Start-Process -FilePath $Url | Out-Null
}

function Start-LoggedProcess(
  [string]$Name,
  [string]$WorkingDir,
  [string]$FilePath,
  [string[]]$ArgumentList = @(),
  [hashtable]$Environment = @{},
  [string]$LogPath,
  [string]$ErrorLogPath,
  [string]$PidPath,
  [int]$Port = 0,
  [int]$StartupTimeoutSec = 0
) {
  New-DirectoryIfMissing (Split-Path -Parent $LogPath)
  if ($ErrorLogPath) {
    New-DirectoryIfMissing (Split-Path -Parent $ErrorLogPath)
  }
  New-DirectoryIfMissing (Split-Path -Parent $PidPath)
  if (Test-Path -LiteralPath $LogPath) { Remove-Item -Force -LiteralPath $LogPath -ErrorAction SilentlyContinue }
  if ($ErrorLogPath -and (Test-Path -LiteralPath $ErrorLogPath)) {
    Remove-Item -Force -LiteralPath $ErrorLogPath -ErrorAction SilentlyContinue
  }

  $previousEnv = @{}
  foreach ($entry in ($Environment.GetEnumerator() | Sort-Object Key)) {
    $previousEnv[$entry.Key] = [System.Environment]::GetEnvironmentVariable($entry.Key, "Process")
    [System.Environment]::SetEnvironmentVariable($entry.Key, [string]$entry.Value, "Process")
  }

  try {
    $startArgs = @{
      FilePath = $FilePath
      WorkingDirectory = $WorkingDir
      ArgumentList = $ArgumentList
      RedirectStandardOutput = $LogPath
      PassThru = $true
      WindowStyle = "Hidden"
    }
    if ($ErrorLogPath) {
      $startArgs.RedirectStandardError = $ErrorLogPath
    }
    $proc = Start-Process @startArgs
  } finally {
    foreach ($entry in $previousEnv.GetEnumerator()) {
      [System.Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
  }

  Set-Content -LiteralPath $PidPath -Value $proc.Id -Encoding ascii
  Write-Step "$Name started (pid=$($proc.Id))"

  if ($Port -gt 0 -and $StartupTimeoutSec -gt 0) {
    Wait-Port -TargetHost "127.0.0.1" -Port $Port -TimeoutSec $StartupTimeoutSec | Out-Null
  }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Defaults
$runBack = $true
$runFront = $true
$runControl = $true
$runSelfTrain = (($env:ENABLE_SELF_TRAIN -as [string]) -eq "1")
$runAutoEdits = (($env:ENABLE_AUTO_EDITS -as [string]) -eq "1")
$openBrowser = (($env:VORTEX_OPEN_BROWSER -as [string]) -ne "0")

foreach ($arg in ($CliArgs | Where-Object { $null -ne $_ } | ForEach-Object { $_.Trim() })) {
  switch -Regex ($arg) {
    '^--all$' { }
    '^--front-only$' { $runBack = $false; $runFront = $true }
    '^--back-only$' { $runBack = $true; $runFront = $false }
    '^--no-control$' { $runControl = $false }
    '^--no-self-train$' { $runSelfTrain = $false }
    '^--no-auto-edits$' { $runAutoEdits = $false }
    '^--no-open-browser$' { $openBrowser = $false }
    '^--help$' {
      @"
Vortex one-command runner (Windows)

Usage:
  .\run.bat [--all] [--front-only|--back-only] [--no-control] [--no-self-train] [--no-auto-edits] [--no-open-browser]

Env:
  C3RNT2_PROFILE=rtx4080_16gb_llama2_7b_q4_local
  VORTEX_BACKEND_PORT=8000
  VORTEX_FRONTEND_PORT=5173
  ENABLE_SELF_TRAIN=1
  ENABLE_AUTO_EDITS=1
  VORTEX_OPEN_BROWSER=0
"@ | Write-Host
      exit 0
    }
    default { if ($arg) { Fail "Unknown arg: $arg" } }
  }
}

$defaultModelProfile = "rtx4080_16gb_llama2_7b_q4_local"

$modelProfile = ($env:C3RNT2_PROFILE -as [string])
if ($modelProfile) { $modelProfile = $modelProfile.Trim() }
if (-not $modelProfile) { $modelProfile = $defaultModelProfile }

$backendPort = $env:VORTEX_BACKEND_PORT
if (-not $backendPort) { $backendPort = $env:BACKEND_PORT }
if (-not $backendPort) { $backendPort = "8000" }

$frontendPort = $env:VORTEX_FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = $env:FRONTEND_PORT }
if (-not $frontendPort) { $frontendPort = "5173" }

$controlPort = $env:VORTEX_CONTROL_PORT
if (-not $controlPort) { $controlPort = $env:CONTROL_PORT }
if (-not $controlPort) { $controlPort = "8765" }

$trainingProfile = ($env:VORTEX_TRAINING_PROFILE -as [string])
if ($trainingProfile) { $trainingProfile = $trainingProfile.Trim() }

$logsDir = Join-Path $root "logs"
$pidsDir = Join-Path $root ".pids"
New-DirectoryIfMissing $logsDir
New-DirectoryIfMissing $pidsDir

$needPython = $runBack -or $runSelfTrain -or $runAutoEdits
$needNode = $runFront

if ($needPython -and -not (Test-CommandAvailable "python")) { Fail "Python not found in PATH." }
if ($needNode -and (-not (Test-CommandAvailable "node") -or -not (Test-CommandAvailable "npm"))) { Fail "Node.js/npm not found in PATH." }

# Python venv
$venvDir = Join-Path $root ".venv"
$py = Join-Path $root ".venv\\Scripts\\python.exe"
if ($needPython) {
  if (-not (Test-PythonExecutable $py)) {
    if (Test-Path -LiteralPath $venvDir) {
      Write-Step "Rebuilding invalid venv (.venv)..."
      python -m venv --clear .venv
    } else {
      Write-Step "Creating venv (.venv)..."
      python -m venv .venv
    }
  }
  if (-not (Test-PythonExecutable $py)) {
    Fail "Python venv bootstrap failed: $py"
  }
  Write-Step "Checking Python deps..."
  & $py -c "import importlib.util as u; import sys; mods=['c3rnt2','fastapi','uvicorn','pytest','transformers','huggingface_hub','PIL','torchvision','llama_cpp']; miss=[m for m in mods if u.find_spec(m) is None]; sys.exit(0 if not miss else 1)" 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Step "Installing backend deps (editable + api + hf + llama_cpp extras)..."
    & $py -m pip install -U pip
    & $py -m pip install -e "c3_rnt2_ai[api,hf,llama_cpp]" pytest
  }
  Ensure-TorchCudaBuild -PythonExe $py
  if (-not (Test-VortexProfile -PythonExe $py -Profile $modelProfile)) {
    if ($modelProfile -ne $defaultModelProfile) {
      Write-Step "Profile '$modelProfile' not found; falling back to '$defaultModelProfile'."
      $modelProfile = $defaultModelProfile
    }
  }
  if (-not (Test-VortexProfile -PythonExe $py -Profile $modelProfile)) {
    Fail "Vortex profile not found: $modelProfile"
  }
  if (-not (Test-TransformersProfileSupport -PythonExe $py -Profile $modelProfile)) {
    Write-Step "Installing mainline Transformers support..."
    & $py -m pip install --upgrade "git+https://github.com/huggingface/transformers.git"
  }
}

# Frontend deps
if ($runFront) {
  $frontendDir = Join-Path $root "vortex-chat"
  if (-not (Test-Path -LiteralPath $frontendDir)) { Fail "Frontend dir not found: $frontendDir" }
  if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "node_modules"))) {
    Write-Step "Installing frontend deps (npm i)..."
    Push-Location $frontendDir
    try { npm i } finally { Pop-Location }
  }
  $frontendEnv = Join-Path $frontendDir ".env.local"
  if (-not (Test-Path -LiteralPath $frontendEnv)) {
    $frontendEnvExample = Join-Path $frontendDir ".env.local.example"
    if (Test-Path -LiteralPath $frontendEnvExample) {
      Copy-Item -Force -LiteralPath $frontendEnvExample -Destination $frontendEnv
    }
  }
}

# Start services
if ($runBack) {
  Test-PortFree -Name "backend" -Port ([int]$backendPort)
  $backendDir = Join-Path $root "c3_rnt2_ai"
  $backendEnv = @{}
  if (Test-HfProfileCached -PythonExe $py -Profile $modelProfile -BackendDir $backendDir) {
    $backendEnv["HF_HUB_OFFLINE"] = "1"
    $backendEnv["TRANSFORMERS_OFFLINE"] = "1"
  }
  Start-LoggedProcess `
    -Name "backend" `
    -WorkingDir $backendDir `
    -FilePath $py `
    -ArgumentList @("-m", "vortex", "serve", "--profile", $modelProfile, "--host", "0.0.0.0", "--port", "$backendPort") `
    -Environment $backendEnv `
    -LogPath (Join-Path $logsDir "backend.out.log") `
    -ErrorLogPath (Join-Path $logsDir "backend.log") `
    -PidPath (Join-Path $pidsDir "backend.pid") `
    -Port ([int]$backendPort) `
    -StartupTimeoutSec 5
}

if ($runControl -and ($runBack -or $runFront)) {
  Test-PortFree -Name "control" -Port ([int]$controlPort)
  $backendDir = Join-Path $root "c3_rnt2_ai"
  $composeFile = Join-Path $backendDir "docker-compose.yml"
  $controlArgs = @(
    "-m", "c3rnt2.control_server",
    "--base-dir", $backendDir,
    "--compose-file", $composeFile,
    "--port", "$controlPort",
    "--api-port", "$backendPort",
    "--frontend-port", "$frontendPort",
    "--api-profile", $modelProfile
  )
  if ($trainingProfile) {
    $controlArgs += @("--training-profile", $trainingProfile)
  }
  Start-LoggedProcess `
    -Name "control" `
    -WorkingDir $backendDir `
    -FilePath $py `
    -ArgumentList $controlArgs `
    -Environment @{ PYTHONPATH = (Join-Path $backendDir "src") } `
    -LogPath (Join-Path $logsDir "vortex-control.log") `
    -ErrorLogPath (Join-Path $logsDir "vortex-control.err.log") `
    -PidPath (Join-Path $pidsDir "control.pid") `
    -Port ([int]$controlPort) `
    -StartupTimeoutSec 20
}

if ($runFront) {
  Test-PortFree -Name "frontend" -Port ([int]$frontendPort)
  $frontendDir = Join-Path $root "vortex-chat"
  $npmCmd = Resolve-CommandPath "npm.cmd"
  if (-not $npmCmd) { $npmCmd = Resolve-CommandPath "npm" }
  if (-not $npmCmd) { Fail "npm executable not found in PATH." }
  Start-LoggedProcess `
    -Name "frontend" `
    -WorkingDir $frontendDir `
    -FilePath $npmCmd `
    -ArgumentList @("run", "dev", "--", "--host", "0.0.0.0", "--port", "$frontendPort") `
    -LogPath (Join-Path $logsDir "frontend.log") `
    -ErrorLogPath (Join-Path $logsDir "frontend.err.log") `
    -PidPath (Join-Path $pidsDir "frontend.pid") `
    -Port ([int]$frontendPort) `
    -StartupTimeoutSec 20
}

if ($runSelfTrain) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  $intervalMin = $env:SELF_TRAIN_INTERVAL_MINUTES
  if (-not $intervalMin) { $intervalMin = "30" }
  Start-LoggedProcess `
    -Name "self-train" `
    -WorkingDir $backendDir `
    -FilePath $py `
    -ArgumentList @("-m", "vortex", "self-train", "--profile", $modelProfile, "--interval-minutes", "$intervalMin") `
    -Environment @{ C3RNT2_NO_NET = "1" } `
    -LogPath (Join-Path $logsDir "self-train.log") `
    -ErrorLogPath (Join-Path $logsDir "self-train.err.log") `
    -PidPath (Join-Path $pidsDir "self-train.pid")
}

if ($runAutoEdits) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  Start-LoggedProcess `
    -Name "auto-edits" `
    -WorkingDir $backendDir `
    -FilePath $py `
    -ArgumentList @("scripts\\auto_edits_watcher.py", "--profile", $modelProfile, "--create-demo-on-start") `
    -Environment @{ C3RNT2_NO_NET = "1"; AUTO_EDITS_CREATE_DEMO = "1" } `
    -LogPath (Join-Path $logsDir "auto-edits.log") `
    -ErrorLogPath (Join-Path $logsDir "auto-edits.err.log") `
    -PidPath (Join-Path $pidsDir "auto-edits.pid")
}

Write-Host ""
Write-Step "Summary"
if ($runBack) { Write-Host ("  Backend:  http://localhost:" + $backendPort) }
if ($runControl -and ($runBack -or $runFront)) { Write-Host ("  Control:  http://localhost:" + $controlPort) }
if ($runFront) { Write-Host ("  Frontend: http://localhost:" + $frontendPort) }
Write-Host ("  Logs:     " + $logsDir)
Write-Host ("  PIDs:     " + $pidsDir)
Write-Host ""
Write-Host "Use .\\status.bat to check, .\\logs.bat backend|frontend|self-train|auto-edits to tail, .\\stop.bat to stop."

if ($runFront -and $openBrowser) {
  $url = ("http://localhost:" + $frontendPort + "/")
  if (-not (Wait-Port -TargetHost "127.0.0.1" -Port ([int]$frontendPort) -TimeoutSec 15)) {
    Write-Step "Frontend not ready yet; opening anyway: $url (check logs if needed)"
  } else {
    Write-Step "Opening frontend in browser: $url"
  }
  Open-UrlInBrowser -Url $url
}
