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
  [string]$CommandLine,
  [string]$LogPath,
  [string]$PidPath,
  [int]$Port = 0
) {
  New-DirectoryIfMissing (Split-Path -Parent $LogPath)
  New-DirectoryIfMissing (Split-Path -Parent $PidPath)
  if (Test-Path -LiteralPath $LogPath) { Remove-Item -Force -LiteralPath $LogPath -ErrorAction SilentlyContinue }

  $cmd = "cd /d `"$WorkingDir`" && $CommandLine > `"$LogPath`" 2>&1"
  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WindowStyle Hidden -PassThru
  $pidToRecord = $proc.Id
  if ($Port -gt 0) {
    if (Wait-Port -TargetHost "127.0.0.1" -Port $Port -TimeoutSec 15) {
      $listenPid = Get-ListeningPid -Port $Port
      if ($listenPid) { $pidToRecord = $listenPid }
    }
  }
  Set-Content -LiteralPath $PidPath -Value $pidToRecord -Encoding ascii
  Write-Step "$Name started (pid=$pidToRecord)"
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Defaults
$runBack = $true
$runFront = $true
$runSelfTrain = (($env:ENABLE_SELF_TRAIN -as [string]) -eq "1")
$runAutoEdits = (($env:ENABLE_AUTO_EDITS -as [string]) -eq "1")
$openBrowser = (($env:VORTEX_OPEN_BROWSER -as [string]) -ne "0")

foreach ($arg in ($CliArgs | Where-Object { $null -ne $_ } | ForEach-Object { $_.Trim() })) {
  switch -Regex ($arg) {
    '^--all$' { }
    '^--front-only$' { $runBack = $false; $runFront = $true }
    '^--back-only$' { $runBack = $true; $runFront = $false }
    '^--no-self-train$' { $runSelfTrain = $false }
    '^--no-auto-edits$' { $runAutoEdits = $false }
    '^--no-open-browser$' { $openBrowser = $false }
    '^--help$' {
      @"
Vortex one-command runner (Windows)

Usage:
  .\run.bat [--all] [--front-only|--back-only] [--no-self-train] [--no-auto-edits] [--no-open-browser]

Env:
  C3RNT2_PROFILE=dev_small
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

$modelProfile = ($env:C3RNT2_PROFILE -as [string])
if ($modelProfile) { $modelProfile = $modelProfile.Trim() }
if (-not $modelProfile) { $modelProfile = "dev_small" }

$backendPort = $env:VORTEX_BACKEND_PORT
if (-not $backendPort) { $backendPort = $env:BACKEND_PORT }
if (-not $backendPort) { $backendPort = "8000" }

$frontendPort = $env:VORTEX_FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = $env:FRONTEND_PORT }
if (-not $frontendPort) { $frontendPort = "5173" }

$logsDir = Join-Path $root "logs"
$pidsDir = Join-Path $root ".pids"
New-DirectoryIfMissing $logsDir
New-DirectoryIfMissing $pidsDir

$needPython = $runBack -or $runSelfTrain -or $runAutoEdits
$needNode = $runFront

if ($needPython -and -not (Test-CommandAvailable "python")) { Fail "Python not found in PATH." }
if ($needNode -and (-not (Test-CommandAvailable "node") -or -not (Test-CommandAvailable "npm"))) { Fail "Node.js/npm not found in PATH." }

# Python venv
$py = Join-Path $root ".venv\\Scripts\\python.exe"
if ($needPython) {
  if (-not (Test-Path -LiteralPath $py)) {
    Write-Step "Creating venv (.venv)..."
    python -m venv .venv
  }
  Write-Step "Checking Python deps..."
  & $py -c "import importlib.util as u; import sys; mods=['c3rnt2','fastapi','uvicorn','pytest']; miss=[m for m in mods if u.find_spec(m) is None]; sys.exit(0 if not miss else 1)" 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Step "Installing backend deps (editable + api extras)..."
    & $py -m pip install -U pip
    & $py -m pip install -e "c3_rnt2_ai[api]" pytest
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
  Start-LoggedProcess `
    -Name "backend" `
    -WorkingDir $backendDir `
    -CommandLine "`"$py`" -m vortex serve --profile $modelProfile --host 0.0.0.0 --port $backendPort" `
    -LogPath (Join-Path $logsDir "backend.log") `
    -PidPath (Join-Path $pidsDir "backend.pid") `
    -Port ([int]$backendPort)
}

if ($runFront) {
  Test-PortFree -Name "frontend" -Port ([int]$frontendPort)
  $frontendDir = Join-Path $root "vortex-chat"
  Start-LoggedProcess `
    -Name "frontend" `
    -WorkingDir $frontendDir `
    -CommandLine "npm run dev -- --host 0.0.0.0 --port $frontendPort" `
    -LogPath (Join-Path $logsDir "frontend.log") `
    -PidPath (Join-Path $pidsDir "frontend.pid") `
    -Port ([int]$frontendPort)
}

if ($runSelfTrain) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  $intervalMin = $env:SELF_TRAIN_INTERVAL_MINUTES
  if (-not $intervalMin) { $intervalMin = "30" }
  Start-LoggedProcess `
    -Name "self-train" `
    -WorkingDir $backendDir `
    -CommandLine "set C3RNT2_NO_NET=1 && `"$py`" -m vortex self-train --profile $modelProfile --interval-minutes $intervalMin" `
    -LogPath (Join-Path $logsDir "self-train.log") `
    -PidPath (Join-Path $pidsDir "self-train.pid")
}

if ($runAutoEdits) {
  $backendDir = Join-Path $root "c3_rnt2_ai"
  Start-LoggedProcess `
    -Name "auto-edits" `
    -WorkingDir $backendDir `
    -CommandLine "set C3RNT2_NO_NET=1 && set AUTO_EDITS_CREATE_DEMO=1 && `"$py`" scripts\\auto_edits_watcher.py --profile $modelProfile --create-demo-on-start" `
    -LogPath (Join-Path $logsDir "auto-edits.log") `
    -PidPath (Join-Path $pidsDir "auto-edits.pid")
}

Write-Host ""
Write-Step "Summary"
if ($runBack) { Write-Host ("  Backend:  http://localhost:" + $backendPort) }
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
