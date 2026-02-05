$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[stop] " + $Message)
}

function Stop-Tree([int]$ProcessId) {
  & cmd.exe /c "taskkill /PID $ProcessId /T >nul 2>nul"
  Start-Sleep -Milliseconds 300
  & cmd.exe /c "taskkill /PID $ProcessId /T /F >nul 2>nul"
}

function Get-ListeningPids([int]$Port) {
  $pids = @()
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
      if ([int]::TryParse($pidStr, [ref]$outPid) -and $outPid -gt 0) { $pids += $outPid }
    }
  } catch {
    return @()
  }
  return ($pids | Select-Object -Unique)
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidsDir = Join-Path $root ".pids"
if (-not (Test-Path -LiteralPath $pidsDir)) {
  Write-Step "No .pids/ directory."
  exit 0
}

$backendPort = $env:VORTEX_BACKEND_PORT
if (-not $backendPort) { $backendPort = $env:BACKEND_PORT }
if (-not $backendPort) { $backendPort = "8000" }

$frontendPort = $env:VORTEX_FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = $env:FRONTEND_PORT }
if (-not $frontendPort) { $frontendPort = "5173" }

$portMap = @{
  "backend" = [int]$backendPort
  "frontend" = [int]$frontendPort
}

$pidFiles = Get-ChildItem -LiteralPath $pidsDir -Filter *.pid -ErrorAction SilentlyContinue
if (-not $pidFiles) {
  Write-Step "No PID files found."
  exit 0
}

foreach ($pf in $pidFiles) {
  $name = $pf.BaseName
  $raw = (Get-Content -LiteralPath $pf.FullName -ErrorAction SilentlyContinue | Select-Object -First 1)
  $procId = 0
  [int]::TryParse(($raw -as [string]), [ref]$procId) | Out-Null

  $port = $null
  if ($portMap.ContainsKey($name)) { $port = $portMap[$name] }

  if ($procId -gt 0) {
    Write-Step "Stopping $name (pid=$procId)..."
    Stop-Tree -ProcessId $procId
  }

  if ($port) {
    Start-Sleep -Milliseconds 200
    foreach ($lp in (Get-ListeningPids -Port $port)) {
      Write-Step "Stopping $name (port=$port pid=$lp)..."
      Stop-Tree -ProcessId $lp
    }
  }

  Remove-Item -Force -LiteralPath $pf.FullName -ErrorAction SilentlyContinue
}

Write-Step "Done."
