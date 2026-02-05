param(
  [int]$Tail = 12
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[status] " + $Message)
}

function Read-Pid([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  $raw = (Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue | Select-Object -First 1)
  $outPid = 0
  if ([int]::TryParse(($raw -as [string]), [ref]$outPid) -and $outPid -gt 0) { return $outPid }
  return $null
}

function Test-ProcessAlive([int]$ProcId) {
  try { return $null -ne (Get-Process -Id $ProcId -ErrorAction SilentlyContinue) } catch { return $false }
}

function Test-PortListening([int]$Port) {
  try {
    $out = & netstat -ano | Select-String -Pattern (":" + $Port + "\s.*\sLISTENING\s")
    return $null -ne $out
  } catch {
    return $false
  }
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidsDir = Join-Path $root ".pids"
$logsDir = Join-Path $root "logs"

$backendPort = $env:VORTEX_BACKEND_PORT
if (-not $backendPort) { $backendPort = $env:BACKEND_PORT }
if (-not $backendPort) { $backendPort = "8000" }

$frontendPort = $env:VORTEX_FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = $env:FRONTEND_PORT }
if (-not $frontendPort) { $frontendPort = "5173" }

$services = @(
  @{ name = "backend"; pid = (Read-Pid (Join-Path $pidsDir "backend.pid")); port = [int]$backendPort; log = (Join-Path $logsDir "backend.log") },
  @{ name = "frontend"; pid = (Read-Pid (Join-Path $pidsDir "frontend.pid")); port = [int]$frontendPort; log = (Join-Path $logsDir "frontend.log") },
  @{ name = "self-train"; pid = (Read-Pid (Join-Path $pidsDir "self-train.pid")); port = $null; log = (Join-Path $logsDir "self-train.log") },
  @{ name = "auto-edits"; pid = (Read-Pid (Join-Path $pidsDir "auto-edits.pid")); port = $null; log = (Join-Path $logsDir "auto-edits.log") }
)

Write-Step "Services"
foreach ($svc in $services) {
  $procId = $svc.pid
  $pidDisplay = if ($procId) { $procId } else { "-" }
  $alive = $false
  if ($procId) { $alive = Test-ProcessAlive -ProcId $procId }
  $port = $svc.port
  $listen = $false
  if ($port) { $listen = Test-PortListening -Port $port }
  $status = if ($procId -and $alive) { "RUNNING" } elseif ($procId -and -not $alive) { "STALE_PID" } else { "STOPPED" }
  if ($port) {
    Write-Host ("  {0,-10} {1,-9} pid={2} port={3} listen={4}" -f $svc.name, $status, $pidDisplay, $port, $listen)
  } else {
    Write-Host ("  {0,-10} {1,-9} pid={2}" -f $svc.name, $status, $pidDisplay)
  }
}

Write-Host ""
Write-Step "Logs (tail=$Tail)"
foreach ($svc in $services) {
  $log = $svc.log
  if (-not (Test-Path -LiteralPath $log)) { continue }
  Write-Host ""
  Write-Host ("--- {0} ---" -f $svc.name)
  try { Get-Content -LiteralPath $log -Tail $Tail } catch {}
}
