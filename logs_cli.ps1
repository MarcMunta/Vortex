param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CliArgs = @()
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$logsDir = Join-Path $root "logs"

if (-not (Test-Path -LiteralPath $logsDir)) {
  Write-Host "[logs] No logs/ directory."
  exit 1
}

$name = ($CliArgs | Select-Object -First 1)
if (-not $name) {
  @"
[logs] Usage:
  .\logs.bat backend|frontend|self-train|auto-edits
"@ | Write-Host
  exit 0
}

$map = @{
  "backend" = "backend.log"
  "frontend" = "frontend.log"
  "self-train" = "self-train.log"
  "auto-edits" = "auto-edits.log"
}

if (-not $map.ContainsKey($name)) {
  Write-Host "[logs] Unknown service: $name"
  exit 1
}

$path = Join-Path $logsDir $map[$name]
if (-not (Test-Path -LiteralPath $path)) {
  Write-Host "[logs] Log not found: $path"
  exit 1
}

Write-Host "[logs] Tailing $path (Ctrl+C to stop)..."
Get-Content -LiteralPath $path -Tail 100 -Wait
