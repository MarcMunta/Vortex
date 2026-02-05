param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$CliArgs = @()
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$runner = Join-Path $root "run_cli.ps1"

if (-not (Test-Path -LiteralPath $runner)) {
  Write-Host ("[run] ERROR: Missing runner script: " + $runner) -ForegroundColor Red
  exit 1
}

& $runner @CliArgs
