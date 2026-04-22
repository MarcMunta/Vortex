$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$composeFile = Join-Path $root "c3_rnt2_ai\docker-compose.yml"

if (-not (Test-Path -LiteralPath $composeFile)) {
  throw "Compose file not found: $composeFile"
}

& docker compose -f $composeFile --profile manual --profile qwen-sglang down --remove-orphans
exit $LASTEXITCODE
