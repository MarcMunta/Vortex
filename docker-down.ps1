$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

& docker compose -f docker-compose.yml down
exit $LASTEXITCODE
