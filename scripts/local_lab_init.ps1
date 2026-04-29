param(
    [string]$Profile = "local_learning_lab_4080",
    [string]$ApiProfile = "rtx4080_16gb_programming_qwen_coder_local",
    [string]$TrainingProfile = "rtx4080_16gb_programming_qwen_coder_train_docker",
    [int]$FrontendPort = 4173,
    [int]$ApiPort = 8000,
    [int]$ControlPort = 8765,
    [switch]$InitOnly,
    [switch]$SkipFrontend,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$startScript = Join-Path $repoRoot "scripts\start_local_stack.ps1"
$pythonCandidates = @(
    (Join-Path $repoRoot ".venv\Scripts\python.exe"),
    (Join-Path $repoRoot "c3_rnt2_ai\.venv\Scripts\python.exe")
)

if ($InitOnly) {
    $python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $python) {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if ($pythonCmd -and $pythonCmd.Source) {
            $python = $pythonCmd.Source
        } else {
            throw "Python was not found."
        }
    }

    $backendRoot = Join-Path $repoRoot "c3_rnt2_ai"
    $env:PYTHONPATH = Join-Path $backendRoot "src"
    Push-Location $backendRoot
    try {
        & $python -m c3rnt2.cli local-lab init --profile $Profile
    } finally {
        Pop-Location
    }
    exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $startScript)) {
    throw "Missing start_local_stack.ps1"
}

$args = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $startScript,
    "-LabProfile", $Profile,
    "-ApiProfile", $ApiProfile,
    "-TrainingProfile", $TrainingProfile,
    "-FrontendPort", [string]$FrontendPort,
    "-ApiPort", [string]$ApiPort,
    "-ControlPort", [string]$ControlPort
)

if ($SkipFrontend) { $args += "-SkipFrontend" }
if ($NoBrowser) { $args += "-NoBrowser" }
if ($DryRun) { $args += "-DryRun" }

& powershell @args

exit $LASTEXITCODE
