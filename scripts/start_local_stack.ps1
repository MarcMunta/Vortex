param(
    [string]$LabProfile = "local_learning_lab_4080",
    [string]$ApiProfile = "rtx4080_16gb_programming_runtime_docker",
    [string]$TrainingProfile = "rtx4080_16gb_programming_train_docker",
    [int]$FrontendPort = 4173,
    [int]$ApiPort = 8000,
    [int]$ControlPort = 8765,
    [switch]$RunLabInit,
    [switch]$SkipFrontend,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
    Write-Host ("[local-stack] " + $Message)
}

function Write-Warn([string]$Message) {
    Write-Host ("[local-stack] WARN: " + $Message) -ForegroundColor Yellow
}

function Fail([string]$Message) {
    throw ("[local-stack] " + $Message)
}

function Test-CommandAvailable([string]$Name) {
    return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
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

function Ensure-Venv([string]$VenvDir) {
    $pythonPath = Join-Path $VenvDir "Scripts\python.exe"
    if (Test-PythonExecutable $pythonPath) {
        return (Resolve-Path $pythonPath).Path
    }
    if (Test-Path -LiteralPath $VenvDir) {
        Write-Warn "Rebuilding invalid venv at $VenvDir"
        python -m venv --clear $VenvDir
    } else {
        Write-Step "Creating venv at $VenvDir"
        python -m venv $VenvDir
    }
    if (-not (Test-PythonExecutable $pythonPath)) {
        Fail "Python venv bootstrap failed: $pythonPath"
    }
    return (Resolve-Path $pythonPath).Path
}

function Resolve-PythonExecutable([string]$RepoRoot, [string]$BackendRoot) {
    $venvDirs = @(
        (Join-Path $RepoRoot ".venv"),
        (Join-Path $BackendRoot ".venv")
    )
    foreach ($venvDir in $venvDirs) {
        $candidate = Join-Path $venvDir "Scripts\python.exe"
        if (Test-PythonExecutable $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }
    foreach ($venvDir in $venvDirs) {
        if ($venvDir -and (Test-Path -LiteralPath $venvDir)) {
            return Ensure-Venv -VenvDir $venvDir
        }
    }
    $preferredVenvDir = $venvDirs | Select-Object -First 1
    if ($preferredVenvDir) {
        return Ensure-Venv -VenvDir $preferredVenvDir
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd -and $pythonCmd.Source) {
        return $pythonCmd.Source
    }
    Fail "Python was not found. Create a venv first or add python to PATH."
}

function New-DirectoryIfMissing([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
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

function Stop-ProcessTree([int]$ProcessId, [string]$Reason = "restarting") {
    if (-not $ProcessId) { return }
    try {
        Write-Warn "Stopping process $ProcessId ($Reason)."
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 800
    } catch {
        Write-Warn "Failed to stop process $ProcessId cleanly."
    }
}

function Wait-HttpOk([string]$Uri, [int]$TimeoutSec = 30) {
    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSec)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec 2
            if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 300) {
                return $true
            }
        } catch {
            Start-Sleep -Milliseconds 750
        }
    }
    return $false
}

function Ensure-FrontendDeps([string]$FrontendRoot) {
    if (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules")) {
        return
    }
    Write-Step "Installing frontend dependencies..."
    Push-Location $FrontendRoot
    try {
        npm install
        if ($LASTEXITCODE -ne 0) {
            Fail "npm install failed for vortex-chat."
        }
    } finally {
        Pop-Location
    }
}

function Start-LoggedFrontend([string]$FrontendRoot, [string]$LogPath, [int]$Port) {
    $existingPid = Get-ListeningPid -Port $Port
    if ($existingPid) {
        Write-Step "Frontend already listening on port $Port (pid=$existingPid)."
        return
    }
    $cmd = "cd /d `"$FrontendRoot`" && npm run dev -- --host 0.0.0.0 --port $Port > `"$LogPath`" 2>&1"
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WindowStyle Hidden | Out-Null
    Write-Step "Frontend started on http://127.0.0.1:$Port"
}

function Invoke-JsonPost([string]$Uri, [hashtable]$Body) {
    return Invoke-RestMethod -Method Post -Uri $Uri -ContentType "application/json" -Body ($Body | ConvertTo-Json -Depth 6)
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$backendRoot = Join-Path $repoRoot "c3_rnt2_ai"
$frontendRoot = Join-Path $repoRoot "vortex-chat"
$composeFile = Join-Path $backendRoot "docker-compose.yml"
$logsDir = Join-Path $repoRoot "logs"
$frontendLog = Join-Path $logsDir "vortex-chat-dev.log"
$controlLog = Join-Path $logsDir "vortex-control.log"
$controlStatusLog = Join-Path $logsDir "vortex-control-status.json"

if (-not (Test-Path -LiteralPath $backendRoot)) {
    Fail "Backend root not found: $backendRoot"
}
if (-not (Test-Path -LiteralPath $composeFile)) {
    Fail "Compose file not found: $composeFile"
}
if ((-not $SkipFrontend) -and (-not (Test-Path -LiteralPath $frontendRoot))) {
    Fail "Frontend root not found: $frontendRoot"
}
if (-not (Test-CommandAvailable "docker")) {
    Fail "docker was not found in PATH."
}
if ((-not $SkipFrontend) -and (-not (Test-CommandAvailable "npm"))) {
    Fail "npm was not found in PATH."
}

$python = Resolve-PythonExecutable -RepoRoot $repoRoot -BackendRoot $backendRoot
New-DirectoryIfMissing $logsDir

if ($RunLabInit) {
    Write-Step "Initializing local-lab layout ($LabProfile)..."
    if (-not $DryRun) {
        $env:PYTHONPATH = Join-Path $backendRoot "src"
        Push-Location $backendRoot
        try {
            & $python -m c3rnt2.cli local-lab init --profile $LabProfile
            if ($LASTEXITCODE -ne 0) {
                Fail "local-lab init failed."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Step "Dry run: would execute local-lab init for $LabProfile"
    }
} else {
    Write-Step "Skipping legacy local-lab init for daily startup."
}

Write-Step "Checking Docker..."
if (-not $DryRun) {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "Docker Desktop is not running."
    }
} else {
    Write-Step "Dry run: would validate Docker Desktop"
}

$controlUrl = "http://127.0.0.1:$ControlPort"
$healthUrl = "$controlUrl/healthz"
$statusUrl = "$controlUrl/control/status"
$bootstrapUrl = "$controlUrl/control/bootstrap"

if (-not $DryRun) {
    $existingControlPid = Get-ListeningPid -Port $ControlPort
    $controlHealthy = $false
    if ($existingControlPid) {
        Write-Step "Control service already listening on port $ControlPort (pid=$existingControlPid)."
        $controlHealthy = Wait-HttpOk -Uri $healthUrl -TimeoutSec 4
        if (-not $controlHealthy) {
            Stop-ProcessTree -Pid $existingControlPid -Reason "stale control service"
            $existingControlPid = $null
        }
    }
    if (-not $existingControlPid) {
        Write-Step "Starting local control service on $controlUrl ..."
        $pyPath = Join-Path $backendRoot "src"
        $cmd = "cd /d `"$backendRoot`" && set PYTHONPATH=`"$pyPath`" && `"$python`" -m c3rnt2.control_server --base-dir `"$backendRoot`" --compose-file `"$composeFile`" --port $ControlPort --api-port $ApiPort --frontend-port $FrontendPort --api-profile `"$ApiProfile`" --training-profile `"$TrainingProfile`" > `"$controlLog`" 2>&1"
        Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $cmd) -WindowStyle Hidden | Out-Null
    }
    if (-not (Wait-HttpOk -Uri $healthUrl -TimeoutSec 180)) {
        $logTail = @()
        if (Test-Path -LiteralPath $controlLog) {
            $logTail = Get-Content -LiteralPath $controlLog -Tail 40
        }
        if ($logTail.Count -gt 0) {
            Write-Warn "Control service log tail:"
            $logTail | ForEach-Object { Write-Host $_ }
        }
        Fail "Control service did not become healthy in time."
    }
    Write-Step "Control service is healthy."
    try {
        $bootstrapResp = Invoke-JsonPost -Uri $bootstrapUrl -Body @{ force = $false; mode = "ensure" }
        Write-Step ("Bootstrap request: " + ($bootstrapResp | ConvertTo-Json -Compress))
    } catch {
        Write-Warn "Failed to trigger bootstrap through control service."
    }
} else {
    Write-Step "Dry run: would start control service on port $ControlPort"
    Write-Step "Dry run: would POST $bootstrapUrl"
}

if (-not $SkipFrontend) {
    if (-not $DryRun) {
        Ensure-FrontendDeps -FrontendRoot $frontendRoot
        Start-LoggedFrontend -FrontendRoot $frontendRoot -LogPath $frontendLog -Port $FrontendPort
    } else {
        Write-Step "Dry run: would start vortex-chat on port $FrontendPort"
    }
}

$readyUrl = "http://127.0.0.1:$ApiPort/readyz"
if (-not $DryRun) {
    Write-Step "Polling control status..."
    $statusPayload = $null
    $deadline = [DateTime]::UtcNow.AddSeconds(180)
    while ([DateTime]::UtcNow -lt $deadline) {
        try {
            $statusPayload = Invoke-RestMethod -Method Get -Uri $statusUrl -TimeoutSec 3
            if ($statusPayload) {
                $statusPayload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $controlStatusLog -Encoding UTF8
                $bootstrapStage = $statusPayload.bootstrap.stage
                $bootstrapMsg = $statusPayload.bootstrap.message
                if ($statusPayload.runtime.readyz.ok) {
                    Write-Step "Backend is reachable."
                    break
                }
                if ($bootstrapStage) {
                    Write-Step ("Bootstrap stage: " + $bootstrapStage + " (" + $bootstrapMsg + ")")
                }
            }
        } catch {
            Start-Sleep -Milliseconds 1200
            continue
        }
        Start-Sleep -Milliseconds 1200
    }
    if (-not (Wait-HttpOk -Uri $readyUrl -TimeoutSec 10)) {
        Write-Warn "Backend is still warming up. The frontend can already show bootstrap progress."
    }
} else {
    Write-Step "Dry run: would poll $statusUrl and $readyUrl"
}

Write-Host ""
Write-Step "Summary"
Write-Host ("  Local-lab profile: " + $LabProfile)
Write-Host ("  API profile:       " + $ApiProfile)
Write-Host ("  Training profile:  " + $TrainingProfile)
Write-Host ("  Control service:   " + $controlUrl)
Write-Host ("  Runtime/API:       http://127.0.0.1:" + $ApiPort)
Write-Host ("  Frontend:          " + ($(if ($SkipFrontend) { "skipped" } else { "http://127.0.0.1:$FrontendPort" })))
Write-Host ("  Compose file:      " + $composeFile)
Write-Host ("  Control log:       " + $controlLog)
Write-Host ("  Control status:    " + $controlStatusLog)
Write-Host ("  Frontend log:      " + $frontendLog)
Write-Host ""
Write-Host "Live logs:"
Write-Host ("  Get-Content -Wait `"$controlLog`"")
Write-Host ("  docker compose -f `"$composeFile`" logs -f sglang-runtime vortex-api")

if ((-not $DryRun) -and (-not $SkipFrontend) -and (-not $NoBrowser)) {
    Start-Process -FilePath ("http://127.0.0.1:" + $FrontendPort + "/") | Out-Null
}
