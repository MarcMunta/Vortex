param(
    [string]$Url = "http://127.0.0.1:4173",
    [string]$BrowserPath = ""
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Windows.Forms

function Resolve-BrowserPath([string]$ExplicitPath) {
    if ($ExplicitPath -and (Test-Path -LiteralPath $ExplicitPath)) {
        return (Resolve-Path $ExplicitPath).Path
    }

    $candidates = @(
        "$Env:ProgramFiles (x86)\Microsoft\Edge\Application\msedge.exe",
        "$Env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
        "$Env:ProgramFiles\Google\Chrome\Application\chrome.exe",
        "$Env:ProgramFiles (x86)\Google\Chrome\Application\chrome.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    throw "No Edge or Chrome installation was found."
}

$screens = [System.Windows.Forms.Screen]::AllScreens
if (-not $screens -or $screens.Count -lt 2) {
    throw "A secondary display was not detected."
}

$targetScreen = $screens | Where-Object { -not $_.Primary } | Select-Object -First 1
if (-not $targetScreen) {
    throw "A non-primary display was not detected."
}

$bounds = $targetScreen.WorkingArea
$browser = Resolve-BrowserPath -ExplicitPath $BrowserPath
$args = @(
    "--new-window",
    "--window-position=$($bounds.X),$($bounds.Y)",
    "--window-size=$($bounds.Width),$($bounds.Height)",
    $Url
)

Write-Host "[vortex-qa] Launching $browser"
Write-Host "[vortex-qa] Secondary display bounds: $($bounds.X),$($bounds.Y) $($bounds.Width)x$($bounds.Height)"
Write-Host "[vortex-qa] URL: $Url"

Start-Process -FilePath $browser -ArgumentList $args
