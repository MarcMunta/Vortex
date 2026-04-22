param(
  [string]$ApiUrl = "http://127.0.0.1:8000",
  [string]$ControlUrl = "http://127.0.0.1:8765",
  [string]$VaultPath = ""
)

$ErrorActionPreference = "Stop"

function Write-Step([string]$Message) {
  Write-Host ("[multimodal-smoke] " + $Message)
}

function Invoke-JsonPost([string]$Url, [hashtable]$Payload) {
  Invoke-RestMethod -Method Post -Uri $Url -ContentType "application/json" -Body ($Payload | ConvertTo-Json -Depth 8)
}

$root = Split-Path -Parent $PSScriptRoot
if (-not $VaultPath) {
  $VaultPath = Join-Path $root "output\obsidian-smoke-vault"
}

New-Item -ItemType Directory -Force -Path $VaultPath | Out-Null

Write-Step "Configuring Obsidian vault..."
$null = Invoke-JsonPost "$ApiUrl/v1/obsidian/config" @{
  enabled = $true
  vault_path = $VaultPath
}
$obsidian = Invoke-RestMethod -Method Get -Uri "$ApiUrl/v1/obsidian/status"

Write-Step "Resetting spatial session..."
$session = Invoke-JsonPost "$ApiUrl/v1/spatial/session" @{
  session_id = "multimodal-smoke"
  selected_object_id = $null
  selected_region = @{
    x = 120
    y = 100
    width = 360
    height = 240
  }
  active_panel_ids = @()
  active_presentation_id = $null
  active_page_index = 0
  interaction_mode = "inspect"
  last_voice_command = $null
  last_gesture_event = $null
  camera_state = $null
  gesture_state = $null
  focused_item = $null
  recent_multimodal_summary = $null
  panels = @()
  updated_at = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  created_at = [double][DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

Write-Step "Opening presentation through voice intent..."
$opened = Invoke-JsonPost "$ApiUrl/v1/voice/transcribe" @{
  text = "open this presentation here"
  language = "en"
}

Write-Step "Saving workspace note to Obsidian..."
$saved = Invoke-JsonPost "$ApiUrl/v1/voice/transcribe" @{
  text = "save this to obsidian"
  language = "en"
}

Write-Step "Reading control multimodal status..."
$control = Invoke-RestMethod -Method Get -Uri "$ControlUrl/control/multimodal/status"

if (-not $opened.ok) { throw "open_panel_failed" }
if (-not $saved.ok) { throw "save_obsidian_failed" }
if (-not $control.ok) { throw "control_multimodal_failed" }

$summary = [pscustomobject]@{
  obsidian_vault = $obsidian.vault_path
  opened_panel_type = $opened.action_result.panel.type
  panel_count = @($opened.action_result.session.panels).Count
  saved_note = $saved.action_result.path
  focused_panel = $control.status.spatial.selected_object_id
  fusion_summary = $control.status.fusion.summary
}

Write-Step "Done."
$summary | ConvertTo-Json -Depth 6
