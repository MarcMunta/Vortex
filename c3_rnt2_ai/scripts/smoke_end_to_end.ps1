param(
  [string]$ProfileCore = "rtx4080_16gb_safe",
  [string]$ProfileHf = "rtx4080_16gb_safe_windows_hf",
  [string]$ProfileLlamaCpp = "rtx4080_16gb_safe_windows_llama_cpp"
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

python -m vortex doctor --deep --profile $ProfileCore
python -m vortex bench --profile $ProfileCore --max-new-tokens 64

python -m vortex doctor --deep --profile $ProfileHf
python -m vortex bench --profile $ProfileHf --max-new-tokens 64

try {
  python -m vortex doctor --deep --profile $ProfileLlamaCpp
  python -m vortex bench --profile $ProfileLlamaCpp --max-new-tokens 64
} catch {
  Write-Host "NOTE: llama_cpp smoke skipped (GGUF model missing or llama-cpp-python not installed)."
}
