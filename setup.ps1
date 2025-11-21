# Setup script for RouterGym (Windows PowerShell)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

. ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
python -m pip install -r "RouterGym/requirements.txt"

@"
vLLM quickstart (after activating the venv):
  setx VLLM_ENGINE_URL http://localhost:8000/v1
  setx VLLM_SLM_MODEL_PATH C:\path\to\your\slm
  python -m pip install "vllm>=0.4.0"
  vllm serve $Env:VLLM_SLM_MODEL_PATH --host 0.0.0.0 --port 8000 --api-key token
"@ | Write-Host
