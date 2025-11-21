#!/usr/bin/env bash
# Setup script for RouterGym (POSIX)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [ ! -d ".venv" ]; then
  python -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r RouterGym/requirements.txt

cat <<'EOF'
vLLM quickstart (after activating the venv):
  export VLLM_ENGINE_URL=http://localhost:8000/v1
  export VLLM_SLM_MODEL_PATH=/path/to/your/slm
  python -m pip install "vllm>=0.4.0"
  vllm serve $VLLM_SLM_MODEL_PATH --host 0.0.0.0 --port 8000 --api-key token
EOF
