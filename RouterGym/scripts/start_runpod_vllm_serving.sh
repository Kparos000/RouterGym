#!/usr/bin/env bash
set -euo pipefail

# Start RouterGym's four vLLM OpenAI-compatible upstreams for RunPod validation.
# Logs and PID files are written under /workspace/serving_logs by default.

LOG_DIR="${ROUTERGYM_SERVING_LOG_DIR:-/workspace/serving_logs}"
API_KEY="${ROUTERGYM_OPENAI_API_KEY:-${ROUTERGYM_VLLM_API_KEY:-EMPTY}}"
HOST="${ROUTERGYM_VLLM_HOST:-127.0.0.1}"

SLM1_MODEL="${ROUTERGYM_SLM1_MODEL_ID:-mistralai/Mistral-7B-Instruct-v0.3}"
SLM2_MODEL="${ROUTERGYM_SLM2_MODEL_ID:-meta-llama/Meta-Llama-3-8B-Instruct}"
LLM1_MODEL="${ROUTERGYM_LLM1_MODEL_ID:-mistralai/Mistral-Small-24B-Instruct-2501}"
LLM2_MODEL="${ROUTERGYM_LLM2_MODEL_ID:-Qwen/Qwen2.5-14B-Instruct}"

SLM1_PORT="${ROUTERGYM_SLM1_PORT:-8000}"
SLM2_PORT="${ROUTERGYM_SLM2_PORT:-8102}"
LLM1_PORT="${ROUTERGYM_LLM1_PORT:-8103}"
LLM2_PORT="${ROUTERGYM_LLM2_PORT:-8104}"

SLM1_GPU="${ROUTERGYM_SLM1_GPU:-0}"
SLM2_GPU="${ROUTERGYM_SLM2_GPU:-1}"
LLM1_GPU="${ROUTERGYM_LLM1_GPU:-2}"
LLM2_GPU="${ROUTERGYM_LLM2_GPU:-3}"

mkdir -p "${LOG_DIR}"

start_server() {
  local key="$1"
  local gpu="$2"
  local model="$3"
  local port="$4"
  local log_path="${LOG_DIR}/${key}.log"
  local pid_path="${LOG_DIR}/${key}.pid"

  if [[ -s "${pid_path}" ]] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
    echo "${key} already appears to be running with PID $(cat "${pid_path}")"
    return
  fi

  echo "Starting ${key}: model=${model} gpu=${gpu} port=${port} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup vllm serve "${model}" \
    --host "${HOST}" \
    --port "${port}" \
    --api-key "${API_KEY}" \
    --enable-prefix-caching \
    --prefix-caching-hash-algo sha256 \
    >"${log_path}" 2>&1 &
  echo "$!" >"${pid_path}"
}

start_server "slm1" "${SLM1_GPU}" "${SLM1_MODEL}" "${SLM1_PORT}"
start_server "slm2" "${SLM2_GPU}" "${SLM2_MODEL}" "${SLM2_PORT}"
start_server "llm1" "${LLM1_GPU}" "${LLM1_MODEL}" "${LLM1_PORT}"
start_server "llm2" "${LLM2_GPU}" "${LLM2_MODEL}" "${LLM2_PORT}"

cat <<EOF
Started requested vLLM servers.

PID files:
  ${LOG_DIR}/slm1.pid
  ${LOG_DIR}/slm2.pid
  ${LOG_DIR}/llm1.pid
  ${LOG_DIR}/llm2.pid

Gateway upstream environment for the requested RunPod layout:
  export ROUTERGYM_GATEWAY_SLM1_UPSTREAM_BASE_URL=http://${HOST}:${SLM1_PORT}/v1
  export ROUTERGYM_GATEWAY_SLM2_UPSTREAM_BASE_URL=http://${HOST}:${SLM2_PORT}/v1
  export ROUTERGYM_GATEWAY_LLM1_UPSTREAM_BASE_URL=http://${HOST}:${LLM1_PORT}/v1
  export ROUTERGYM_GATEWAY_LLM2_UPSTREAM_BASE_URL=http://${HOST}:${LLM2_PORT}/v1

Tail logs with:
  tail -f ${LOG_DIR}/slm1.log ${LOG_DIR}/slm2.log ${LOG_DIR}/llm1.log ${LOG_DIR}/llm2.log
EOF
