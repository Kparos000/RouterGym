# Local Four-Model Serving Patch Report

Date: 2026-04-27

## Goal

Patch RouterGym so the corrected final benchmark can serve **all four logical
model keys locally** through one OpenAI-compatible/vLLM gateway instead of
routing `slm1`/`slm2` through HF remote while `llm1`/`llm2` use local serving.

## Files changed

- `RouterGym/engines/model_registry.py`
- `RouterGym/engines/openai_compatible.py`
- `RouterGym/engines/local_openai_gateway.py`
- `RouterGym/scripts/run_local_openai_gateway.py`
- `RouterGym/scripts/assert_local_openai_serving.py`
- `RouterGym/scripts/smoke_openai_compatible_model.py`
- `RouterGym/tests/test_openai_compatible_backend.py`
- `RouterGym/tests/test_model_registry.py`
- `RouterGym/tests/test_local_openai_serving.py`
- `RouterGym/docs/vllm_serving_runbook.md`

## Exact logical model mappings

- `slm1` -> `mistralai/Mistral-7B-Instruct-v0.3`
- `slm2` -> `meta-llama/Meta-Llama-3-8B-Instruct`
- `llm1` -> `mistralai/Mistral-Small-24B-Instruct-2501`
- `llm2` -> `Qwen/Qwen2.5-14B-Instruct`

## Backend selection change

Previous behavior under `ROUTERGYM_MODEL_BACKEND=openai_compatible`:

- `slm1` / `slm2` -> HF remote `RemoteInferenceEngine`
- `llm1` / `llm2` -> local `OpenAICompatibleEngine`

Patched behavior under `ROUTERGYM_MODEL_BACKEND=openai_compatible`:

- `slm1` -> local `OpenAICompatibleEngine`
- `slm2` -> local `OpenAICompatibleEngine`
- `llm1` -> local `OpenAICompatibleEngine`
- `llm2` -> local `OpenAICompatibleEngine`

RouterGym now sends the **logical key** (`slm1`, `slm2`, `llm1`, `llm2`) to one
RouterGym-facing gateway endpoint:

- `ROUTERGYM_OPENAI_BASE_URL=http://127.0.0.1:8000/v1`

The gateway rewrites that logical key to the correct upstream raw model ID and
forwards the request to the corresponding local vLLM server.

## Gateway route table

The new route-table layer expects:

- `ROUTERGYM_GATEWAY_SLM1_UPSTREAM_BASE_URL`
- `ROUTERGYM_GATEWAY_SLM2_UPSTREAM_BASE_URL`
- `ROUTERGYM_GATEWAY_LLM1_UPSTREAM_BASE_URL`
- `ROUTERGYM_GATEWAY_LLM2_UPSTREAM_BASE_URL`

Optional replica support:

- `ROUTERGYM_GATEWAY_LLM2_REPLICA_BASE_URLS`

The gateway exposes `/v1/models` with the logical keys and their upstream raw
model IDs so RouterGym can assert the local-serving contract before preflight.

## 4-GPU serving layout

- GPU 0: `slm1`
- GPU 1: `slm2`
- GPU 2: `llm1`
- GPU 3: `llm2`
- gateway: port `8000`

Example commands:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 127.0.0.1 --port 8101 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --host 127.0.0.1 --port 8102 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-Small-24B-Instruct-2501 --host 127.0.0.1 --port 8103 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8104 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
python -m RouterGym.scripts.run_local_openai_gateway
```

## 6-GPU serving layout

- GPU 0: `slm1`
- GPU 1: `slm2`
- GPU 2: `llm1`
- GPU 3: `llm2`
- GPU 4: `llm2` replica
- GPU 5: `llm2` replica
- gateway: port `8000`

Example commands:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.3 --host 127.0.0.1 --port 8101 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --host 127.0.0.1 --port 8102 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-Small-24B-Instruct-2501 --host 127.0.0.1 --port 8103 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8104 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8105 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-14B-Instruct --host 127.0.0.1 --port 8106 --api-key replace-with-your-vllm-api-key --enable-prefix-caching --prefix-caching-hash-algo sha256
python -m RouterGym.scripts.run_local_openai_gateway
```

Set:

```powershell
$env:ROUTERGYM_GATEWAY_LLM2_REPLICA_BASE_URLS="http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1"
```

## Exact smoke and assertion commands before preflight

Set RouterGym to the gateway:

```powershell
$env:ROUTERGYM_MODEL_BACKEND="openai_compatible"
$env:ROUTERGYM_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:ROUTERGYM_OPENAI_API_KEY="replace-with-your-vllm-api-key"
```

Smoke every logical key:

```powershell
python -m RouterGym.scripts.smoke_openai_compatible_model --model slm1
python -m RouterGym.scripts.smoke_openai_compatible_model --model slm2
python -m RouterGym.scripts.smoke_openai_compatible_model --model llm1
python -m RouterGym.scripts.smoke_openai_compatible_model --model llm2
```

Assert the full local-serving contract:

```powershell
python -m RouterGym.scripts.assert_local_openai_serving --models slm1 slm2 llm1 llm2
```

## Estimated runtime implications

These are architecture-level estimates, not a fresh benchmark measurement:

- **4 GPUs**
  - four configs can run concurrently
  - two configs queue behind them
  - expected corrected full-run wall clock: roughly **12 to 14 hours**
- **6 GPUs**
  - all six approved configs can run concurrently
  - expected corrected full-run wall clock: roughly **6.5 to 8 hours**

Caveat:

- the previous 47k run had the generator parser bug, so these are throughput
  estimates only
- corrected `slm_dominant` behavior may reduce universal escalation and shift
  runtime somewhat
- llm2 replicas in the 6-GPU layout should help smooth the heaviest dominant
  path, but RouterGym still uses config-level orchestration rather than
  intra-config tensor parallelism

## Test results

Executed locally:

- `ruff check .` -> passed
- `mypy RouterGym` -> passed
- `pytest -q RouterGym/tests/test_openai_compatible_backend.py` -> `9 passed`
- `pytest -q RouterGym/tests/test_model_registry.py` -> `9 passed`
- `pytest -q RouterGym/tests/test_local_openai_serving.py` -> `4 passed`

## Outcome

RouterGym is now patched so `openai_compatible` can represent a fully local,
four-model serving layout behind one gateway endpoint. This removes the old
unfair split where SLM configs still depended on HF remote serving while LLM
configs were local.
