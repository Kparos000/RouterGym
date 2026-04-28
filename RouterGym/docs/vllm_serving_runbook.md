# vLLM Local Four-Model Serving Runbook

This runbook covers the corrected local-serving layout for RouterGym when
`ROUTERGYM_MODEL_BACKEND=openai_compatible`.

RouterGym now sends **all four logical model keys** through one local
OpenAI-compatible gateway:

- `slm1` -> `mistralai/Mistral-7B-Instruct-v0.3`
- `slm2` -> `meta-llama/Meta-Llama-3-8B-Instruct`
- `llm1` -> `mistralai/Mistral-Small-24B-Instruct-2501`
- `llm2` -> `Qwen/Qwen2.5-14B-Instruct`

The gateway exposes one RouterGym-facing endpoint:

- `ROUTERGYM_OPENAI_BASE_URL=http://127.0.0.1:8000/v1`

RouterGym sends the logical key (`slm1`, `slm2`, `llm1`, `llm2`) to the
gateway. The gateway rewrites that to the upstream raw model ID and forwards
the request to the correct local vLLM server.

## Required RouterGym environment

```powershell
$env:ROUTERGYM_MODEL_BACKEND="openai_compatible"
$env:ROUTERGYM_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:ROUTERGYM_OPENAI_API_KEY="replace-with-your-vllm-api-key"
```

Supported aliases still work:

- `ROUTERGYM_VLLM_BASE_URL`
- `ROUTERGYM_VLLM_API_KEY`
- backend alias `vllm_openai`

## Gateway route variables

Set one upstream URL per logical model key:

```powershell
$env:ROUTERGYM_GATEWAY_SLM1_UPSTREAM_BASE_URL="http://127.0.0.1:8101/v1"
$env:ROUTERGYM_GATEWAY_SLM2_UPSTREAM_BASE_URL="http://127.0.0.1:8102/v1"
$env:ROUTERGYM_GATEWAY_LLM1_UPSTREAM_BASE_URL="http://127.0.0.1:8103/v1"
$env:ROUTERGYM_GATEWAY_LLM2_UPSTREAM_BASE_URL="http://127.0.0.1:8104/v1"
```

Optional llm2 replicas:

```powershell
$env:ROUTERGYM_GATEWAY_LLM2_REPLICA_BASE_URLS="http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1"
```

Optional gateway bind controls:

```powershell
$env:ROUTERGYM_GATEWAY_BIND_HOST="127.0.0.1"
$env:ROUTERGYM_GATEWAY_BIND_PORT="8000"
```

## 4-GPU layout

Recommended mapping:

- GPU 0: `slm1`
- GPU 1: `slm2`
- GPU 2: `llm1`
- GPU 3: `llm2`
- gateway: port `8000`

Launch commands:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --host 127.0.0.1 --port 8101 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 127.0.0.1 --port 8102 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-Small-24B-Instruct-2501 \
  --host 127.0.0.1 --port 8103 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8104 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

Start the RouterGym gateway:

```bash
python -m RouterGym.scripts.run_local_openai_gateway
```

## 6-GPU layout

Recommended mapping:

- GPU 0: `slm1`
- GPU 1: `slm2`
- GPU 2: `llm1`
- GPU 3: `llm2`
- GPU 4: `llm2` replica
- GPU 5: `llm2` replica
- gateway: port `8000`

Primary servers:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
  --host 127.0.0.1 --port 8101 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 127.0.0.1 --port 8102 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve mistralai/Mistral-Small-24B-Instruct-2501 \
  --host 127.0.0.1 --port 8103 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8104 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

llm2 replicas:

```bash
CUDA_VISIBLE_DEVICES=4 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8105 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

```bash
CUDA_VISIBLE_DEVICES=5 vllm serve Qwen/Qwen2.5-14B-Instruct \
  --host 127.0.0.1 --port 8106 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

Route the replicas through:

```powershell
$env:ROUTERGYM_GATEWAY_LLM2_REPLICA_BASE_URLS="http://127.0.0.1:8105/v1,http://127.0.0.1:8106/v1"
```

Start the RouterGym gateway:

```bash
python -m RouterGym.scripts.run_local_openai_gateway
```

## Smoke and assertion commands

Smoke each logical key:

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

This assertion fails if:

- any requested model key resolves to HF remote instead of local OpenAI-compatible serving
- any requested model key is missing from gateway `/v1/models`
- any gateway model ID mapping does not match RouterGym’s registry
- any smoke call fails

## Preflight sequence before the corrected rerun

1. Start all vLLM servers.
2. Start the RouterGym gateway on port `8000`.
3. Set RouterGym env vars to the gateway endpoint.
4. Run the four smoke commands above.
5. Run `assert_local_openai_serving`.
6. Run the corrected 100-ticket preflight with quality abort enabled.

## RunPod note

This design keeps RouterGym pointed at one local OpenAI-compatible endpoint while
still letting you place each model server on its own GPU and add llm2 replicas
when the dominant configs need extra throughput.
