# vLLM Serving Runbook

This runbook covers the dedicated larger-model serving path for RouterGym.

## Purpose

Use a dedicated OpenAI-compatible endpoint for `llm1` and `llm2` so production LLM runs do not depend on Hugging Face Inference Providers.

Current larger-model mappings:

- `llm1` -> `mistralai/Mistral-Small-24B-Instruct-2501`
- `llm2` -> `Qwen/Qwen2.5-14B-Instruct`

## Required environment variables

Set the backend:

```powershell
$env:ROUTERGYM_MODEL_BACKEND="openai_compatible"
```

Set the OpenAI-compatible base URL and API key:

```powershell
$env:ROUTERGYM_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:ROUTERGYM_OPENAI_API_KEY="replace-with-your-vllm-api-key"
```

Supported aliases:

- `ROUTERGYM_VLLM_BASE_URL`
- `ROUTERGYM_VLLM_API_KEY`
- backend alias: `vllm_openai`

RouterGym normalizes the base URL to include `/v1` when needed.

## Example vLLM server commands

Serve `llm1`:

```bash
vllm serve mistralai/Mistral-Small-24B-Instruct-2501 \
  --api-key replace-with-your-vllm-api-key
```

Serve `llm2`:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --api-key replace-with-your-vllm-api-key
```

The vLLM OpenAI-compatible server listens on `http://localhost:8000` by default unless you override host or port.

## Prefix caching for the final benchmark

For the final long benchmark run, enable vLLM automatic prefix caching.

Why this is low risk:

- it is a serving-side KV-cache reuse optimization
- it does not change RouterGym prompts, routing, pricing, or benchmark semantics
- it only helps when many requests share the same prompt prefix

This fits RouterGym well because benchmark requests share stable prompt scaffolding
such as task instructions, schema hints, and other repeated prompt boilerplate.

Recommended launch command for `llm1`:

```bash
vllm serve mistralai/Mistral-Small-24B-Instruct-2501 \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

Recommended launch command for `llm2`:

```bash
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --api-key replace-with-your-vllm-api-key \
  --enable-prefix-caching \
  --prefix-caching-hash-algo sha256
```

Notes:

- `--enable-prefix-caching` turns on vLLM automatic prefix caching
- `--prefix-caching-hash-algo sha256` makes the safe default explicit
- this optimization mainly improves the prompt/prefill phase; it does not reduce decode time
- if prompts do not share prefixes, the benefit will be limited
- no RouterGym code change is required to use this; it is a launch-time serving setting

## RouterGym smoke test

Dry run:

```powershell
python -m RouterGym.scripts.smoke_openai_compatible_model --model llm1 --dry-run
```

Live smoke call:

```powershell
python -m RouterGym.scripts.smoke_openai_compatible_model --model llm2
```

## RunPod / VS Code workflow

Recommended pattern:

1. Create a RunPod Pod with enough VRAM for the target model.
2. Attach a persistent volume or network volume for model weights and cache.
3. Start the vLLM server inside the Pod.
4. Connect from VS Code using Remote SSH.
5. Point RouterGym to the forwarded or public OpenAI-compatible base URL.

Suggested environment on the development side:

```powershell
$env:ROUTERGYM_MODEL_BACKEND="openai_compatible"
$env:ROUTERGYM_OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
$env:ROUTERGYM_OPENAI_API_KEY="replace-with-your-vllm-api-key"
```

## Operational note

Keep `hf_inference` available for small-model development and lightweight fallback work. Use `openai_compatible` for the dedicated larger-model serving path.
