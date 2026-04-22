# Production Execution Runbook

RouterGym now supports chunked, resumable execution over the frozen benchmark
matrix without changing benchmark semantics.

## What the runner does

- Uses the frozen 36-config matrix from `RouterGym.benchmark_spec`
- Processes each config in deterministic ticket chunks
- Writes per-chunk JSONL outputs incrementally
- Maintains a per-config JSON manifest
- Skips completed chunks on resume
- Merges completed chunk outputs into final per-config JSONL artifacts

## Default production chunking

- Default chunk size: `100`
- Chunk boundaries are deterministic and based on dataset row order
- Chunk filenames include both chunk index and ticket range

Example chunk naming:

- `chunk_0000__tickets_000000_000099__results.jsonl`
- `chunk_0000__tickets_000000_000099__failures.jsonl`
- `chunk_0000__tickets_000000_000099__metadata.json`

## Manifest behavior

Each config/backend run writes:

- `manifest.json`

The manifest records:

- spec version
- pricing version
- backend name
- config identifier
- total tickets expected
- chunk size
- completed chunks
- failed chunks
- output paths
- timestamps
- overall run status

Resume behavior:

- completed chunks are skipped if their output files still exist
- failed chunks remain listed in the manifest and can be retried
- rerunning the same config/backend path does not restart from zero

## Files produced

Under:

- `RouterGym/results/production_runs/<backend>/<config_identifier>/`

The runner writes:

- `manifest.json`
- `chunks/*.jsonl`
- `chunks/*__metadata.json`
- `merged/<config_identifier>__results_merged.jsonl`
- `merged/<config_identifier>__failures_merged.jsonl`

## Preflight usage

Use the same runner for smaller operational validation runs:

- `--preflight-size 100`
- `--preflight-size 500`
- `--preflight-size 2000`

Example dry run:

```powershell
python -m RouterGym.scripts.run_chunked_benchmark `
  --config-id slm_only__base_slm1__mem_none `
  --preflight-size 100 `
  --chunk-size 40 `
  --dry-run
```

## Controlled backend configuration

The runner preserves existing backend choices and is ready for controlled
OpenAI-compatible serving later.

Common environment variables:

- `ROUTERGYM_MODEL_BACKEND=openai_compatible`
- `ROUTERGYM_OPENAI_BASE_URL=http://127.0.0.1:8000/v1`
- `ROUTERGYM_OPENAI_API_KEY=<token>`

The runner can also take an explicit backend override:

```powershell
python -m RouterGym.scripts.run_chunked_benchmark --backend openai_compatible
```

## How this fits the later RunPod + vLLM phase

- RunPod/vLLM details stay outside the core runner logic
- the runner only needs a stable OpenAI-compatible endpoint
- manifests and chunk outputs allow long runs to resume after interruption
- merged outputs remain suitable for later analysis and dissertation reporting

## Merge-only mode

If chunks already exist, merge them without rerunning tickets:

```powershell
python -m RouterGym.scripts.run_chunked_benchmark `
  --config-id slm_dominant__base_slm1__esc_llm1__mem_rag_bm25 `
  --merge-only
```
