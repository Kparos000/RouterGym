# Pre-rerun Safety Hardening Report

## Goal

Add automatic post-chunk generation-quality aborts and a reproducibility runtime manifest so a corrected 47k rerun can fail fast instead of silently wasting GPU time.

## Files Changed

- `RouterGym/experiments/chunked_execution.py`
- `RouterGym/scripts/run_chunked_benchmark.py`
- `RouterGym/scripts/check_generation_quality_gate.py`
- `RouterGym/tests/test_chunked_execution.py`
- `RouterGym/tests/test_quality_gate.py`

## Options Added

The chunked benchmark runner now supports:

- `--enable-quality-abort`
- `--quality-check-after-chunks N`
- `--placeholder-answer-max-rate`
- `--empty-steps-max-rate`
- `--min-raw-response-saved-rate`
- `--min-generation-valid-rate`
- `--allow-slm-dominant-full-escalation`

### Default behavior

- Quality abort is **disabled by default**.
- When enabled and no explicit `--quality-check-after-chunks` is given:
  - short/preflight runs check after **every chunk**
  - longer runs check after **every 5 chunks**

## Quality-abort Behavior

When quality abort is enabled, the runner:

1. saves a chunk normally
2. re-evaluates all completed outputs for that config so far
3. applies the configured generation-quality thresholds
4. aborts the config immediately if the gate fails

On failure it now:

- marks the config status as `failed_quality_gate`
- writes:
  - `quality_gate_failure_report.json`
  - `quality_gate_failure_report.md`
- updates the config status file and top-level backend status file
- exits non-zero through the CLI path so orchestration does not treat the worker as successful

### Resume behavior after quality failure

A manifest with `run_status = failed_quality_gate` is **not treated as completed**.

For safety, rerunning with normal resume now raises and requires an explicit fresh rerun with:

- `--no-resume`

This avoids quietly continuing from already-failed partial outputs.

## Runtime Manifest Fields

Each config run now writes `runtime_manifest.json` containing:

- git commit SHA
- git branch
- dirty working tree flag
- command line args
- backend
- config identifier
- chunk size
- model IDs for `slm1`, `slm2`, `llm1`, `llm2`
- Python version and executable
- platform info
- `vllm` version if installed
- `torch` version if installed
- CUDA availability
- CUDA version if available
- visible GPU IDs
- `nvidia-smi` summary if available
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- relevant environment variable names present, without secret values
- `encoder_calibrated_head.npz` path, existence, and SHA256 if present
- quality-gate settings and thresholds used

## Test Results

All requested narrow checks passed:

- `ruff check .`
- `mypy RouterGym`
- `pytest -q RouterGym/tests/test_chunked_execution.py`
- `pytest -q RouterGym/tests/test_quality_gate.py`

Key new test coverage:

- quality abort passes on healthy rows
- quality abort fails on placeholder answers
- quality abort fails on missing raw responses
- quality gate failure writes useful JSON/Markdown reports
- runtime manifest contains git SHA and model IDs
- resume does not treat `failed_quality_gate` as completed

## Exact Commands

### Corrected 100-ticket preflight with quality abort enabled

```bash
python -m RouterGym.scripts.run_chunked_benchmark \
  --config-ids \
    slm_only__base_slm1__mem_rag_bm25 \
    slm_only__base_slm2__mem_rag_bm25 \
    llm_only__base_llm1__mem_rag_bm25 \
    llm_only__base_llm2__mem_rag_bm25 \
    slm_dominant__base_slm1__esc_llm2__mem_rag_bm25 \
    slm_dominant__base_slm2__esc_llm2__mem_rag_bm25 \
  --backend openai_compatible \
  --parallel-workers 6 \
  --gpu-ids 0,1,2,3,4,5 \
  --chunk-size 100 \
  --preflight-size 100 \
  --enable-quality-abort \
  --quality-check-after-chunks 1
```

### Corrected 500-ticket soak with quality abort enabled

```bash
python -m RouterGym.scripts.run_chunked_benchmark \
  --config-ids \
    slm_only__base_slm1__mem_rag_bm25 \
    slm_only__base_slm2__mem_rag_bm25 \
    llm_only__base_llm1__mem_rag_bm25 \
    llm_only__base_llm2__mem_rag_bm25 \
    slm_dominant__base_slm1__esc_llm2__mem_rag_bm25 \
    slm_dominant__base_slm2__esc_llm2__mem_rag_bm25 \
  --backend openai_compatible \
  --parallel-workers 6 \
  --gpu-ids 0,1,2,3,4,5 \
  --chunk-size 100 \
  --preflight-size 500 \
  --enable-quality-abort \
  --quality-check-after-chunks 1
```

If the pod has fewer than 6 GPUs, reduce `--parallel-workers` and `--gpu-ids` accordingly.

## Conclusion

The repository now has two new hard protections before any corrected 47k rerun:

1. automatic early abort on bad generation outputs
2. per-config runtime manifests for reproducibility and postmortem debugging

This is enough to make a corrected **100-ticket preflight** and **500-ticket soak** materially safer before any full rerun.
