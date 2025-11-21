# RouterGym

**Dissertation:** From LLM-First to SLM-Dominant: A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems.

RouterGym operationalizes NVIDIA Research's paper "Small Language Models Are the Future of Agents" and asks a focused question: **Can SLM-dominant agent stacks match or outperform LLM-first stacks on cost, speed, groundedness, schema adherence, and accuracy?** This repository provides routing, memory, contract enforcement, and evaluation harnesses to run that comparison end-to-end.

## Why this exists
Large models are expensive and slow, while small language models (SLMs) are cheap, fast, and easy to self-host. The emerging pattern is to route most queries to SLMs and only escalate to an LLM when necessary. RouterGym implements and benchmarks that pattern with configurable routers, memories, contracts, and evaluation metrics so the SLM-vs-LLM trade-off is measured, not guessed.

## System architecture
```
User Prompt
   |
   v
+----------+        +------------------+        +-------------------+
| Classify | -----> | Memory (optional)| ----> | Contract / Schema |
+----------+        +------------------+        +-------------------+
      |                        |                         |
      |                        v                         v
      |                +---------------+          +--------------+
      +--------------> | Response Gen. | <------> | Validator /  |
                       |   (SLM/LLM)   |          | Fallback     |
                       +---------------+          +--------------+
```
- Classification chooses SLM-first, LLM-first, or hybrid specialist routes.
- Memory injects none / static / dynamic / salience-gated RAG context before generation.
- Response is produced under JSON contracts; invalid outputs are retried or escalated.

## RouterGym in detail
- Routing strategies
  - LLM-first: default to the strongest LLM; optionally downshift to SLM on cheap paths.
  - SLM-dominant: prefer SLM; escalate to LLM on low-confidence, contract failure, or safety triggers.
  - Hybrid specialist: route by domain or task class to specialist SLMs with an LLM safety net.
- Memory systems: none -> static context -> dynamic retrieved context -> salience-gated RAG (scores relevance, trims, then injects).
- Supported models: configure any 2 SLMs and 2 LLMs (e.g., Phi-3 / Mistral for SLM; GPT-4 / Claude for LLM) via provider adapters.
- Contract enforcement: JSON schema validation plus type coercion; retries/fallbacks on invalid outputs; deterministic schemas per task.
- Evaluation metrics: groundedness, schema validity, latency, cost, fallback rate, and task accuracy (task-specific scorers).

## Repository structure
- `agents/`: Agent wrappers around models (SLM and LLM) with shared interfaces
- `routing/`: Routers, classifiers, and escalation policies
- `memory/`: Memory backends (none, static, dynamic, salience-gated RAG)
- `evaluation/`: Metric implementations, scoring adapters, and reports
- `experiments/`: Configs plus `run_grid.py` for batch sweeps
- `prompts/`: Prompt templates and contracts (JSON schemas)
- `utils/`: Shared helpers (logging, tracing, cost calculators)
- `tests/`: Unit and integration tests
- `workflows/ci.yml`: GitHub Actions pipeline (lint -> type-check -> test -> smoke)

## Installation
Requirements: Python 3.11+, `pip`, and (if using local SLMs) access to model weights/keys.

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .\.venv\Scripts\activate
pip install -U pip
pip install -e .[dev]      # installs runtime + ruff + black + mypy + pytest
```

## Linting and type checks
```bash
ruff check .
black .
mypy .
```

## Tests
```bash
pytest
```

## Running experiments
`run_grid.py` sweeps routers, memories, models, and seeds. The default grid matches the dissertation benchmark.

```bash
python -m experiments.run_grid \
  --routers slm_dominant llm_first hybrid \
  --memories none static dynamic salience \
  --slms slm_a slm_b slm_c \
  --llms llm_a llm_b \
  --contracts on off \
  --seeds 1 2 3 \
  --output runs/latest
```

## Benchmarking grid (dissertation)
- 3 SLMs x 1-2 LLM fallbacks x 3 routers x 4 memories x contracts on/off x 3 seeds.
- Total cells: `3 * (1 or 2) * 3 * 4 * 2 * 3` = 216-432 runs, each logged with latency, cost, groundedness, schema validity, fallback rate, and accuracy.
- Each run stores raw generations, validation outcomes, and cost/latency traces for reproducibility.

## Reproducing dissertation results
1) Install dependencies and set provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).
2) Populate `experiments/configs/grid.yaml` (or CLI args) with the grid above.
3) Run `python -m experiments.run_grid --config experiments/configs/grid.yaml --seeds 1 2 3`.
4) Summarize with `python -m evaluation.summarize --runs runs/latest --format csv markdown`.
5) Compare `runs/latest/summary.*` against the dissertation tables; regression deltas are printed per metric.

## Methodology highlights
- RouterGym mirrors NVIDIA's SLM-centric agent design: start cheap, escalate rarely, and use memory plus contracts to keep quality.
- Routing uses task classification confidence, contract failures, and safety/PII filters to trigger LLM fallback.
- Memory co-design pairs router style with memory depth (for example, hybrid specialists with salience-gated RAG).
- Contract validation enforces JSON schemas with structured retries; hard failures count toward fallback metrics.
- Ground truth alignment uses groundedness scorers and task-specific accuracy checks for benchmark parity.

## CI/CD
- GitHub Actions: `.github/workflows/ci.yml` runs `ruff`, `black --check`, `mypy`, `pytest` (smoke focus), and uploads summary artifacts.
- Optional: `pre-commit install` to run the same checks locally on every commit.

## Quickstart
- Edit `experiments/configs/grid.yaml` to match your providers and models.
- Run `python -m experiments.run_grid --config experiments/configs/grid.yaml`.
- Inspect metrics in `runs/<timestamp>/summary.md` and raw traces under `runs/<timestamp>/`.
- Tune router thresholds and memory budgets in `routing/` and `memory/` configs to push SLM dominance while keeping accuracy.

## Citation
If you use RouterGym, please cite the dissertation:

> Kparobor Akpomiemie (2025). From LLM-First to SLM-Dominant: A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems.
