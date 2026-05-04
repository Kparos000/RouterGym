# RouterGym

**From LLM-First to SLM-Dominant: A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems**

RouterGym is a dissertation benchmark for studying whether enterprise support agents can move away
from an expensive **LLM-first** design toward a more scalable **SLM-dominant** design.

The central question is:

> Can smaller language models handle most support-ticket resolutions when they are paired with a
> router, BM25 retrieval memory, structured output validation, and selective escalation to a larger
> model?

This repository contains the RouterGym codebase, benchmark utilities, analysis scripts, result
summaries, plots, and manual-audit workflow used to evaluate that question.

## Why This Project Exists

Large language models are powerful, but they are also slower and more expensive to operate at
scale. In many agentic systems, every request is sent to a large model by default. RouterGym tests a
different design:

1. Classify the support ticket.
2. Retrieve relevant context from a memory layer.
3. Route the ticket to a small or large model.
4. Generate a structured resolution.
5. Validate the output.
6. Measure quality, cost, latency, reliability, and escalation behavior.

The project is not trying to prove that small models are always better. It measures when smaller
models are good enough, when larger models still help, and what tradeoffs appear when routing is
introduced.

## Final Dissertation Experiment

The production-scale benchmark uses a balanced local inference dataset:

- **60,000 generated outputs**
- **6 configurations**
- **10,000 matched ticket IDs per configuration**
- **Fixed memory mode: BM25 RAG**
- **Task domain: enterprise/support-ticket resolution**

The six final configurations are:

| Config family | Base model | Escalation model | Memory |
|---|---|---|---|
| LLM-only | LLM1 | none | BM25 RAG |
| LLM-only | LLM2 | none | BM25 RAG |
| SLM-only | SLM1 | none | BM25 RAG |
| SLM-only | SLM2 | none | BM25 RAG |
| SLM-dominant | SLM1 | LLM2 | BM25 RAG |
| SLM-dominant | SLM2 | LLM2 | BM25 RAG |

RouterGym supports multiple memory/context modes in code, but the final production-scale
dissertation result fixes memory to **BM25 RAG**. This is intentional: the dissertation compares
router/model strategies under a consistent operational memory layer. A full memory-mode ablation is
treated as future work, not claimed as a production-scale result.

## Agent Flow

```text
Support ticket
    |
    v
Classifier
    |
    v
Router policy
    |
    +--> LLM-only path
    |
    +--> SLM-only path
    |
    +--> SLM-dominant path with possible LLM escalation
    |
    v
BM25 RAG memory retrieval
    |
    v
Model generation
    |
    v
JSON/schema parsing and validation
    |
    v
Saved result row with quality, cost, token, latency, and routing metadata
```

Each generated result can include:

- ticket text
- gold label
- classifier-predicted category
- generated-predicted category
- final answer
- reasoning
- resolution steps
- escalation flags
- parse/validation status
- raw model response metadata
- token counts
- cost
- latency
- configuration identifiers

## Repository Structure

```text
RouterGym/
  agents/                  Agent wrappers
  analysis/                Dissertation analysis and plotting scripts
  classification/          Classifier training/evaluation support
  classifiers/             Classifier implementations
  contracts/               Structured output contracts and validation helpers
  data/                    Local data assets, including frozen gold eval records
  engines/                 Model backend and OpenAI-compatible serving support
  evaluation/              Scoring utilities, including gold-resolution scoring
  experiments/             Benchmark configuration and experiment utilities
  memory/                  Memory/RAG components
  prompts/                 Prompt templates
  routing/                 Router policies and escalation logic
  scripts/                 Benchmark, merge, smoke-test, and infrastructure scripts
  tests/                   Unit and regression tests
  utils/                   Shared helpers
```

## Data and Version-Control Policy

The raw production inference data is intentionally **not committed** because it is large.

Ignored local result locations include:

- `RouterGym/results/analysis_input/`
- `RouterGym/results/archives/`
- large JSONL archives
- tar/tar.gz benchmark artifacts

The repository is intended to track:

- source code
- analysis scripts
- lightweight CSV summaries
- dissertation plots
- README/report files
- gold-evaluation definitions
- manual-audit templates/workbooks where appropriate

## Analysis Workflow

Run from the repository root:

```bash
python RouterGym/analysis/audit_balanced_60k_schema.py
python RouterGym/analysis/analyze_balanced_60k.py
python RouterGym/analysis/plot_balanced_60k.py
```

These scripts produce local outputs under:

```text
RouterGym/results/analysis_outputs/
```

Common outputs include:

- `summary_by_config.csv`
- `classification_metrics_by_config.csv`
- `generation_quality_by_config.csv`
- `token_cost_summary_by_config.csv`
- `latency_summary_by_config.csv`
- `cost_savings_vs_llm_baseline.csv`
- `plots/`

## Generated-Resolution Correctness

Classifier accuracy is not the same as answer correctness. RouterGym therefore separates:

- **Classifier-derived category accuracy**: classifier prediction vs. gold category.
- **Generated-category accuracy**: model-emitted category vs. gold category.
- **Gold-resolution quality**: deterministic scoring against frozen gold resolutions.
- **Manual audit quality**: blinded human judgment of generated answers.

The frozen gold evaluation file is:

```text
RouterGym/data/gold_eval/gold_eval_final.jsonl
```

It contains 96 frozen gold records. Of those, 76 overlap with the balanced production dataset,
giving:

```text
76 tickets x 6 configs = 456 generated outputs
```

Gold-resolution scoring workflow:

```bash
python RouterGym/analysis/build_gold_resolution_eval_subset.py
python RouterGym/analysis/score_gold_resolution_outputs.py
python RouterGym/analysis/plot_gold_resolution_quality.py
```

Manual-audit workflow:

```bash
python RouterGym/analysis/create_manual_audit_sample.py --all-gold-matched
```

This creates a blinded reviewer workbook:

```text
RouterGym/results/analysis_outputs/manual_audit/manual_audit_full_blinded.xlsx
```

The reviewer workbook uses anonymous system labels such as `System A` through `System F`. The true
configuration mapping is stored separately in `manual_audit_full_key.csv` and should not be opened
until scoring is complete.

After manual scoring is complete:

```bash
python RouterGym/analysis/aggregate_manual_audit.py \
  --audit-file RouterGym/results/analysis_outputs/manual_audit/manual_audit_full_blinded.csv \
  --key-file RouterGym/results/analysis_outputs/manual_audit/manual_audit_full_key.csv
```

## Current Results Snapshot

The following values come from the generated local analysis summaries.

### Full 60k Production Dataset

| Config | Rows | Classifier accuracy | Usable output rate | Avg cost/ticket | Mean latency |
|---|---:|---:|---:|---:|---:|
| LLM1 | 10,000 | 96.63% | 99.92% | $0.006654 | 5742 ms |
| LLM2 | 10,000 | 96.63% | 99.82% | $0.003085 | 3879 ms |
| SLM-dominant SLM1 -> LLM2 | 10,000 | 96.63% | 92.81% | $0.001079 | 2121 ms |
| SLM-dominant SLM2 -> LLM2 | 10,000 | 96.63% | 99.91% | $0.001244 | 1449 ms |
| SLM1 | 10,000 | 96.63% | 92.14% | $0.000655 | 3538 ms |
| SLM2 | 10,000 | 96.63% | 99.53% | $0.000700 | 2971 ms |

The identical classifier accuracy across configurations is expected because the classifier stage is
shared; it should not be interpreted as generated-answer correctness.

### Gold-Matched Generated-Resolution Evaluation

| Config | Gold quality | Pass >= 0.70 | Generated category accuracy | Avg cost/ticket | Avg latency |
|---|---:|---:|---:|---:|---:|
| LLM1 | 0.635 | 19.7% | 81.6% | $0.006565 | 4640 ms |
| LLM2 | 0.599 | 5.3% | 80.3% | $0.003036 | 2784 ms |
| SLM-dominant SLM1 -> LLM2 | 0.611 | 13.2% | 66.2% | $0.001052 | 2119 ms |
| SLM-dominant SLM2 -> LLM2 | 0.593 | 3.9% | 68.4% | $0.001090 | 1391 ms |
| SLM1 | 0.613 | 14.5% | 65.7% | $0.000647 | 2449 ms |
| SLM2 | 0.591 | 3.9% | 64.5% | $0.000686 | 1944 ms |

These deterministic scores are best interpreted as a structured proxy for generated-resolution
quality. The dissertation also uses blinded manual audit to provide human judgment.

## Cost and Latency Findings

Using LLM2 as the lower-cost LLM-only baseline:

- SLM-only SLM1 reduces average cost per ticket by about **78.8%**.
- SLM-only SLM2 reduces average cost per ticket by about **77.3%**.
- SLM-dominant SLM1 -> LLM2 reduces average cost per ticket by about **65.0%**.
- SLM-dominant SLM2 -> LLM2 reduces average cost per ticket by about **59.7%**.

Latency also improves for the SLM-dominant paths in the full 60k summary:

- LLM1 mean latency: **5742 ms**
- LLM2 mean latency: **3879 ms**
- SLM-dominant SLM1 -> LLM2 mean latency: **2121 ms**
- SLM-dominant SLM2 -> LLM2 mean latency: **1449 ms**

This supports the main dissertation framing: SLM-dominant routing can materially reduce cost and
latency, while generated-resolution quality must be evaluated separately through gold scoring and
manual audit.

## Installation

Python 3.10+ is required. Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

For Excel manual-audit workbook generation:

```bash
python -m pip install openpyxl
```

## Development Commands

```bash
ruff check .
ruff format .
mypy RouterGym
pytest
```

The project configuration is in `pyproject.toml`.

## Inference and Infrastructure Notes

The repository includes scripts for large-scale and local inference engineering, including:

- chunked benchmark execution
- result merging
- OpenAI-compatible model smoke tests
- local gateway checks
- RunPod/vLLM serving helpers
- quality-gate checks
- token-cap validation

Relevant scripts live in:

```text
RouterGym/scripts/
```

Examples include:

- `run_chunked_benchmark.py`
- `merge_results_chunks.py`
- `check_generation_quality_gate.py`
- `assert_local_openai_serving.py`
- `smoke_openai_compatible_model.py`
- `start_runpod_vllm_serving.sh`

Do not run production inference unless you have configured the required model backends and output
paths.

## What This Repository Does Not Claim

This repository does **not** claim that:

- the production-scale result compares every memory mode;
- classifier accuracy proves generated-answer correctness;
- SLMs are universally better than LLMs;
- deterministic gold scoring replaces human judgment.

The claim is narrower and more defensible:

> Under a fixed BM25 RAG memory condition, RouterGym evaluates whether SLM-dominant routing can
> reduce cost and latency while preserving useful generated-resolution quality, with answer quality
> assessed through deterministic gold scoring and blinded manual audit.

## Citation

If you use this repository, cite the dissertation project:

```text
Kparobor Akpomiemie. From LLM-First to SLM-Dominant:
A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems.
Dissertation project, 2026.
```

## License

License information was not found in the repository at the time this README was written. Add a
`LICENSE` file before public release if the project will be shared outside the dissertation review
context.
