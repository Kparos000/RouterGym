# RouterGym Balanced 60k Analysis Workflow

This workflow analyzes the local production-scale inference results for the dissertation:

> From LLM-First to SLM-Dominant: A Router-Memory Co-Design and Conversion Benchmark for Agentic Systems

The main empirical dataset is the balanced 60,000-row inference output located under
`RouterGym/results/analysis_input`. That raw directory is intentionally not committed because it is
large. Commit scripts, lightweight CSV summaries, plots, and documentation instead.

## Experimental Scope

The production-scale benchmark fixes memory/context to BM25 RAG. RouterGym supports multiple
memory/context modes, but the dissertation-scale run uses BM25 RAG as the operational memory layer
to keep the experiment tractable, reproducible, and cost-aware.

This dataset supports router/model strategy comparison under a consistent memory condition:

- `llm_only`
- `slm_only`
- `slm_dominant`

It does not claim a production-scale comparison across `none`, transcript, dense, BM25, and hybrid
memory modes. Broader memory-mode ablation is a limitation and future-work item unless additional
memory-mode results are later added.

## Commands

Run from the repository root:

```bash
python RouterGym/analysis/audit_balanced_60k_schema.py
python RouterGym/analysis/analyze_balanced_60k.py
python RouterGym/analysis/plot_balanced_60k.py
```

## Outputs

All generated artifacts are written to:

`RouterGym/results/analysis_outputs`

Key outputs include:

- `dataset_integrity_report.json`
- `metric_column_detection_report.json`
- `balanced_60k_all_configs_flat.csv`
- `summary_by_config.csv`
- `classification_metrics_by_config.csv`
- `generation_quality_by_config.csv`
- `token_cost_summary_by_config.csv`
- `projected_47k_cost_by_config.csv`
- `cost_savings_vs_llm_baseline.csv`
- `latency_summary_by_config.csv`
- `routing_escalation_summary.csv` when escalation fields are available
- `plots/` containing dissertation-ready PNG figures

`balanced_60k_all_configs_flat.csv` is generated for local inspection but is ignored by git because
it is a large derived extract. Prefer committing compact summaries and plots.

## Recommended Dissertation Tables

- Dataset integrity by configuration
- Classification metrics by configuration
- Classification accuracy by ticket category
- Generation quality and reliability by configuration
- Token and cost summary by configuration
- Projected 47k cost by configuration
- Cost savings versus LLM-only baseline
- Latency summary by configuration
- Routing escalation summary for SLM-dominant configurations

## Recommended Dissertation Figures

- Classification accuracy by configuration
- Macro and weighted F1 by configuration
- Cost per ticket by configuration
- Projected 47k cost by configuration
- Accuracy versus cost
- Usable output rate versus cost
- Latency by configuration
- Escalation rate by configuration
- Router-family quality and cost comparisons

Memory-mode plots should only be generated when multiple memory modes are present in the input
dataset. For the current balanced production-scale result, BM25 RAG is the only memory mode.
