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
python RouterGym/analysis/build_gold_resolution_eval_subset.py
python RouterGym/analysis/score_gold_resolution_outputs.py
python RouterGym/analysis/plot_gold_resolution_quality.py
python RouterGym/analysis/create_manual_audit_sample.py
python RouterGym/analysis/create_manual_audit_sample.py --all-gold-matched
```

Do not run `python RouterGym/analysis/aggregate_manual_audit.py` until the manual audit CSV has
been filled by reviewers.

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
- `gold_resolution_eval/` containing deterministic generated-resolution quality scoring
- `manual_audit/` containing blinded human-audit CSVs, the reviewer Excel workbook, keys, and rubric

`balanced_60k_all_configs_flat.csv` is generated for local inspection but is ignored by git because
it is a large derived extract. Prefer committing compact summaries and plots.

## Metric Interpretation

The workflow separates four kinds of evidence:

- Classifier-derived category accuracy: `predicted_category` / `classifier_predicted_category`
  compared with `gold_label`. This evaluates the calibrated classifier, not generated answer
  correctness.
- Generated-category accuracy: `generated_predicted_category` compared with the gold category when
  the model emitted a category.
- Deterministic gold-resolution quality: generated `resolution_steps`, `final_answer`,
  `escalation_flags`, and `kb_policy_ids` scored against `gold_eval_final.jsonl`.
- Manual audit quality: blinded human review of generated answers using a 0-2 component rubric and
  0-10 overall score.

Generated-resolution correctness should be discussed through the gold-resolution scorer and manual
audit, not through classifier-derived category accuracy.

## Recommended Dissertation Tables

- Dataset integrity by configuration
- Classification metrics by configuration
- Classification accuracy by ticket category
- Generation quality and reliability by configuration
- Gold-resolution quality by configuration
- Manual-audit quality by configuration after review completion
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
- Gold resolution quality versus cost
- Gold resolution quality versus latency
- Manual quality versus cost and latency after review completion

Memory-mode plots should only be generated when multiple memory modes are present in the input
dataset. For the current balanced production-scale result, BM25 RAG is the only memory mode.

## Manual Audit Workbook

The full gold-matched manual audit workbook is:

`RouterGym/results/analysis_outputs/manual_audit/manual_audit_full_blinded.xlsx`

It is the primary reviewer artifact. The `Review` sheet is blinded and contains only anonymous
system labels such as `System A` through `System F`; real configuration names are kept only in
`manual_audit_full_key.csv`. Reviewers should not open the key until scoring is complete.

The workbook provides dropdowns for the six 0-2 component score columns, the 0-10
`overall_manual_quality` column, and `reviewer_id`. The CSV version is preserved for compatibility,
but reviewers should use the Excel workbook when possible.
