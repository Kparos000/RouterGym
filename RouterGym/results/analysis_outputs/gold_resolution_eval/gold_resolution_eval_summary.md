# Gold Resolution Evaluation Summary

- Scored rows: 456
- Configs: 6
- Best mean quality: `llm_only__base_llm1__mem_rag_bm25` = 0.6354
- Lowest average cost: `slm_only__base_slm1__mem_rag_bm25` = 0.000647
- Lowest average latency: `slm_dominant__base_slm2__esc_llm2__mem_rag_bm25` = 1390.80 ms

This evaluation scores generated resolution outputs against the frozen gold-resolution set.
It is separate from classifier-derived category accuracy.

## Quality by Config

| Config | Rows | Mean Quality | Median | Pass >=0.70 | Pass >=0.80 | Step | Acceptance | Escalation | Policy | Avg Cost | Avg Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `llm_only__base_llm1__mem_rag_bm25` | 76 | 0.6354 | 0.6407 | 0.1974 | 0.0132 | 0.3407 | 1.0000 | 1.0000 | 0.4956 | 0.006565 | 4640.44 |
| `llm_only__base_llm2__mem_rag_bm25` | 76 | 0.5990 | 0.6028 | 0.0526 | 0.0000 | 0.2497 | 1.0000 | 1.0000 | 0.4956 | 0.003036 | 2783.84 |
| `slm_dominant__base_slm1__esc_llm2__mem_rag_bm25` | 76 | 0.6110 | 0.6098 | 0.1316 | 0.0132 | 0.2796 | 1.0000 | 1.0000 | 0.4956 | 0.001052 | 2118.96 |
| `slm_dominant__base_slm2__esc_llm2__mem_rag_bm25` | 76 | 0.5934 | 0.5972 | 0.0395 | 0.0000 | 0.2357 | 1.0000 | 1.0000 | 0.4956 | 0.001090 | 1390.80 |
| `slm_only__base_slm1__mem_rag_bm25` | 76 | 0.6132 | 0.6098 | 0.1447 | 0.0132 | 0.2852 | 1.0000 | 1.0000 | 0.4956 | 0.000647 | 2449.19 |
| `slm_only__base_slm2__mem_rag_bm25` | 76 | 0.5913 | 0.5972 | 0.0395 | 0.0000 | 0.2304 | 1.0000 | 1.0000 | 0.4956 | 0.000686 | 1944.25 |
