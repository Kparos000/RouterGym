# Final 47k Production Benchmark Summary

Date: 2026-04-27

Dataset:
- 47,837 support tickets

Configurations:
- slm_only__base_slm1__mem_rag_bm25
- slm_only__base_slm2__mem_rag_bm25
- llm_only__base_llm1__mem_rag_bm25
- llm_only__base_llm2__mem_rag_bm25
- slm_dominant__base_slm1__esc_llm2__mem_rag_bm25
- slm_dominant__base_slm2__esc_llm2__mem_rag_bm25

Outcome:
- All 6 configs completed
- 47,837 rows per config
- 287,022 total result rows
- 2,874 total chunk files
- 0 failed chunks
- 0 non-empty failure output files
- Backend: openai_compatible
- Memory: rag_bm25
- Chunk size: 100

Raw result bundle:
- routergym_final_47k_results_20260427_114304.tar.gz
- SHA256: 0bb0e9677067260c1e504ebee1f4903d1e7c9917743452351a3cfa61731942d1

Note:
The full raw output bundle is stored outside normal Git because it is large.
