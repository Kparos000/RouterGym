# Generator Bug Fix Plan And Patch Report

## Summary

The final 47,837-ticket production run is not usable for answer-quality evaluation because the generator validated intermediate model drafts against the full `AgentOutput` schema too early. When that validation failed, the draft was replaced with `{}`, which caused placeholder answers, empty `resolution_steps`, and universal `slm_dominant` escalation.

This patch fixes that failure mode without rerunning inference.

## Exact Bug Fixed

Before this patch, `RouterGym/agents/generator.py` did this inside `run_ticket_pipeline()`:

1. call the model
2. parse a partial draft
3. immediately run `validate_agent_output(...)` on that partial draft
4. on failure, replace `parsed_output` with `{}`

That was incorrect because `validate_agent_output(...)` enforces the full final `AgentOutput` contract, while the intermediate model draft only contains a small subset of fields such as:

- `final_answer`
- `reasoning`
- `resolution_steps`
- optional category fields

Once the draft was emptied, downstream code saw:

- no answer
- no steps
- invalid schema

and then emitted:

- `final_answer = "No valid answer produced"`
- `resolution_steps = []`
- fallback reasoning text

## Patch Applied

### 1. Relaxed intermediate draft validation

Added `DraftOutputSchema` in `RouterGym/contracts/schema_contract.py`.

This draft schema validates only the intermediate generation payload, not the full final benchmark row.

### 2. Draft parsing now preserves diagnostics

Added `DraftParseResult` and a new `_parse_draft_output(...)` path in `RouterGym/agents/generator.py`.

Each model call now preserves:

- `raw_model_response_text`
- `parse_error`
- `validation_error`
- `parsed_output_before_validation`
- normalized draft output

### 3. Natural-language normalization

Non-JSON but usable model outputs are now normalized into a draft structure instead of being discarded.

This supports outputs such as:

- `Answer: ...`
- `Reasoning: ...`
- numbered or bulleted resolution steps

### 4. Invalid generations no longer look successful

Rows now expose:

- `generation_valid`
- `generation_invalid_reason`
- `has_real_final_answer`
- `has_resolution_steps`
- `placeholder_answer`
- `raw_response_saved`

If the chosen generation is invalid, the chunk executor records that row as a failure (`GenerationInvalidError`) instead of silently treating it as a successful ticket.

### 5. Final payload validation moved to the end

The full `validate_agent_output(...)` check now happens only after draft normalization into the final output shape.

## Quality Gates Added

Added `RouterGym/scripts/check_generation_quality_gate.py`.

It fails preflight when any config exceeds these thresholds:

- placeholder answer rate > `2%`
- empty `resolution_steps` rate > `5%`
- `raw_response_saved` rate < `100%`
- `generation_valid` rate < `95%`
- `slm_dominant` escalation rate == `1.0` unless explicitly allowed

## Tests Added / Updated

Updated:

- `RouterGym/tests/test_generator.py`
- `RouterGym/tests/test_chunked_execution.py`

Added:

- `RouterGym/tests/test_quality_gate.py`

Covered cases:

- valid JSON model output survives the intermediate draft path
- valid natural-language output is normalized successfully
- malformed output preserves raw text and parse/validation errors
- intermediate drafts are not judged by the full final schema too early
- the quality gate rejects placeholder-answer outputs
- chunk execution marks `generation_valid = false` rows as failures

## Verification

The following checks passed after the patch:

- `ruff check .`
- `mypy RouterGym`
- `pytest -q RouterGym/tests/test_chunked_execution.py`
- `pytest -q RouterGym/tests/test_openai_compatible_backend.py`
- `pytest -q RouterGym/tests/test_generator.py RouterGym/tests/test_quality_gate.py`

## Operational Conclusion

Yes, a **100-ticket preflight is now safe to run**.

Reason:

- invalid generations will be preserved for debugging instead of being collapsed into placeholder rows
- bad outputs will fail explicitly at chunk execution time
- the new quality gate will fail the preflight before a large production run can proceed unnoticed

## Required Next Step Before Re-running 47k

Run a fresh **100-ticket preflight** through the real serving path, then immediately run the quality gate against the merged output. If that passes, the full inference rerun is justified.
