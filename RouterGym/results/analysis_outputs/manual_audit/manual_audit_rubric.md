# Manual Audit Rubric

Use this rubric to score generated resolution quality against the gold reference. Do not open or
inspect `manual_audit_key.csv` or `manual_audit_full_key.csv` until after scoring is complete.

## Component Score Columns

- `category_understanding_manual`: Does the generated response understand the ticket category and user need?
- `answer_actionable_manual`: Does it give concrete actions that could be followed by support staff or the user?
- `answer_complete_manual`: Does it cover the important parts of the gold resolution and avoid major omissions?
- `resolution_steps_correct_manual`: Are the generated resolution steps correct, ordered, and relevant?
- `escalation_appropriate_manual`: Does it escalate when escalation is needed and avoid unnecessary escalation?
- `policy_grounded_manual`: Is it consistent with the supplied policies, KB expectations, and acceptance criteria?

Score each component as:

- 0 = poor, wrong, unsafe, missing, or not usable
- 1 = partial or acceptable but incomplete
- 2 = good, correct, useful, and operationally actionable

## Overall Score

`overall_manual_quality` should be scored from 0 to 10:

- 0 = completely wrong, unsafe, or useless
- 1-2 = incorrect or mostly unusable resolution
- 3-4 = weak resolution with major omissions
- 5 = partially useful but incomplete
- 6 = useful but clearly missing important detail
- 7-8 = good resolution suitable for most operational use
- 9 = excellent resolution with only minor differences from the gold reference
- 10 = excellent, correct, actionable, complete, and closely aligned with the gold reference

## Consistency Guidance

Reviewers should compare the generated answer and generated resolution steps against the gold
summary, gold steps, and acceptance criteria. Penalize hallucinated steps, missing required actions,
incorrect escalation decisions, and vague answers that are not operationally useful.

The Excel workbook is the primary review artifact. Use the dropdowns in the `Review` sheet for all
component scores, `overall_manual_quality`, and `reviewer_id`. Do not manually add scoring columns
or decode system identities while reviewing.

The anonymous system ID must not be decoded during review. If multiple reviewers score the file,
each reviewer should fill `reviewer_id` so inter-reviewer agreement can be computed later.
