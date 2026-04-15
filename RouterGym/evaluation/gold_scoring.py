"""Deterministic scoring of generated outputs against frozen gold examples."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

GOLD_SCORING_VERSION = "gold_quality/v1"
STEP_COVERAGE_WEIGHT = 0.40
ACCEPTANCE_ALIGNMENT_WEIGHT = 0.20
ESCALATION_CORRECTNESS_WEIGHT = 0.20
POLICY_GROUNDING_WEIGHT = 0.20

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(slots=True)
class GoldScoreResult:
    """Component and aggregate scores for a single generated record."""

    gold_match_found: bool
    gold_match_reason: str
    gold_ticket_index: Optional[int]
    gold_review_status: str
    step_coverage_score: Optional[float]
    acceptance_criteria_alignment_score: Optional[float]
    escalation_correctness_score: Optional[float]
    policy_grounding_match_score: Optional[float]
    overall_gold_quality_score: Optional[float]
    gold_scoring_version: str = GOLD_SCORING_VERSION

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "gold_match_found": self.gold_match_found,
            "gold_match_reason": self.gold_match_reason,
            "gold_ticket_index": self.gold_ticket_index,
            "gold_review_status": self.gold_review_status,
            "step_coverage_score": self.step_coverage_score,
            "acceptance_criteria_alignment_score": self.acceptance_criteria_alignment_score,
            "escalation_correctness_score": self.escalation_correctness_score,
            "policy_grounding_match_score": self.policy_grounding_match_score,
            "overall_gold_quality_score": self.overall_gold_quality_score,
            "gold_scoring_version": self.gold_scoring_version,
        }


def _normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().split())


def _tokenize(text: Any) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(str(text or ""))]


def _token_set(text: Any) -> set[str]:
    return set(_tokenize(text))


def _coerce_str_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _extract_text_similarity(left: Any, right: Any) -> float:
    left_text = _normalize_text(left)
    right_text = _normalize_text(right)
    if not left_text or not right_text:
        return 0.0
    left_tokens = _token_set(left_text)
    right_tokens = _token_set(right_text)
    token_overlap = len(left_tokens & right_tokens) / max(len(right_tokens), 1)
    sequence_ratio = SequenceMatcher(None, left_text, right_text).ratio()
    return max(0.0, min(1.0, 0.8 * token_overlap + 0.2 * sequence_ratio))


def _coverage_score(predicted_items: Sequence[str], gold_items: Sequence[str]) -> float:
    if not gold_items and not predicted_items:
        return 1.0
    if not gold_items:
        return 0.0
    if not predicted_items:
        return 0.0
    best_scores = []
    for gold_item in gold_items:
        best = max(_extract_text_similarity(pred_item, gold_item) for pred_item in predicted_items)
        best_scores.append(best)
    return sum(best_scores) / len(best_scores)


def _policy_f1(predicted_policies: Iterable[str], gold_policies: Iterable[str]) -> float:
    pred = {str(pid).strip() for pid in predicted_policies if str(pid).strip()}
    gold = {str(pid).strip() for pid in gold_policies if str(pid).strip()}
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    overlap = len(pred & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def _extract_ticket_index(record: Mapping[str, Any]) -> Optional[int]:
    for key in ("ticket_index", "ticket_id"):
        if key not in record:
            continue
        value = record.get(key)
        if value in (None, ""):
            continue
        try:
            return int(str(value))
        except (TypeError, ValueError):
            continue
    return None


def _extract_gold_resolution(gold_record: Mapping[str, Any]) -> Dict[str, Any]:
    resolution = gold_record.get("gold_resolution", {})
    if not isinstance(resolution, dict):
        return {}
    return dict(resolution)


def _extract_generated_steps(record: Mapping[str, Any]) -> List[str]:
    for key in ("resolution_steps", "steps"):
        values = record.get(key)
        if isinstance(values, list):
            return _coerce_str_list(values)
    gold_resolution = record.get("gold_resolution")
    if isinstance(gold_resolution, dict):
        return _coerce_str_list(gold_resolution.get("steps", []))
    output = record.get("output")
    if isinstance(output, dict):
        for key in ("resolution_steps", "steps"):
            values = output.get(key)
            if isinstance(values, list):
                return _coerce_str_list(values)
    return []


def _extract_generated_acceptance(record: Mapping[str, Any]) -> List[str]:
    values = record.get("acceptance_criteria")
    if isinstance(values, list):
        return _coerce_str_list(values)
    gold_resolution = record.get("gold_resolution")
    if isinstance(gold_resolution, dict):
        acceptance = _coerce_str_list(gold_resolution.get("acceptance_criteria", []))
        if acceptance:
            return acceptance
    output = record.get("output")
    if isinstance(output, dict):
        acceptance = _coerce_str_list(output.get("acceptance_criteria", []))
        if acceptance:
            return acceptance
        final_answer = str(output.get("final_answer", "")).strip()
        return [final_answer] if final_answer else []
    final_answer = str(record.get("final_answer", "")).strip()
    return [final_answer] if final_answer else []


def _extract_generated_policies(record: Mapping[str, Any]) -> List[str]:
    values = record.get("kb_policy_ids")
    if isinstance(values, list):
        return _coerce_str_list(values)
    gold_resolution = record.get("gold_resolution")
    if isinstance(gold_resolution, dict):
        policies = _coerce_str_list(gold_resolution.get("kb_policies", []))
        if policies:
            return policies
    output = record.get("output")
    if isinstance(output, dict):
        return _coerce_str_list(output.get("kb_policy_ids", []))
    return []


def _extract_generated_escalation_required(record: Mapping[str, Any]) -> bool:
    if isinstance(record.get("escalation_required"), bool):
        return bool(record.get("escalation_required"))
    gold_resolution = record.get("gold_resolution")
    if isinstance(gold_resolution, dict) and isinstance(gold_resolution.get("escalation_required"), bool):
        return bool(gold_resolution.get("escalation_required"))
    flags = record.get("escalation_flags")
    if isinstance(flags, dict):
        return bool(flags.get("needs_human")) or bool(flags.get("policy_gap"))
    output = record.get("output")
    if isinstance(output, dict) and isinstance(output.get("escalation_required"), bool):
        return bool(output.get("escalation_required"))
    return False


def _extract_generated_escalation_reason(record: Mapping[str, Any]) -> str:
    reason = str(record.get("escalation_reason", "")).strip()
    if reason:
        return reason
    gold_resolution = record.get("gold_resolution")
    if isinstance(gold_resolution, dict):
        reason = str(gold_resolution.get("escalation_reason", "")).strip()
        if reason:
            return reason
    flags = record.get("escalation_flags")
    if isinstance(flags, dict):
        reasons = flags.get("reasons", [])
        if isinstance(reasons, list):
            return "; ".join(str(reason).strip() for reason in reasons if str(reason).strip())
    output = record.get("output")
    if isinstance(output, dict):
        return str(output.get("escalation_reason", "")).strip()
    return ""


def escalation_correctness_score(
    predicted_required: bool,
    predicted_reason: str,
    gold_required: bool,
    gold_reason: str,
) -> float:
    """Score whether the generated escalation decision matches the gold label."""
    if predicted_required != gold_required:
        return 0.0
    if not gold_required:
        return 1.0
    if predicted_reason.strip() and gold_reason.strip():
        return 1.0
    if predicted_reason.strip():
        return 0.75
    return 0.5


def combine_gold_quality_scores(
    step_coverage: float,
    acceptance_alignment: float,
    escalation_correctness: float,
    policy_grounding_match: float,
) -> float:
    """Combine component scores using transparent fixed weights."""
    return max(
        0.0,
        min(
            1.0,
            step_coverage * STEP_COVERAGE_WEIGHT
            + acceptance_alignment * ACCEPTANCE_ALIGNMENT_WEIGHT
            + escalation_correctness * ESCALATION_CORRECTNESS_WEIGHT
            + policy_grounding_match * POLICY_GROUNDING_WEIGHT,
        ),
    )


def score_record_against_gold(
    record: Mapping[str, Any],
    gold_record: Optional[Mapping[str, Any]],
) -> GoldScoreResult:
    """Score a generated record against a frozen gold example."""
    if gold_record is None:
        return GoldScoreResult(
            gold_match_found=False,
            gold_match_reason="missing_gold_record",
            gold_ticket_index=_extract_ticket_index(record),
            gold_review_status="",
            step_coverage_score=None,
            acceptance_criteria_alignment_score=None,
            escalation_correctness_score=None,
            policy_grounding_match_score=None,
            overall_gold_quality_score=None,
        )

    gold_resolution = _extract_gold_resolution(gold_record)
    gold_steps = _coerce_str_list(gold_resolution.get("steps", []))
    gold_acceptance = _coerce_str_list(gold_resolution.get("acceptance_criteria", []))
    gold_policies = _coerce_str_list(gold_resolution.get("kb_policies", []))
    gold_escalation = bool(gold_resolution.get("escalation_required", False))
    gold_escalation_reason = str(gold_resolution.get("escalation_reason", "")).strip()

    predicted_steps = _extract_generated_steps(record)
    predicted_acceptance = _extract_generated_acceptance(record)
    predicted_policies = _extract_generated_policies(record)
    predicted_escalation = _extract_generated_escalation_required(record)
    predicted_escalation_reason = _extract_generated_escalation_reason(record)

    step_score = _coverage_score(predicted_steps, gold_steps)
    acceptance_score = _coverage_score(predicted_acceptance, gold_acceptance)
    escalation_score = escalation_correctness_score(
        predicted_required=predicted_escalation,
        predicted_reason=predicted_escalation_reason,
        gold_required=gold_escalation,
        gold_reason=gold_escalation_reason,
    )
    policy_score = _policy_f1(predicted_policies, gold_policies)
    overall_score = combine_gold_quality_scores(
        step_coverage=step_score,
        acceptance_alignment=acceptance_score,
        escalation_correctness=escalation_score,
        policy_grounding_match=policy_score,
    )

    return GoldScoreResult(
        gold_match_found=True,
        gold_match_reason="matched_by_ticket_index",
        gold_ticket_index=_extract_ticket_index(gold_record),
        gold_review_status=str(gold_record.get("review_status", "")).strip(),
        step_coverage_score=step_score,
        acceptance_criteria_alignment_score=acceptance_score,
        escalation_correctness_score=escalation_score,
        policy_grounding_match_score=policy_score,
        overall_gold_quality_score=overall_score,
    )


def build_gold_index(records: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Index frozen gold records by ticket index."""
    indexed: Dict[int, Dict[str, Any]] = {}
    for record in records:
        ticket_index = _extract_ticket_index(record)
        if ticket_index is None:
            continue
        indexed[ticket_index] = dict(record)
    return indexed


__all__ = [
    "GOLD_SCORING_VERSION",
    "GoldScoreResult",
    "build_gold_index",
    "combine_gold_quality_scores",
    "escalation_correctness_score",
    "score_record_against_gold",
]
