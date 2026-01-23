"""Tests for automated gold eval autofill pipeline."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from RouterGym.scripts import autofill_gold_eval as autofill


def _stub_retrieval(policy_id: str = "kb1") -> Dict[str, Any]:
    snippet = {
        "policy_id": policy_id,
        "category": "Access",
        "title": "VPN Access",
        "text": "Follow the VPN access policy steps.",
        "score": 0.9,
        "escalation_notes": "",
    }
    return {"context": "", "snippets": [snippet], "policy_ids": [policy_id]}


def test_read_jsonl_parses_and_raises(tmp_path: Any) -> None:
    fpath = tmp_path / "gold.jsonl"
    fpath.write_text('{"ticket_index":1}\ninvalid\n', encoding="utf-8")
    records = autofill._read_jsonl(fpath)
    assert len(records) == 1
    assert records[0]["ticket_index"] == 1
    missing_path = tmp_path / "missing.jsonl"
    with pytest.raises(FileNotFoundError):
        autofill._read_jsonl(missing_path)


def test_autofill_marks_review_on_validation_fail(monkeypatch: Any) -> None:
    records = [
        {"ticket_index": 0, "topic_group": "Access", "ticket_text": "Need vpn access", "gold_resolution": {}}
    ]
    monkeypatch.setattr(autofill, "retrieve_kb", lambda text, memory_mode, top_k: _stub_retrieval("kb1"))
    monkeypatch.setattr(autofill, "_valid_policy_ids", lambda: {"kb1"})

    responses: List[str] = [
        json.dumps(
            {
                "summary": "",
                "steps": ["only one"],
                "escalation_required": True,
                "escalation_reason": "",
                "kb_policies": ["kb2"],
                "acceptance_criteria": [],
            }
        ),
        json.dumps(
            {
                "summary": "",
                "steps": ["only one"],
                "escalation_required": True,
                "escalation_reason": "",
                "kb_policies": ["kb2"],
                "acceptance_criteria": [],
            }
        ),
        json.dumps(
            {
                "summary": "",
                "steps": ["only one"],
                "escalation_required": True,
                "escalation_reason": "",
                "kb_policies": ["kb2"],
                "acceptance_criteria": [],
            }
        ),
        json.dumps(
            {
                "summary": "",
                "steps": ["only one"],
                "escalation_required": True,
                "escalation_reason": "",
                "kb_policies": ["kb2"],
                "acceptance_criteria": [],
            }
        ),
    ]

    def fake_call_model(model: Any, prompt: str) -> str:
        return responses.pop(0)

    monkeypatch.setattr(autofill, "call_model", fake_call_model)
    models = {"llm1": object(), "llm2": object()}

    filled, review_queue = autofill.autofill_records(
        records,
        models,
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
        seed=123,
    )

    assert len(filled) == 1
    assert filled[0]["needs_human_review"] is True
    reasons = filled[0]["review_reasons"]
    assert "steps_count_out_of_range" in reasons
    assert "empty_acceptance_criteria" in reasons
    assert "kb_policies_missing" in reasons
    assert "summary_missing" in reasons
    assert len(review_queue) == 1


def test_autofill_passes_validation(monkeypatch: Any) -> None:
    records = [
        {"ticket_index": 1, "topic_group": "Access", "ticket_text": "Need vpn setup", "gold_resolution": {}}
    ]
    monkeypatch.setattr(autofill, "retrieve_kb", lambda text, memory_mode, top_k: _stub_retrieval("kb-ok"))
    monkeypatch.setattr(autofill, "_valid_policy_ids", lambda: {"kb-ok"})

    responses: List[str] = [
        json.dumps(
            {
                "summary": "draft",
                "steps": ["a"],
                "escalation_required": False,
                "kb_policies": ["kb-ok"],
                "acceptance_criteria": ["draft"],
            }
        ),
        json.dumps(
            {
                "summary": "Provide VPN setup steps.",
                "steps": [
                    "Confirm user identity and account status.",
                    "Provision VPN access per policy.",
                    "Share VPN client download instructions.",
                ],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["kb-ok"],
                "acceptance_criteria": [
                    "User connects to VPN successfully.",
                    "VPN client is configured per policy.",
                ],
            }
        ),
        json.dumps(
            {
                "summary": "Provide VPN setup steps.",
                "steps": [
                    "Confirm user identity and account status.",
                    "Provision VPN access per policy.",
                    "Share VPN client download and configuration instructions.",
                ],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["kb-ok"],
                "acceptance_criteria": [
                    "User connects to VPN successfully.",
                    "VPN client is configured per policy.",
                ],
            }
        ),
    ]

    def fake_call_model(model: Any, prompt: str) -> str:
        return responses.pop(0)

    monkeypatch.setattr(autofill, "call_model", fake_call_model)
    models = {"llm1": object(), "llm2": object()}

    filled, review_queue = autofill.autofill_records(
        records,
        models,
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
        seed=7,
    )

    assert len(filled) == 1
    assert filled[0]["needs_human_review"] is False
    gold = filled[0]["gold_resolution"]
    assert gold["kb_policies"] == ["kb-ok"]
    assert len(gold["steps"]) == 3
    assert gold["acceptance_criteria"] == [
        "User connects to VPN successfully.",
        "VPN client is configured per policy.",
    ]
    assert review_queue == []


def test_dynamic_policy_ids_are_filtered(monkeypatch: Any) -> None:
    records = [{"ticket_index": 2, "topic_group": "Purchase", "ticket_text": "Need laptop", "gold_resolution": {}}]

    def fake_retrieve(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {
            "context": "",
            "snippets": [
                {"policy_id": "dynamic:74", "text": "user ticket text"},
                {"policy_id": "kb-valid", "text": "approved hardware purchase policy"},
            ],
            "policy_ids": ["dynamic:74", "kb-valid"],
        }

    monkeypatch.setattr(autofill, "retrieve_kb", fake_retrieve)
    monkeypatch.setattr(
        autofill,
        "_valid_policy_ids",
        lambda: {"kb-valid"},
    )

    responses: List[str] = [
        json.dumps(
            {
                "summary": "draft",
                "steps": ["Confirm request", "Check inventory", "Issue order"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["dynamic:74", "kb-valid"],
                "acceptance_criteria": ["Order placed", "Inventory updated"],
            }
        ),
        json.dumps(
            {
                "summary": "final",
                "steps": ["Confirm request", "Check inventory", "Issue order"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["dynamic:74", "kb-valid"],
                "acceptance_criteria": ["Order placed", "Inventory updated"],
            }
        ),
    ]

    monkeypatch.setattr(autofill, "call_model", lambda model, prompt: responses.pop(0))
    models = {"llm1": object(), "llm2": object()}

    filled, review_queue = autofill.autofill_records(
        records,
        models,
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
        seed=1,
    )

    assert len(filled) == 1
    gold = filled[0]["gold_resolution"]
    assert gold["kb_policies"] == ["kb-valid"]
    assert all("dynamic:" not in pid for pid in gold["kb_policies"])
    assert filled[0]["needs_human_review"] is False
    assert review_queue == []


def test_retrieve_kb_filters_dynamic(monkeypatch: Any) -> None:
    autofill._kb_index_by_id.cache_clear()
    monkeypatch.setattr(
        autofill.kb_loader,
        "load_kb_index",
        lambda: [
            {"id": "kb-valid", "category": "Hardware", "title": "T", "content": "Body", "escalation_notes": "", "path": "p", "tags": []}
        ],
    )

    class StubRetrieval:
        def __init__(self) -> None:
            self.retrieved_context = "ctx"
            self.retrieval_metadata = {
                "snippets": [
                    {"policy_id": "kb-valid", "text": "ok", "score": 1.0},
                    {"policy_id": "dynamic:1", "text": "user text", "score": 0.1},
                ]
            }

    class StubMemory:
        def __init__(self, top_k: int = 0) -> None:
            self.top_k = top_k

        def retrieve(self, query: str) -> StubRetrieval:
            return StubRetrieval()

    monkeypatch.setattr(autofill, "get_memory_class", lambda mode: StubMemory)

    result = autofill.retrieve_kb("need laptop", "rag_hybrid", 3)
    assert len(result["snippets"]) == 1
    assert result["snippets"][0]["policy_id"] == "kb-valid"
    assert result["policy_ids"] == ["kb-valid"]


def test_validate_gold_resolution_flags_fluff() -> None:
    res = {
        "summary": "As an AI I cannot access your system",
        "steps": ["a", "b", "c", "d"],
        "escalation_required": False,
        "escalation_reason": "",
        "kb_policies": ["kb1"],
        "acceptance_criteria": ["done", "verified"],
    }
    ok, reasons = autofill.validate_gold_resolution(res, ["kb1"])
    assert ok is False
    assert "non_actionable_fluff" in reasons


def test_retrieve_kb_accepts_id_and_source(monkeypatch: Any) -> None:
    autofill._kb_index_by_id.cache_clear()
    monkeypatch.setattr(
        autofill.kb_loader,
        "load_kb_index",
        lambda: [
            {"id": "kb-one", "category": "Access", "title": "One", "content": "VPN", "escalation_notes": "", "path": "p1", "tags": []},
            {"id": "kb-two", "category": "Access", "title": "Two", "content": "SSO", "escalation_notes": "", "path": "p2", "tags": []},
        ],
    )

    class StubRetrieval:
        def __init__(self) -> None:
            self.retrieved_context = "ctx"
            self.retrieval_metadata = {
                "snippets": [
                    {"id": "kb-one", "text": "first"},
                    {"source": "kb-two", "text": "second"},
                ]
            }

    class StubMemory:
        def __init__(self, top_k: int = 0) -> None:
            self.top_k = top_k

        def retrieve(self, query: str) -> StubRetrieval:
            return StubRetrieval()

    monkeypatch.setattr(autofill, "get_memory_class", lambda mode: StubMemory)
    result = autofill.retrieve_kb("vpn access", "rag_hybrid", 2)
    assert set(result["policy_ids"]) == {"kb-one", "kb-two"}


def test_fallback_selects_valid_ids(monkeypatch: Any) -> None:
    autofill._kb_index_by_id.cache_clear()
    monkeypatch.setattr(
        autofill.kb_loader,
        "load_kb_index",
        lambda: [
            {
                "id": "access.vpn",
                "category": "Access",
                "title": "VPN Access",
                "content": "VPN setup steps",
                "escalation_notes": "",
                "path": "p",
                "tags": [],
            }
        ],
    )
    monkeypatch.setattr(autofill, "retrieve_kb", lambda text, memory_mode, top_k: {"snippets": [], "policy_ids": []})

    responses: List[str] = [
        json.dumps(
            {
                "summary": "VPN setup",
                "steps": ["Confirm identity", "Provision VPN access", "Provide client setup"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["access.vpn"],
                "acceptance_criteria": ["User connects", "VPN configured"],
            }
        ),
        json.dumps(
            {
                "summary": "VPN setup",
                "steps": ["Confirm identity", "Provision VPN access", "Provide client setup"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["access.vpn"],
                "acceptance_criteria": ["User connects", "VPN configured"],
            }
        ),
    ]

    monkeypatch.setattr(autofill, "call_model", lambda model, prompt: responses.pop(0))
    models = {"llm1": object(), "llm2": object()}
    filled, review_queue = autofill.autofill_records(
        [{"ticket_index": 3, "topic_group": "Access", "ticket_text": "vpn access", "gold_resolution": {}}],
        models,
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
    )
    assert filled[0]["needs_human_review"] is False
    assert "kb_fallback_used" in filled[0]["review_reasons"]
    assert filled[0]["gold_resolution"]["kb_policies"] == ["access.vpn"]
    assert review_queue == []


def test_repair_loop_recovers_valid_output(monkeypatch: Any) -> None:
    autofill._kb_index_by_id.cache_clear()
    monkeypatch.setattr(
        autofill.kb_loader,
        "load_kb_index",
        lambda: [
            {
                "id": "access.vpn",
                "category": "Access",
                "title": "VPN Access",
                "content": "VPN steps",
                "escalation_notes": "",
                "path": "p",
                "tags": [],
            }
        ],
    )
    monkeypatch.setattr(autofill, "retrieve_kb", lambda text, memory_mode, top_k: _stub_retrieval("access.vpn"))

    responses: List[str] = [
        json.dumps({"summary": "", "steps": ["x"], "escalation_required": False, "kb_policies": ["access.vpn"], "acceptance_criteria": []}),
        json.dumps(
            {
                "summary": "VPN setup",
                "steps": ["Confirm identity", "Provision VPN access", "Provide client setup"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["access.vpn"],
                "acceptance_criteria": ["User connects", "VPN configured"],
            }
        ),
        json.dumps(
            {
                "summary": "VPN setup",
                "steps": ["Confirm identity", "Provision VPN access", "Provide client setup"],
                "escalation_required": False,
                "escalation_reason": "",
                "kb_policies": ["access.vpn"],
                "acceptance_criteria": ["User connects", "VPN configured"],
            }
        ),
    ]
    prompts: List[str] = []

    def fake_call_model(model: Any, prompt: str) -> str:
        prompts.append(prompt)
        return responses.pop(0)

    monkeypatch.setattr(autofill, "call_model", fake_call_model)
    models = {"llm1": object(), "llm2": object()}
    filled, _ = autofill.autofill_records(
        [{"ticket_index": 4, "topic_group": "Access", "ticket_text": "vpn access", "gold_resolution": {}}],
        models,
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
    )
    assert any("Validation errors" in prompt for prompt in prompts)
    assert filled[0]["needs_human_review"] is False
    assert filled[0]["gold_resolution"]["kb_policies"] == ["access.vpn"]


def test_kb_index_missing_queues_review(monkeypatch: Any) -> None:
    autofill._kb_index_by_id.cache_clear()
    monkeypatch.setattr(autofill, "_kb_index_by_id", lambda: {})
    filled, review_queue = autofill.autofill_records(
        [{"ticket_index": 5, "topic_group": "Access", "ticket_text": "vpn access", "gold_resolution": {}}],
        {"llm1": object(), "llm2": object()},
        memory_mode="rag_hybrid",
        top_k=2,
        annotator_model="llm2",
        reviewer_model="llm1",
    )
    assert filled[0]["needs_human_review"] is True
    assert filled[0]["review_reasons"] == ["kb_index_empty_or_unreadable"]
    assert review_queue == filled
