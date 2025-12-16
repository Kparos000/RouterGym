"""Tests for the tuned TF-IDF classifier baseline."""

from __future__ import annotations

from typing import List, Tuple

from RouterGym.classifiers.tfidf_classifier import TFIDFClassifier
from RouterGym.label_space import CANONICAL_LABELS


def _synthetic_corpus() -> List[Tuple[str, str]]:
    return [
        ("reset password login issue", "Access"),
        ("vpn access locked account", "Access"),
        ("admin rights needed to install software", "Administrative rights"),
        ("add user to security group with admin permissions", "Administrative rights"),
        ("laptop battery failure and monitor issue", "Hardware"),
        ("printer paper jam and monitor not working", "Hardware"),
        ("payroll benefits question for hr", "HR Support"),
        ("vacation approval request and hr question", "HR Support"),
        ("buy new laptop invoice and renewal", "Purchase"),
        ("renew software subscription and pay invoice", "Purchase"),
        ("general inquiry no clear category", "Miscellaneous"),
        ("miscellaneous other general question", "Miscellaneous"),
        ("internal project roadmap and stakeholders", "Internal Project"),
        ("storage quota increase and disk expansion", "Storage"),
    ]


def test_tfidf_classifier_produces_probs_for_all_labels() -> None:
    corpus = _synthetic_corpus()
    clf = TFIDFClassifier(labels=CANONICAL_LABELS, corpus=corpus, random_state=0)

    text = "reset my vpn access and unlock account"
    probs = clf.predict_proba(text)
    assert set(probs.keys()) == set(CANONICAL_LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    pred = clf.predict_label(text)
    assert pred in CANONICAL_LABELS


def test_tfidf_classifier_prefers_hr_and_purchase_over_misc() -> None:
    corpus = _synthetic_corpus()
    clf = TFIDFClassifier(labels=CANONICAL_LABELS, corpus=corpus, random_state=0)

    hr_text = "payroll deduction and benefits correction required"
    hr_probs = clf.predict_proba(hr_text)
    assert hr_probs["HR Support"] > hr_probs["Miscellaneous"]

    purchase_text = "need to renew software subscription and pay vendor invoice"
    purchase_probs = clf.predict_proba(purchase_text)
    assert purchase_probs["Purchase"] > purchase_probs["Miscellaneous"]
