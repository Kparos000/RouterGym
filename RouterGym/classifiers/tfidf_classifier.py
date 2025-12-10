"""Lightweight TF-IDF + centroid classifier."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    DEFAULT_LABELS,
    canonical_label,
    normalize_probabilities,
    apply_lexical_prior,
)

_DEFAULT_CORPUS: List[Tuple[str, str]] = [
    ("Reset my password immediately", "access"),
    ("Login keeps failing for vpn", "access"),
    ("Need VPN unlock and MFA reset", "access"),
    ("Please grant admin rights on laptop", "administrative rights"),
    ("Need elevated permissions to install software", "administrative rights"),
    ("Update security group membership for user", "administrative rights"),
    ("Laptop battery swollen and dock broken", "hardware"),
    ("Printer jams every morning", "hardware"),
    ("External monitor not detected", "hardware"),
    ("Questions about benefits and payroll", "hr support"),
    ("Need help updating maternity leave", "hr support"),
    ("Please purchase a new Tableau license", "purchase"),
    ("Need approval to buy new monitor", "purchase"),
    ("Renew annual subscription and process invoice", "purchase"),
    ("General inquiry unsure which team", "miscellaneous"),
    ("Strange request without a category", "miscellaneous"),
]

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class TFIDFClassifier(ClassifierProtocol):
    """Bag-of-words centroid classifier trained on a seed corpus."""

    def __init__(self, labels: Optional[Iterable[str]] = None, corpus: Optional[List[Tuple[str, str]]] = None) -> None:
        self.labels = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.corpus = corpus or _DEFAULT_CORPUS
        self.metadata = ClassifierMetadata(
            name="TF-IDF",
            mode="tfidf",
            provider="builtin",
            model_reference="tfidf-centroid",
            token_cost=0.0,
            latency_ms=2.0,
            description="TF-IDF centroid classifier",
        )
        self._idf: Dict[str, float] = {}
        self._label_vectors: Dict[str, Dict[str, float]] = {}
        self._train()

    def _train(self) -> None:
        doc_freq: Counter[str] = Counter()
        tokenized_docs: List[Tuple[List[str], str]] = []
        for text, label in self.corpus:
            tokens = _tokenize(text)
            tokenized_docs.append((tokens, canonical_label(label)))
            doc_freq.update(set(tokens))
        total_docs = max(len(tokenized_docs), 1)
        self._idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }
        label_vectors: defaultdict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for tokens, label in tokenized_docs:
            counts = Counter(tokens)
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                if token not in self._idf:
                    continue
                tf = count / total
                tfidf = tf * self._idf[token]
                label_vectors[label][token] += tfidf
        self._label_vectors = {label: dict(vec) for label, vec in label_vectors.items()}

    def _text_vector(self, text: str) -> Dict[str, float]:
        counts = Counter(_tokenize(text or ""))
        total = sum(counts.values()) or 1
        vector: Dict[str, float] = {}
        for token, count in counts.items():
            if token not in self._idf:
                continue
            vector[token] = (count / total) * self._idf[token]
        return vector

    def predict_proba(self, text: str) -> Dict[str, float]:
        vector = self._text_vector(text)
        scores: Dict[str, float] = {}
        for label in self.labels:
            centroid = self._label_vectors.get(label, {})
            dot = sum(vector[token] * centroid.get(token, 0.0) for token in vector)
            scores[label] = max(dot, 0.0)
        base_probs = normalize_probabilities(scores, self.labels)
        # TF-IDF is brittle; lean a bit more on lexical priors to capture obvious cues.
        return apply_lexical_prior(text, base_probs, alpha=0.6, beta=0.4)

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.__getitem__)


__all__ = ["TFIDFClassifier"]
