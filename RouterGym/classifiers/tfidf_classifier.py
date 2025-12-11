"""Production-grade TF-IDF + Logistic Regression baseline classifier."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from RouterGym.classifiers.utils import (
    ClassifierMetadata,
    ClassifierProtocol,
    DEFAULT_LABELS,
    canonical_label,
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


class TFIDFClassifier(ClassifierProtocol):
    """Bag-of-words TF-IDF + Logistic Regression classifier tuned as a strong lexical baseline.

    Design choices (common in production-grade baselines):
    - ngram_range=(1,3) to capture short IT phrases (e.g., "password reset", "vpn access").
    - sublinear_tf=True and strip_accents="unicode" to smooth term frequencies and normalize text.
    - English stopwords removal to reduce noise from generic filler (issue/problem/etc.).
    - Balanced multinomial logistic regression to handle class imbalance across the six labels.
    """

    def __init__(
        self,
        labels: Optional[Iterable[str]] = None,
        corpus: Optional[Sequence[tuple[str, str]]] = None,
        random_state: int = 42,
    ) -> None:
        self.labels: List[str] = [canonical_label(lbl) for lbl in (labels or DEFAULT_LABELS)]
        self.label_set = set(self.labels)
        self.corpus = list(corpus) if corpus is not None else list(_DEFAULT_CORPUS)
        self.random_state = random_state
        self.metadata = ClassifierMetadata(
            name="TF-IDF",
            mode="tfidf",
            provider="sklearn",
            model_reference="tfidf-logreg",
            token_cost=0.0,
            latency_ms=4.0,
            description="TF-IDF + multinomial logistic regression (balanced)",
        )
        self.pipeline: Pipeline = self._build_pipeline()
        self.classes_: np.ndarray = np.array(self.labels)
        self._fitted = False
        # Fit immediately on seed corpus so the classifier is usable out-of-the-box.
        seed_texts = [text for text, _ in self.corpus]
        seed_labels = [canonical_label(lbl) for _, lbl in self.corpus]
        self.fit(seed_texts, seed_labels)

    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            stop_words="english",
        )
        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=None,
            random_state=self.random_state,
        )
        return Pipeline([("tfidf", vectorizer), ("clf", clf)])

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "TFIDFClassifier":
        """Fit the TF-IDF + Logistic Regression pipeline on provided data."""
        y = [canonical_label(lbl) for lbl in labels]
        X = list(texts)
        self.pipeline.fit(X, y)
        # Align class order to the canonical labels for stable downstream usage.
        fitted_classes = list(self.pipeline.named_steps["clf"].classes_)
        self.classes_ = np.array([lbl for lbl in self.labels if lbl in fitted_classes], dtype=object)
        self._fitted = True
        return self

    def _predict_proba_vector(self, text: str) -> np.ndarray:
        if not self._fitted:
            # Safety net: fit on the seed corpus if fit was skipped.
            seed_texts = [t for t, _ in self.corpus]
            seed_labels = [canonical_label(lbl) for _, lbl in self.corpus]
            self.fit(seed_texts, seed_labels)
        proba_matrix = self.pipeline.predict_proba([text or ""])
        raw_probs = proba_matrix[0]
        clf_classes = self.pipeline.named_steps["clf"].classes_
        prob_by_class = {canonical_label(cls): float(prob) for cls, prob in zip(clf_classes, raw_probs)}
        ordered = np.array([prob_by_class.get(lbl, 0.0) for lbl in self.classes_], dtype=float)
        if ordered.sum() <= 0:
            ordered = np.full_like(ordered, 1.0 / max(len(ordered), 1))
        else:
            ordered = ordered / ordered.sum()
        return ordered

    def predict_proba(self, text: str) -> Dict[str, float]:
        probs_vec = self._predict_proba_vector(text)
        base_probs = {lbl: float(probs_vec[idx]) for idx, lbl in enumerate(self.classes_)}
        # Blend with lexical priors to capture obvious keyword cues without overpowering the model.
        return apply_lexical_prior(text, base_probs, alpha=0.7, beta=0.3)

    def predict_label(self, text: str) -> str:
        probabilities = self.predict_proba(text)
        label = max(probabilities, key=probabilities.__getitem__)
        return canonical_label(label if label in self.label_set else "miscellaneous")


__all__ = ["TFIDFClassifier"]
