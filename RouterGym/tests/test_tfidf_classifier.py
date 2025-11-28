from pathlib import Path

import pandas as pd

from RouterGym.classification.tfidf_classifier import TFIDFClassifier


def test_tfidf_classifier_train_and_predict(tmp_path: Path) -> None:
    data = {
        "text": [
            "reset my password",
            "password reset please",
            "printer is jammed",
            "paper jam in printer",
            "unlock my account",
            "account locked",
            "monitor not working",
            "screen flickering monitor",
        ],
        "label": [
            "access",
            "access",
            "hardware",
            "hardware",
            "access",
            "access",
            "hardware",
            "hardware",
        ],
    }
    df = pd.DataFrame(data)
    clf = TFIDFClassifier()
    acc = clf.train(df)
    assert acc >= 0.0  # training completes

    label, conf, probs = clf.predict_with_confidence("password reset")
    assert label in {"access", "hardware"}
    assert 0.0 <= conf <= 1.0
    assert isinstance(probs, dict) and probs

    save_path = tmp_path / "tfidf.pkl"
    clf.save(save_path)
    assert save_path.exists()

    loaded = TFIDFClassifier.load(save_path)
    label2, conf2, probs2 = loaded.predict_with_confidence("paper jam")
    assert label2 in {"access", "hardware"}
    assert 0.0 <= conf2 <= 1.0
    assert isinstance(probs2, dict) and probs2
