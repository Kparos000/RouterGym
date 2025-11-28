"""Train TF-IDF + Logistic Regression classifier for RouterGym tickets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from RouterGym.classification.tfidf_classifier import TFIDFClassifier
from RouterGym.data.tickets import dataset_loader


def main() -> None:
    df = dataset_loader.load_dataset(None)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Failed to load tickets dataset")
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")

    train_df, val_df = train_test_split(df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"])
    clf = TFIDFClassifier()
    clf.train(train_df, test_size=0.0)
    # Evaluate on validation split
    preds = []
    for text in val_df["text"].astype(str):
        label, _, _ = clf.predict_with_confidence(text)
        preds.append(label)
    val_accuracy = (pd.Series(preds).reset_index(drop=True) == val_df["label"].reset_index(drop=True)).mean()
    print(f"[TFIDFClassifier] Validation accuracy: {val_accuracy:.4f}")

    model_dir = Path("RouterGym/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tfidf_classifier.pkl"
    clf.save(model_path)
    print(f"[TFIDFClassifier] Saved model to {model_path}")


if __name__ == "__main__":
    main()
