"""Antrenează intent classifier-ul: TF-IDF + LogisticRegression.

Pipeline:  TfidfVectorizer(ngram_range=(1,2))  →  LogisticRegression(max_iter=1000)
Salvează modelul cu joblib pentru producție (încărcat o singură dată la runtime).

Rulează (din proiect/):
    python -m intent.train_intent
    # sau direct:
    python intent/train_intent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# proiect root pe path → importul absolut `intent.*` merge și când rulezi direct scriptul
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from intent.intent_data import LABELS, TEST_DATA, TRAINING_DATA

MODEL_PATH = Path(__file__).parent / "intent_classifier.joblib"


def build_pipeline() -> Pipeline:
    """Pipeline TF-IDF (unigrame+bigrame) + LogisticRegression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    texts = [t for t, _ in TRAINING_DATA]
    labels = [lbl for _, lbl in TRAINING_DATA]

    counts = {lbl: labels.count(lbl) for lbl in sorted(set(labels))}
    print(f"Antrenez pe {len(texts)} exemple — {counts}")

    clf = build_pipeline()
    clf.fit(texts, labels)

    # Evaluare pe setul held-out (formulări noi) — accuracy realistă, nu pe train.
    if TEST_DATA:
        x_test = [t for t, _ in TEST_DATA]
        y_test = [lbl for _, lbl in TEST_DATA]
        y_pred = clf.predict(x_test)
        acc = sum(p == g for p, g in zip(y_pred, y_test)) / len(y_test)
        print(f"\n=== Held-out test ({len(y_test)} exemple) — accuracy={acc:.1%} ===")
        print(classification_report(y_test, y_pred, labels=LABELS, zero_division=0))

    joblib.dump(clf, MODEL_PATH)
    print(f"✓ Model salvat: {MODEL_PATH}")


if __name__ == "__main__":
    main()
