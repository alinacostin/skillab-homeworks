"""Intent classifier — inferență.

Încarcă o singură dată modelul joblib antrenat de `train_intent.py` și expune
`detect_intent(query) -> (label, confidence)`. Decizia de fallback la LLM
(confidence sub prag) aparține apelantului (vezi `QAAgent`).

    from intent import detect_intent, is_trained
    label, conf = detect_intent("găsește facturile din martie")  # → ("search", 0.94)
"""
from __future__ import annotations

from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "intent_classifier.joblib"
LABELS = ["search", "extract", "summarize"]

# Modelul se încarcă o singură dată (cold start ~50ms), apoi e refolosit.
_classifier = None


def is_trained() -> bool:
    """True dacă modelul a fost antrenat și salvat pe disc."""
    return MODEL_PATH.exists()


def load_classifier():
    """Încarcă (lazy, cached) pipeline-ul TF-IDF + LogisticRegression."""
    global _classifier
    if _classifier is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model neantrenat: {MODEL_PATH.name} lipsește. "
                f"Rulează mai întâi: python -m intent.train_intent"
            )
        _classifier = joblib.load(MODEL_PATH)
    return _classifier


def detect_intent(query: str) -> tuple[str, float]:
    """Detectează intentul (search/extract/summarize) — înlocuiește un apel LLM.

    Returns:
        (label, confidence) unde confidence = max(predict_proba).
    """
    clf = load_classifier()
    label = str(clf.predict([query])[0])
    confidence = float(max(clf.predict_proba([query])[0]))
    return label, confidence
