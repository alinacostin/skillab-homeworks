"""Intent classification — TF-IDF + LogisticRegression.

Înlocuiește un apel LLM de routing/intent cu un classifier local: ~100× mai ieftin,
~100× mai rapid, offline. Antrenare: `train_intent.py`; inferență:
`classifier.py`. Fallback-ul la LLM când confidence e sub prag e decis de apelant (QAAgent).
"""
from .classifier import LABELS, MODEL_PATH, detect_intent, is_trained

__all__ = ["detect_intent", "is_trained", "MODEL_PATH", "LABELS"]
