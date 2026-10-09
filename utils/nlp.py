"""Shared, cached spaCy pipelines (one per language)."""
from functools import lru_cache

import spacy

from config.settings import SPACY_MODELS


@lru_cache(maxsize=None)
def get_nlp(language: str):
    """Load the best installed spaCy model for *language* (cached)."""
    candidates = SPACY_MODELS.get(language, SPACY_MODELS['english'])
    for name in candidates:
        try:
            return spacy.load(name)
        except OSError:
            continue
    raise OSError(
        f"No spaCy model installed for '{language}'. "
        f"Run: python -m spacy download {candidates[0]}"
    )
