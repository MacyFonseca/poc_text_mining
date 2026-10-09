"""Configuration for text preprocessing."""
from dataclasses import dataclass


@dataclass
class TextPreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    remove_numbers: bool = False
    min_word_length: int = 2
    language: str = "english"


# spaCy models per language, in order of preference.  The medium models give
# noticeably better lemmas and morphology (needed for generic-masculine
# detection); the small ones are kept as a fallback.
SPACY_MODELS = {
    'english': ['en_core_web_md', 'en_core_web_sm'],
    'spanish': ['es_core_news_md', 'es_core_news_sm'],
}
