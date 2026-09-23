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


# Mapping from language name to spaCy model
SPACY_MODELS = {
    'english': 'en_core_web_sm',
    'spanish': 'es_core_news_sm',
}
