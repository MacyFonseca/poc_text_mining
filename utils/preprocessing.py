"""Text preprocessing utilities."""
import re
import string
from typing import List, Tuple, Union
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from config.settings import TextPreprocessingConfig

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """Comprehensive text preprocessing pipeline."""

    def __init__(self, config: TextPreprocessingConfig):
        """Initialize preprocessor with configuration."""
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords_set = set(stopwords.words(config.language))
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spacy model...")
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text."""
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def lowercase(self, text: str) -> str:
        """Convert text to lowercase."""
        if self.config.lowercase:
            text = text.lower()
        return text

    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        return text

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from tokenized text."""
        if self.config.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords_set]
        return tokens

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def filter_by_length(self, tokens: List[str]) -> List[str]:
        """Filter tokens by minimum length."""
        return [token for token in tokens if len(token) >= self.config.min_word_length]

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        return tokens

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        # Step 1: Clean text
        text = self.clean_text(text)
        # Step 2: Lowercase
        text = self.lowercase(text)
        # Step 3: Remove punctuation
        text = self.remove_punctuation(text)
        # Step 4: Remove numbers
        text = self.remove_numbers(text)
        # Step 5: Tokenize
        tokens = self.tokenize(text)
        # Step 6: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        # Step 7: Filter by length
        tokens = self.filter_by_length(tokens)
        # Step 8: Lemmatize
        tokens = self.lemmatize(tokens)
        # Join tokens back into string
        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess multiple texts."""
        return [self.preprocess(text) for text in texts]

    def extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]

    def extract_ngrams(self, text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """Extract n-grams from text."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.filter_by_length(tokens)
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def extract_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Extract POS tags from text."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
