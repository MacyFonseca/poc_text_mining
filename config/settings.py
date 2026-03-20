"""Configuration management for text mining pipeline."""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TextPreprocessingConfig:
    """Configuration for text preprocessing."""
    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    remove_numbers: bool = False
    min_word_length: int = 2
    language: str = "english"


@dataclass
class BiasDetectionConfig:
    """Configuration for bias detection."""
    model_name: str = "bert-base-uncased"
    device: str = "cuda"
    batch_size: int = 16
    max_length: int = 512
    threshold: float = 0.5


@dataclass
class TopicModelingConfig:
    """Configuration for BERTopic."""
    min_topic_size: int = 10
    nr_topics: Optional[int] = None
    language: str = "english"
    embedding_model: str = "all-MiniLM-L6-v2"
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 10


@dataclass
class ClusteringConfig:
    """Configuration for clustering algorithms."""
    n_clusters: int = 5
    algorithm: str = "kmeans"  # kmeans, agglomerative, dbscan
    random_state: int = 42


@dataclass
class ClassificationConfig:
    """Configuration for classification tasks."""
    random_state: int = 42
    test_size: float = 0.2
    algorithms: list = None

    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ["logistic_regression", "random_forest", "svm"]


@dataclass
class DecisionEngineConfig:
    """Configuration for decision-making engine."""
    confidence_threshold: float = 0.7
    recommendation_count: int = 5
    explanation_depth: str = "detailed"  # brief, detailed, comprehensive


# Mapping from language name to spacy model
SPACY_MODELS = {
    'english': 'en_core_web_sm',
    'spanish': 'es_core_news_sm',
}


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    preprocessing: TextPreprocessingConfig = None
    bias_detection: BiasDetectionConfig = None
    topic_modeling: TopicModelingConfig = None
    clustering: ClusteringConfig = None
    classification: ClassificationConfig = None
    decision_engine: DecisionEngineConfig = None
    language: str = "english"
    random_state: int = 42
    n_jobs: int = -1

    def __post_init__(self):
        if self.preprocessing is None:
            self.preprocessing = TextPreprocessingConfig(language=self.language)
        if self.bias_detection is None:
            self.bias_detection = BiasDetectionConfig()
        if self.topic_modeling is None:
            self.topic_modeling = TopicModelingConfig(language=self.language)
        if self.clustering is None:
            self.clustering = ClusteringConfig()
        if self.classification is None:
            self.classification = ClassificationConfig()
        if self.decision_engine is None:
            self.decision_engine = DecisionEngineConfig()
        # Propagate language to sub-configs
        self.preprocessing.language = self.language
        self.topic_modeling.language = self.language


def load_config() -> PipelineConfig:
    """Load configuration from environment or use defaults."""
    return PipelineConfig()
