"""Topic modeling using BERTopic."""
from typing import List, Tuple, Dict, Optional
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from config.settings import TopicModelingConfig


class BERTopicModeler:
    """Topic modeling using BERTopic."""

    def __init__(self, config: TopicModelingConfig):
        """Initialize BERTopic modeler."""
        self.config = config
        self.model: Optional[BERTopic] = None
        self.documents = None
        self.topics = None
        self.probabilities = None

    def train(self, documents: List[str], verbose: bool = True) -> 'BERTopicModeler':
        """Train BERTopic model."""
        self.documents = documents
        
        # Configure vectorizer with adaptive parameters for small datasets
        n_docs = len(documents)
        min_df = 1  # Always allow terms in at least 1 document
        # For small datasets, allow terms in all documents; for larger datasets, 95% max
        max_df = n_docs if n_docs < 5 else int(0.95 * n_docs) + 1
        
        vectorizer_model = CountVectorizer(
            stop_words="english",
            max_features=None,  # Don't limit features for small datasets
            min_df=min_df,
            max_df=max_df,
            token_pattern=r"(?u)\b[a-z]{2,}\b"  # Only words with 2+ chars
        )
        
        # Initialize and train model
        self.model = BERTopic(
            language=self.config.language,
            embedding_model=self.config.embedding_model,
            vectorizer_model=vectorizer_model,
            min_topic_size=self.config.min_topic_size,
            nr_topics=self.config.nr_topics,
            verbose=verbose
        )
        
        self.topics, self.probabilities = self.model.fit_transform(documents)
        
        return self

    def get_topics(self) -> Dict:
        """Get topic information."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.get_topics()

    def get_topic_summary(self) -> Dict:
        """Get summary of topics."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        topics_dict = {}
        for topic_id, topic_info in self.model.get_topics().items():
            if topic_id != -1:  # Exclude outliers
                topics_dict[int(topic_id)] = {
                    'keywords': [str(word) for word, _ in topic_info[:5]],
                    'word_weights': [float(weight) for _, weight in topic_info[:5]]
                }
        
        return topics_dict

    def predict(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """Predict topics for new documents."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.transform(documents)

    def search_topics(self, search_term: str, top_n: int = 5) -> Dict:
        """Search for similar topics based on keyword."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        similar_topics = self.model.find_topics(search_term, top_n=top_n)
        
        results = {}
        for topic_id, similarity in similar_topics:
            if topic_id != -1:
                topic_info = self.model.get_topics()[topic_id]
                keywords = [str(word) for word, _ in topic_info[:5]]
                results[int(topic_id)] = {
                    'keywords': keywords,
                    'similarity': float(similarity)
                }
        
        return results

    def get_document_topic_assignment(self) -> Dict:
        """Get topic assignment for each document."""
        if self.model is None or self.topics is None:
            raise ValueError("Model not trained. Call train() first.")
        
        assignments = {}
        for idx, topic_id in enumerate(self.topics):
            assignments[int(idx)] = {
                'document': self.documents[int(idx)][:100],
                'topic_id': int(topic_id),
                'probability': float(self.probabilities[int(idx)].max()) if self.probabilities is not None else None
            }
        
        return assignments

    def get_topic_distribution(self) -> Dict:
        """Get distribution of documents across topics."""
        if self.topics is None:
            raise ValueError("Model not trained. Call train() first.")
        
        unique, counts = np.unique(self.topics, return_counts=True)
        distribution = {}
        
        for topic_id, count in zip(unique, counts):
            percentage = (float(count) / len(self.topics)) * 100
            distribution[int(topic_id)] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        return distribution

    def get_representative_documents(self, topic_id: int, top_n: int = 5) -> List[str]:
        """Get representative documents for a topic."""
        if self.topics is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Find documents with this topic
        indices = np.where(self.topics == topic_id)[0]
        
        if len(indices) == 0:
            return []
        
        # Sort by probability if available
        if self.probabilities is not None:
            probs = self.probabilities[indices].max(axis=1)
            top_indices = indices[np.argsort(probs)[::-1][:top_n]]
        else:
            top_indices = indices[:top_n]
        
        return [self.documents[i][:200] for i in top_indices]

    def visualize_topics(self):
        """Visualize topic model (requires jupyter environment)."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            return self.model.visualize_topics()
        except Exception as e:
            print(f"Visualization requires jupyter environment: {e}")
            return None

    def visualize_distribution(self, doc_topic_matrix=None):
        """Visualize topic distribution."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        try:
            return self.model.visualize_distribution(self.probabilities)
        except Exception as e:
            print(f"Distribution visualization error: {e}")
            return None

    def save_model(self, path: str):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.save(path)

    def load_model(self, path: str):
        """Load model from disk."""
        self.model = BERTopic.load(path)
        return self

    def generate_topic_report(self) -> str:
        """Generate a comprehensive topic report."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        report = []
        report.append("=" * 60)
        report.append("TOPIC MODELING REPORT")
        report.append("=" * 60)
        
        report.append(f"\nTotal Documents: {len(self.documents)}")
        report.append(f"\nTopic Distribution:")
        
        distribution = self.get_topic_distribution()
        for topic_id, info in sorted(distribution.items()):
            report.append(f"  Topic {topic_id}: {info['count']} docs ({info['percentage']}%)")
            if topic_id != -1:
                keywords = [word for word, _ in self.model.get_topics()[topic_id][:5]]
                report.append(f"    Keywords: {', '.join(keywords)}")
        
        return '\n'.join(report)
