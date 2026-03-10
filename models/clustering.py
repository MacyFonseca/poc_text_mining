"""Clustering algorithms for document grouping."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from config.settings import ClusteringConfig


class DocumentClusterer:
    """Document clustering using various algorithms."""

    def __init__(self, config: ClusteringConfig):
        """Initialize clusterer."""
        self.config = config
        self.documents = None
        self.vectorizer = None
        self.features = None
        self.clusters = None
        self.model = None

    def vectorize_documents(self, documents: List[str]) -> np.ndarray:
        """Vectorize documents using TF-IDF."""
        # Configure vectorizer with adaptive parameters for small datasets
        n_docs = len(documents)
        min_df = 1  # Always allow terms in at least 1 document
        # For small datasets, allow terms in all documents; for larger datasets, 95% max
        max_df = n_docs if n_docs < 5 else int(0.95 * n_docs) + 1
        
        self.vectorizer = TfidfVectorizer(
            max_features=None,  # Don't limit features for small datasets
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            token_pattern=r"(?u)\b[a-z]{2,}\b"  # Only words with 2+ chars
        )
        self.features = self.vectorizer.fit_transform(documents).toarray()
        return self.features

    def fit_kmeans(self, documents: List[str]) -> 'DocumentClusterer':
        """Fit KMeans clustering."""
        self.documents = documents
        features = self.vectorize_documents(documents)
        
        self.model = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init=10
        )
        self.clusters = self.model.fit_predict(features)
        
        return self

    def fit_agglomerative(self, documents: List[str]) -> 'DocumentClusterer':
        """Fit Agglomerative clustering."""
        self.documents = documents
        features = self.vectorize_documents(documents)
        
        self.model = AgglomerativeClustering(
            n_clusters=self.config.n_clusters,
            linkage='ward'
        )
        self.clusters = self.model.fit_predict(features)
        
        return self

    def fit_dbscan(self, documents: List[str], eps: float = 0.5, min_samples: int = 5) -> 'DocumentClusterer':
        """Fit DBSCAN clustering."""
        self.documents = documents
        features = self.vectorize_documents(documents)
        
        # Normalize features for DBSCAN
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = self.model.fit_predict(features_scaled)
        
        return self

    def fit(self, documents: List[str]) -> 'DocumentClusterer':
        """Fit clustering based on configured algorithm."""
        if self.config.algorithm == 'kmeans':
            return self.fit_kmeans(documents)
        elif self.config.algorithm == 'agglomerative':
            return self.fit_agglomerative(documents)
        elif self.config.algorithm == 'dbscan':
            return self.fit_dbscan(documents)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

    def predict(self, documents: List[str]) -> np.ndarray:
        """Predict clusters for new documents."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Vectorize new documents
        features = self.vectorizer.transform(documents).toarray()
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(features)
        else:
            raise ValueError(f"Model {type(self.model)} does not support predict.")

    def get_cluster_distribution(self) -> Dict:
        """Get distribution of documents across clusters."""
        if self.clusters is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        unique, counts = np.unique(self.clusters, return_counts=True)
        distribution = {}
        
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(self.clusters)) * 100
            distribution[int(cluster_id)] = {
                'count': int(count),
                'percentage': round(percentage, 2)
            }
        
        return distribution

    def get_cluster_documents(self, cluster_id: int) -> List[str]:
        """Get documents in a specific cluster."""
        if self.clusters is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        indices = np.where(self.clusters == cluster_id)[0]
        return [self.documents[i] for i in indices]

    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """Get cluster centers (for KMeans)."""
        if hasattr(self.model, 'cluster_centers_'):
            return self.model.cluster_centers_
        return None

    def get_cluster_keywords(self, cluster_id: int, top_n: int = 10) -> List[str]:
        """Get top keywords for a cluster."""
        if self.clusters is None or self.vectorizer is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get documents in cluster
        indices = np.where(self.clusters == cluster_id)[0]
        if len(indices) == 0:
            return []
        
        # Average TF-IDF scores for cluster
        cluster_features = self.features[indices].mean(axis=0)
        
        # Get top feature indices
        top_indices = np.argsort(cluster_features)[::-1][:top_n]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        return [feature_names[i] for i in top_indices]

    def evaluate(self) -> Dict:
        """Evaluate clustering quality."""
        if self.clusters is None or self.features is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Check if there's more than one cluster
        unique_clusters = np.unique(self.clusters)
        if len(unique_clusters) < 2:
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': 0.0,
                'note': 'Only one cluster found'
            }
        
        # Remove noise points (-1 clusters from DBSCAN)
        valid_mask = self.clusters != -1
        if valid_mask.sum() < 2:
            return {
                'silhouette_score': 0.0,
                'davies_bouldin_score': None,
                'calinski_harabasz_score': 0.0,
                'note': 'Not enough valid clusters'
            }
        
        valid_features = self.features[valid_mask]
        valid_clusters = self.clusters[valid_mask]
        
        # Check if we have enough samples for metrics
        # Both silhouette and calinski-harabasz require: num_clusters >= 2 and num_clusters < num_samples
        unique_valid_clusters = len(np.unique(valid_clusters))
        n_valid_samples = len(valid_clusters)
        
        result = {}
        
        # Silhouette score: requires 2 <= num_clusters < num_samples
        if unique_valid_clusters >= 2 and unique_valid_clusters < n_valid_samples:
            try:
                result['silhouette_score'] = float(silhouette_score(valid_features, valid_clusters))
            except:
                result['silhouette_score'] = 0.0
        else:
            result['silhouette_score'] = 0.0
        
        # Davies-Bouldin score: also requires 2 <= num_clusters < num_samples
        if unique_valid_clusters >= 2 and unique_valid_clusters < n_valid_samples:
            try:
                result['davies_bouldin_score'] = float(davies_bouldin_score(valid_features, valid_clusters))
            except:
                result['davies_bouldin_score'] = None
        else:
            result['davies_bouldin_score'] = None
        
        # Calinski-Harabasz score: also requires 2 <= num_clusters < num_samples
        if unique_valid_clusters >= 2 and unique_valid_clusters < n_valid_samples:
            try:
                result['calinski_harabasz_score'] = float(calinski_harabasz_score(valid_features, valid_clusters))
            except:
                result['calinski_harabasz_score'] = 0.0
        else:
            result['calinski_harabasz_score'] = 0.0
        
        return result

    def generate_clustering_report(self) -> str:
        """Generate clustering report."""
        if self.clusters is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        report = []
        report.append("=" * 60)
        report.append("CLUSTERING REPORT")
        report.append("=" * 60)
        
        report.append(f"\nAlgorithm: {self.config.algorithm.upper()}")
        report.append(f"Total Documents: {len(self.documents)}")
        
        distribution = self.get_cluster_distribution()
        report.append(f"\nCluster Distribution:")
        for cluster_id, info in sorted(distribution.items()):
            report.append(f"  Cluster {cluster_id}: {info['count']} docs ({info['percentage']}%)")
            keywords = self.get_cluster_keywords(cluster_id, top_n=5)
            if keywords:
                report.append(f"    Keywords: {', '.join(keywords)}")
        
        # Add evaluation metrics
        evaluation = self.evaluate()
        report.append(f"\nEvaluation Metrics:")
        for metric, value in evaluation.items():
            if metric != 'note':
                report.append(f"  {metric}: {value:.4f}")
        
        return '\n'.join(report)
