"""Classification algorithms for bias detection."""
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from config.settings import ClassificationConfig


class TextClassifier:
    """Classification model for bias detection."""

    def __init__(self, config: ClassificationConfig):
        """Initialize classifier."""
        self.config = config
        self.vectorizer = None
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _create_models(self) -> Dict:
        """Create classification models."""
        models = {}
        
        if 'logistic_regression' in self.config.algorithms:
            models['logistic_regression'] = LogisticRegression(
                random_state=self.config.random_state,
                max_iter=1000
            )
        
        if 'random_forest' in self.config.algorithms:
            models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        if 'svm' in self.config.algorithms:
            models['svm'] = SVC(
                kernel='rbf',
                random_state=self.config.random_state,
                probability=True
            )
        
        return models

    def prepare_data(self, texts: List[str], labels: List[int]) -> 'TextClassifier':
        """Prepare and vectorize text data."""
        # Configure vectorizer with adaptive parameters for small datasets
        n_docs = len(texts)
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
        X = self.vectorizer.fit_transform(texts).toarray()
        
        # Adaptive test size for small datasets
        n_samples = len(texts)
        unique_labels = len(set(labels))
        # For small datasets, use smaller test size (min 1 sample * num_classes)
        min_test_samples = max(1, unique_labels)
        test_size = max(min_test_samples / n_samples, self.config.test_size)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, labels,
            test_size=test_size,
            random_state=self.config.random_state,
            stratify=labels
        )
        
        return self

    def train(self) -> 'TextClassifier':
        """Train all models."""
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        self.models = self._create_models()
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            model.fit(self.X_train, self.y_train)
        
        return self

    def evaluate(self) -> Dict:
        """Evaluate all models."""
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")
        
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            
            results[model_name] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(self.y_test, y_pred, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
                'classification_report': classification_report(self.y_test, y_pred, zero_division=0)
            }
            
            # Add ROC-AUC for binary classification
            if len(np.unique(self.y_test)) == 2:
                try:
                    y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                    results[model_name]['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)
                except Exception:
                    results[model_name]['roc_auc'] = None
        
        return results

    def predict(self, texts: List[str]) -> Dict:
        """Predict labels for new texts."""
        if not self.models or self.vectorizer is None:
            raise ValueError("Models not trained. Call train() first.")
        
        X = self.vectorizer.transform(texts).toarray()
        predictions = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(X)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                predictions[model_name] = {
                    'labels': pred.tolist(),
                    'probabilities': proba.tolist()
                }
            else:
                predictions[model_name] = {
                    'labels': pred.tolist(),
                    'probabilities': None
                }
        
        return predictions

    def get_feature_importance(self, model_name: str = 'random_forest', top_n: int = 20) -> List[Tuple[str, float]]:
        """Get feature importance for a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(f"Model {model_name} does not have feature_importances_.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        
        # Get top features
        top_indices = np.argsort(importances)[::-1][:top_n]
        
        return [(feature_names[i], importances[i]) for i in top_indices]

    def get_coefficients(self, model_name: str = 'logistic_regression', top_n: int = 20) -> Tuple[List, List]:
        """Get model coefficients for interpretation."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'coef_'):
            raise ValueError(f"Model {model_name} does not have coefficients.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Get top positive and negative features
        top_positive_indices = np.argsort(coefficients)[::-1][:top_n]
        top_negative_indices = np.argsort(coefficients)[:top_n]
        
        positive_features = [(feature_names[i], coefficients[i]) for i in top_positive_indices]
        negative_features = [(feature_names[i], coefficients[i]) for i in top_negative_indices]
        
        return positive_features, negative_features

    def generate_classification_report(self) -> str:
        """Generate comprehensive classification report."""
        evaluation = self.evaluate()
        
        report = []
        report.append("=" * 60)
        report.append("CLASSIFICATION REPORT")
        report.append("=" * 60)
        
        report.append(f"\nTrain set size: {len(self.y_train)}")
        report.append(f"Test set size: {len(self.y_test)}")
        
        report.append(f"\nModel Performance:")
        for model_name, metrics in evaluation.items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Accuracy: {metrics['accuracy']:.4f}")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1-Score: {metrics['f1']:.4f}")
            if metrics.get('roc_auc'):
                report.append(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return '\n'.join(report)
