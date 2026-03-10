"""Utility functions for data loading, visualization, and reporting."""
import os
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class DataLoader:
    """Load and manage datasets for analysis."""
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath)
    
    @staticmethod
    def load_json_lines(filepath: str) -> List[Dict]:
        """Load JSONL file (one JSON object per line)."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    @staticmethod
    def load_txt(filepath: str, delimiter: str = '\n\n') -> List[str]:
        """Load text file with custom delimiters."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.split(delimiter)
    
    @staticmethod
    def save_dataframe(df: pd.DataFrame, filepath: str, format: str = 'csv'):
        """Save dataframe to file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        if format == 'csv':
            df.to_csv(filepath, index=False)
        elif format == 'json':
            df.to_json(filepath, orient='records', indent=2)
        elif format == 'excel':
            df.to_excel(filepath, index=False)


class Visualizer:
    """Create visualizations for analysis results."""
    
    @staticmethod
    def plot_bias_distribution(bias_scores: List[float], threshold: float = 0.5):
        """Visualize bias score distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(bias_scores, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
        ax.set_xlabel('Bias Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Bias Scores')
        ax.legend()
        plt.grid(True, alpha=0.3)
        return fig
    
    @staticmethod
    def plot_cluster_distribution(cluster_distribution: Dict):
        """Visualize cluster distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))
        clusters = list(cluster_distribution.keys())
        counts = [cluster_distribution[c]['count'] for c in clusters]
        colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        ax.bar(clusters, counts, color=colors, edgecolor='black')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Document Count')
        ax.set_title('Document Distribution Across Clusters')
        plt.grid(True, alpha=0.3, axis='y')
        return fig
    
    @staticmethod
    def plot_model_comparison(evaluation_results: Dict):
        """Compare performance of multiple models."""
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model].get(metric, 0) for model in models]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim([0, 1.1])
        plt.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    @staticmethod
    def plot_feature_importance(features: List[Tuple[str, float]], top_n: int = 20):
        """Visualize feature importance."""
        features = features[:top_n]
        names, importances = zip(*features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        ax.barh(range(len(names)), importances, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Important Features')
        ax.invert_yaxis()
        plt.tight_layout()
        
        return fig


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    @staticmethod
    def create_html_report(pipeline_results: Dict, output_path: str):
        """Generate HTML report from pipeline results."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Text Mining Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                .section { margin: 20px 0; padding: 10px; border-left: 4px solid #007bff; }
                .metric { margin: 10px 0; }
                .value { font-weight: bold; color: #007bff; }
                table { border-collapse: collapse; width: 100%; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .warning { color: #ff6b6b; }
                .success { color: #51cf66; }
            </style>
        </head>
        <body>
            <h1>Text Mining and Bias Detection Report</h1>
            <div class="metric">Generated: <span class="value">""" + datetime.now().isoformat() + """</span></div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">Documents Processed: <span class="value">""" + str(pipeline_results['documents_processed']) + """</span></div>
            </div>
            
            <!-- Additional sections based on results -->
            
        </body>
        </html>
        """
        
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to {output_path}")
    
    @staticmethod
    def summarize_results(pipeline_results: Dict) -> str:
        """Create text summary of results."""
        summary = []
        summary.append("=" * 70)
        summary.append("ANALYSIS SUMMARY")
        summary.append("=" * 70)
        summary.append(f"\nTimestamp: {pipeline_results.get('timestamp', 'N/A')}")
        summary.append(f"Documents Processed: {pipeline_results.get('documents_processed', 0)}")
        
        results = pipeline_results.get('pipeline_results', {})
        
        if 'bias_detection' in results:
            bd = results['bias_detection']
            summary.append(f"\nBias Detection:")
            summary.append(f"  - Biased Documents: {bd.get('biased_documents', 0)}/{bd.get('documents_analyzed', 0)}")
            summary.append(f"  - Average Bias Score: {bd.get('average_bias_score', 0):.2%}")
        
        if 'topic_modeling' in results:
            tm = results['topic_modeling']
            summary.append(f"\nTopic Modeling:")
            summary.append(f"  - Topics Identified: {tm.get('num_topics', 0)}")
        
        if 'clustering' in results:
            cl = results['clustering']
            summary.append(f"\nClustering:")
            summary.append(f"  - Algorithm: {cl.get('algorithm', 'N/A').upper()}")
            summary.append(f"  - Number of Clusters: {cl.get('num_clusters', 0)}")
        
        return '\n'.join(summary)


class ResultsManager:
    """Manage and analyze pipeline results."""
    
    @staticmethod
    def save_results(results: Dict, filepath: str):
        """Save results to file."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
    
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """Load results from file."""
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
    
    @staticmethod
    def compare_analyses(results1: Dict, results2: Dict) -> Dict:
        """Compare two analysis results."""
        comparison = {
            'differences': {},
            'similarities': {}
        }
        
        # Compare basic metrics
        if 'documents_processed' in results1 and 'documents_processed' in results2:
            if results1['documents_processed'] == results2['documents_processed']:
                comparison['similarities']['doc_count'] = results1['documents_processed']
            else:
                comparison['differences']['doc_count'] = {
                    'first': results1['documents_processed'],
                    'second': results2['documents_processed']
                }
        
        return comparison


class TextAnalysisUtils:
    """Utility functions for text analysis."""
    
    @staticmethod
    def get_text_statistics(texts: List[str]) -> Dict:
        """Calculate text statistics."""
        word_counts = [len(text.split()) for text in texts]
        char_counts = [len(text) for text in texts]
        
        return {
            'total_documents': len(texts),
            'avg_words_per_doc': np.mean(word_counts),
            'avg_chars_per_doc': np.mean(char_counts),
            'max_words': max(word_counts),
            'min_words': min(word_counts),
            'total_words': sum(word_counts),
            'total_characters': sum(char_counts)
        }
    
    @staticmethod
    def identify_outliers(texts: List[str], metric: str = 'word_count') -> List[int]:
        """Identify outlier documents based on length."""
        if metric == 'word_count':
            lengths = [len(text.split()) for text in texts]
        else:
            lengths = [len(text) for text in texts]
        
        lengths = np.array(lengths)
        mean = np.mean(lengths)
        std = np.std(lengths)
        
        outliers = np.where(np.abs(lengths - mean) > 3 * std)[0]
        return outliers.tolist()
    
    @staticmethod
    def filter_texts(texts: List[str], min_length: int = 10, max_length: Optional[int] = None) -> Tuple[List[str], List[int]]:
        """Filter texts by length."""
        filtered = []
        indices = []
        
        for i, text in enumerate(texts):
            length = len(text.split())
            if length >= min_length and (max_length is None or length <= max_length):
                filtered.append(text)
                indices.append(i)
        
        return filtered, indices
