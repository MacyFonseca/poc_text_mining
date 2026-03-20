"""Main text mining pipeline orchestrating all components."""
from typing import Dict, List, Optional, Any
import json
from datetime import datetime
from config.settings import PipelineConfig, load_config
from utils.preprocessing import TextPreprocessor
from analysis.bias_detector import BiasDetector
from analysis.topic_modeler import BERTopicModeler
from models.clustering import DocumentClusterer
from models.classification import TextClassifier
from models.decision_engine import DecisionEngine


class TextMiningPipeline:
    """Comprehensive text mining and bias detection pipeline."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize pipeline with components."""
        self.config = config or load_config()
        
        # Initialize components
        self.preprocessor = TextPreprocessor(self.config.preprocessing)
        self.bias_detector = BiasDetector(
            model_name=self.config.bias_detection.model_name,
            device=self.config.bias_detection.device,
            language=self.config.language
        )
        self.topic_modeler = BERTopicModeler(self.config.topic_modeling)
        self.clusterer = DocumentClusterer(self.config.clustering, language=self.config.language)
        self.classifier = TextClassifier(self.config.classification, language=self.config.language)
        self.decision_engine = DecisionEngine(self.config.decision_engine)
        
        # Results storage
        self.results = {}
        self.documents = []
        self.preprocessed_documents = []

    def load_documents(self, documents: List[str]) -> 'TextMiningPipeline':
        """Load documents into pipeline."""
        self.documents = documents
        return self

    def preprocess(self) -> 'TextMiningPipeline':
        """Preprocess documents."""
        print("Preprocessing documents...")
        self.preprocessed_documents = self.preprocessor.preprocess_batch(self.documents)
        self.results['preprocessing'] = {
            'original_count': len(self.documents),
            'preprocessed_count': len(self.preprocessed_documents)
        }
        print(f"✓ Preprocessed {len(self.documents)} documents")
        return self

    def detect_bias(self) -> 'TextMiningPipeline':
        """Detect bias in documents."""
        print("Detecting bias...")
        bias_results = self.bias_detector.batch_analysis(self.documents)
        
        # Aggregate results
        biased_count = sum(1 for r in bias_results if r['is_biased'])
        avg_bias_score = sum(r['overall_bias_score'] for r in bias_results) / len(bias_results)
        
        self.results['bias_detection'] = {
            'documents_analyzed': len(bias_results),
            'biased_documents': biased_count,
            'average_bias_score': avg_bias_score,
            'detailed_results': bias_results
        }
        print(f"✓ Bias detection complete: {biased_count} biased documents found")
        return self

    def topic_modeling(self) -> 'TextMiningPipeline':
        """Perform topic modeling."""
        print("Performing topic modeling with BERTopic...")
        self.topic_modeler.train(self.preprocessed_documents, verbose=False)
        
        topics_summary = self.topic_modeler.get_topic_summary()
        distribution = self.topic_modeler.get_topic_distribution()
        
        self.results['topic_modeling'] = {
            'num_topics': len(topics_summary),
            'topics': topics_summary,
            'distribution': distribution,
            'model_type': 'BERTopic'
        }
        print(f"✓ Topic modeling complete: {len(topics_summary)} topics identified")
        return self

    def clustering(self) -> 'TextMiningPipeline':
        """Perform document clustering."""
        print(f"Clustering documents with {self.config.clustering.algorithm}...")
        self.clusterer.fit(self.documents)
        
        distribution = self.clusterer.get_cluster_distribution()
        evaluation = self.clusterer.evaluate()
        
        self.results['clustering'] = {
            'algorithm': self.config.clustering.algorithm,
            'num_clusters': self.config.clustering.n_clusters,
            'distribution': distribution,
            'evaluation_metrics': evaluation
        }
        print(f"✓ Clustering complete: {len(distribution)} clusters created")
        return self

    def classification(self, labels: Optional[List[int]] = None) -> 'TextMiningPipeline':
        """Train and evaluate classification models."""
        if labels is None:
            print("Warning: No labels provided for classification. Skipping...")
            return self
        
        print("Training classification models...")
        self.classifier.prepare_data(self.documents, labels)
        self.classifier.train()
        
        evaluation = self.classifier.evaluate()
        
        self.results['classification'] = {
            'models_trained': list(self.classifier.models.keys()),
            'evaluation': evaluation
        }
        print(f"✓ Classification complete: {len(self.classifier.models)} models trained")
        return self

    def decision_making(self) -> 'TextMiningPipeline':
        """Generate decisions and recommendations."""
        print("Generating decisions and recommendations...")
        
        decisions = {}
        
        # Decision from bias analysis
        if 'bias_detection' in self.results and self.results['bias_detection']['detailed_results']:
            bias_analysis = self.results['bias_detection']['detailed_results'][0]
            decisions['bias_analysis'] = self.decision_engine.analyze_bias_results(bias_analysis)
        
        # Decision from classification
        if 'classification' in self.results and self.results['classification']['models_trained']:
            decisions['classification'] = self.decision_engine.analyze_classification_results(
                self.results['classification']['evaluation']
            )
        
        # Decision from topic analysis
        if 'topic_modeling' in self.results:
            decisions['topic_analysis'] = self.decision_engine.analyze_topic_results(
                self.results['topic_modeling']['topics'],
                self.results.get('bias_detection', {})
            )
        
        self.results['decisions'] = decisions
        print(f"✓ Decision making complete")
        return self

    def run_full_pipeline(self, documents: List[str], labels: Optional[List[int]] = None) -> Dict[str, Any]:
        """Run the complete pipeline."""
        print("\n" + "=" * 70)
        print("STARTING TEXT MINING PIPELINE")
        print("=" * 70 + "\n")
        
        start_time = datetime.now()
        
        self.load_documents(documents)
        self.preprocess()
        self.detect_bias()
        self.topic_modeling()
        self.clustering()
        
        if labels is not None:
            self.classification(labels)
        
        self.decision_making()
        
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print("PIPELINE EXECUTION COMPLETE")
        print("=" * 70)
        print(f"Total execution time: {execution_time:.2f} seconds\n")
        
        return self.get_results()

    def get_results(self) -> Dict[str, Any]:
        """Get all pipeline results."""
        return {
            'timestamp': datetime.now().isoformat(),
            'documents_processed': len(self.documents),
            'pipeline_results': self.results
        }

    def generate_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 70)
        report.append("TEXT MINING ANALYSIS REPORT")
        report.append("=" * 70)
        report.append(f"Timestamp: {datetime.now().isoformat()}\n")
        
        # Preprocessing summary
        if 'preprocessing' in self.results:
            report.append("PREPROCESSING SUMMARY:")
            report.append(f"  Documents processed: {self.results['preprocessing']['original_count']}")
            report.append(f"  Preprocessing successful: True\n")
        
        # Bias detection summary
        if 'bias_detection' in self.results:
            bd = self.results['bias_detection']
            report.append("BIAS DETECTION SUMMARY:")
            report.append(f"  Documents analyzed: {bd['documents_analyzed']}")
            report.append(f"  Biased documents: {bd['biased_documents']}")
            report.append(f"  Average bias score: {bd['average_bias_score']:.2%}\n")
        
        # Topic modeling summary
        if 'topic_modeling' in self.results:
            tm = self.results['topic_modeling']
            report.append("TOPIC MODELING SUMMARY:")
            report.append(f"  Number of topics: {tm['num_topics']}")
            report.append(f"  Top topics: {list(tm['topics'].keys())[:5]}\n")
        
        # Clustering summary
        if 'clustering' in self.results:
            cl = self.results['clustering']
            report.append("CLUSTERING SUMMARY:")
            report.append(f"  Algorithm: {cl['algorithm'].upper()}")
            report.append(f"  Number of clusters: {cl['num_clusters']}")
            if 'silhouette_score' in cl['evaluation_metrics']:
                report.append(f"  Silhouette score: {cl['evaluation_metrics']['silhouette_score']:.4f}\n")
        
        # Classification summary
        if 'classification' in self.results:
            cf = self.results['classification']
            report.append("CLASSIFICATION SUMMARY:")
            report.append(f"  Models trained: {', '.join(cf['models_trained'])}\n")
        
        # Decisions
        if 'decisions' in self.results:
            report.append("RECOMMENDATIONS:")
            for decision_type, decision in self.results['decisions'].items():
                report.append(f"\n  {decision_type.upper()}:")
                report.append(f"    Action: {decision.action}")
                report.append(f"    Priority: {decision.priority.upper()}")
        
        return '\n'.join(report)

    def export_results(self, filepath: str, format: str = 'json'):
        """Export results to file."""
        results = self.get_results()
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results exported to {filepath}")
        elif format == 'txt':
            with open(filepath, 'w') as f:
                f.write(self.generate_report())
            print(f"Report exported to {filepath}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of analysis results."""
        summary = {
            'total_documents': len(self.documents),
            'documents_with_bias': self.results.get('bias_detection', {}).get('biased_documents', 0),
            'num_topics': self.results.get('topic_modeling', {}).get('num_topics', 0),
            'num_clusters': self.results.get('clustering', {}).get('num_clusters', 0),
            'high_priority_items': sum(
                1 for d in self.results.get('decisions', {}).values()
                if d.priority == 'high'
            )
        }
        return summary
