# Text Mining and Bias Detection - Proof of Concept

A comprehensive machine learning pipeline for analyzing textual content in research projects to identify patterns, trends, and potential bias issues, including gender bias and discriminatory language detection.

## 🎯 Project Overview

This repository implements a sophisticated text mining and NLP solution that combines multiple machine learning techniques to:

- **Detect Bias**: Gender bias and discriminatory language identification
- **Extract Topics**: BERTopic for semantic topic extraction  
- **Cluster Documents**: Grouping similar documents for pattern analysis
- **Classify Text**: ML-based bias classification with multiple algorithms
- **Generate Insights**: Personalized recommendations and actionable guidance

### Key Technologies

- **NLP Models**: BERT, Transformers, Sentence Embeddings
- **ML Algorithms**: Classification, Clustering, Topic Modeling
- **Deep Learning**: TensorFlow, PyTorch, PyTorch Lightning
- **Topic Modeling**: BERTopic with UMAP and HDBSCAN
- **Text Processing**: spaCy, NLTK, TextBlob
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📦 Main Features

### 1. Text Preprocessing
- Tokenization, lemmatization, POS tagging
- Named entity recognition
- Stopword removal and text cleaning
- N-gram extraction

### 2. Bias Detection  
- Gender bias identification (male/female keyword analysis)
- Discriminatory language detection (age, race, disability, appearance)
- ML-based zero-shot classification for bias patterns

### 3. Topic Modeling
- BERTopic for semantic topic extraction
- Topic distribution analysis
- Representative document identification
- Topic-keyword relationships

### 4. Document Clustering
- Multiple algorithms: KMeans, Agglomerative, DBSCAN
- TF-IDF vectorization
- Cluster evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Keyword extraction per cluster

### 5. Classification Models
- Logistic Regression, Random Forest, SVM
- Performance evaluation and comparison
- Feature importance analysis
- Cross-validation support

### 6. Decision Engine
- Confidence-based decision making
- Personalized recommendations
- Priority-based action items
- Comprehensive guidance generation

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
cd poc_text_mining
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

### Basic Usage - Full Pipeline
```python
from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import PipelineConfig

# Your documents
documents = ["Research project text 1...", "Research project text 2..."]

# Initialize pipeline (auto-configures for available hardware)
pipeline = TextMiningPipeline()

# Load and process documents
pipeline.load_documents(documents)
pipeline.preprocess()
pipeline.detect_bias()
pipeline.topic_modeling()
pipeline.clustering()
pipeline.decision_making()

# Get results
results = pipeline.get_results()
print(f"Biased documents found: {results['bias_detection']['biased_documents']}")
print(f"Topics identified: {results['topic_modeling']['num_topics']}")

# Export results
pipeline.export_results('results.json', format='json')
```

### Run Examples
```bash
python example_analysis.py
jupyter notebook notebooks/text_mining_analysis.ipynb
```

## � Detailed Usage Examples

### Example 1: Detect Gender Bias in Job Postings
```python
from analysis.bias_detector import BiasDetector

detector = BiasDetector(device='cpu')

# Analyze job posting
job_posting = """
We are looking for a strong male engineer to lead our technical team.
The ideal candidate should be aggressive in pursuing technical excellence.
We prefer ambitious professionals who can take charge.
"""

analysis = detector.comprehensive_bias_analysis(job_posting)

# Check results
if analysis['is_biased']:
    print(f"⚠️  Bias detected: {analysis['overall_bias_score']:.1%}")
    print(f"Type: Gender bias ({analysis['gender_bias']['bias_direction']}-biased)")
    print(f"Confidence: {analysis['gender_bias']['confidence']:.1%}")
    
    # Show evidence
    for evidence in analysis['gender_bias']['male_evidence']:
        print(f"  Evidence: {evidence}")
```

### Example 2: Analyze Multiple Documents for Biases
```python
from analysis.bias_detector import BiasDetector

documents = [
    "We seek a talented engineer with strong problem-solving skills.",
    "Beautiful young woman needed for customer relations role.",
    "Experienced professional for leadership position. Equal opportunities.",
    "Seeking energetic employees aged 25-35 for startup team.",
]

detector = BiasDetector(device='cpu')
results = detector.batch_analysis(documents)

# Summary statistics
biased_count = sum(1 for r in results if r['is_biased'])
print(f"Biased documents: {biased_count}/{len(documents)}")

for i, result in enumerate(results, 1):
    print(f"\nDocument {i}:")
    print(f"  Biased: {result['is_biased']}")
    print(f"  Score: {result['overall_bias_score']:.1%}")
    print(f"  Gender Bias: {result['gender_bias']['has_gender_bias']}")
    
    # Show discriminatory language
    for category, data in result['discriminatory_language'].items():
        if data['has_discriminatory_language']:
            print(f"  ⚠️  {category}: {data['confidence_score']:.1%}")
```

### Example 3: Full Pipeline with Bias Analysis
```python
from pipelines.text_mining_pipeline import TextMiningPipeline

# Research project descriptions
documents = [
    "Male-led research in AI and software engineering...",
    "Dynamic team working on innovative cloud solutions...",
    "Need strong, aggressive project leaders...",
]

pipeline = TextMiningPipeline()
pipeline.load_documents(documents)

# Preprocess documents
pipeline.preprocess()

# Detect bias
pipeline.detect_bias()
bias_results = pipeline.results['bias_detection']
print(f"Found {bias_results['biased_documents']} biased documents")
print(f"Average bias score: {bias_results['average_bias_score']:.1%}")

# Continue with other analyses
pipeline.topic_modeling()
pipeline.clustering()

# Export all results
pipeline.export_results('analysis_results.json', format='json')
```

### Example 4: Generate Bias Reports
```python
from analysis.bias_detector import BiasDetector

detector = BiasDetector(device='cpu')

text = """
The successful male candidate will lead our engineering team.
We need someone decisive and competitive in market negotiations.
Nursing and administrative support staff welcome.
"""

analysis = detector.comprehensive_bias_analysis(text)

# Generate detailed report
report = detector.generate_bias_report(analysis)
print(report)

# Access structured results
print("\n=== STRUCTURED ANALYSIS ===")
print(f"Overall Bias Score: {analysis['overall_bias_score']:.1%}")
print(f"Gender Bias Detected: {analysis['gender_bias']['has_gender_bias']}")
print(f"Gender Direction: {analysis['gender_bias']['bias_direction']}")
print(f"Confidence: {analysis['gender_bias']['confidence']:.1%}")

# Show all detected discrimination types
detected_categories = analysis['bias_summary']['discriminatory_categories']
if detected_categories:
    print(f"Discriminatory Categories: {', '.join(detected_categories)}")
else:
    print("No discriminatory language detected")
```

### Example 5: Custom Configuration
```python
from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import PipelineConfig, BiasDetectionConfig

# Create custom configuration
bias_config = BiasDetectionConfig(
    device='cpu',  # or 'cuda' for GPU
    batch_size=16,
    max_length=512
)

config = PipelineConfig(
    bias_detection=bias_config,
    random_state=42
)

# Use custom config
pipeline = TextMiningPipeline(config)
pipeline.load_documents(documents)
pipeline.detect_bias()
```

## �📊 Project Structure

```
poc_text_mining/
├── config/              # Configuration classes
│   ├── settings.py      # Pipeline configuration
│   └── __init__.py
├── utils/              # Utility functions
│   ├── preprocessing.py # Text preprocessing
│   ├── analysis_utils.py # Analysis utilities
│   └── __init__.py
├── analysis/           # Analysis components
│   ├── bias_detector.py     # Bias detection
│   ├── topic_modeler.py     # Topic modeling
│   └── __init__.py
├── models/            # ML models
│   ├── clustering.py         # Document clustering
│   ├── classification.py      # Text classification
│   ├── decision_engine.py     # Decision making
│   └── __init__.py
├── pipelines/         # Main pipeline
│   ├── text_mining_pipeline.py
│   └── __init__.py
├── notebooks/         # Jupyter notebooks
│   └── text_mining_analysis.ipynb
├── data/              # Data directory
├── output/            # Results and reports
├── example_analysis.py # Example usage
├── requirements.txt    # Dependencies
├── QUICKSTART.md      # Quick start guide
├── PROJECT_DOCUMENTATION.md # Full documentation
└── README.md          # This file
```

## 📚 Key Modules

### TextPreprocessor
```python
from utils.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor(config)
cleaned = preprocessor.preprocess("text here")
```

### BiasDetector
```python
from analysis.bias_detector import BiasDetector

# Initialize detector (auto-detects optimized device: GPU or CPU)
detector = BiasDetector(device='cpu')

# Analyze text for bias
text = "The male engineer led the project while the female nurse provided support."
analysis = detector.comprehensive_bias_analysis(text)

# Access results
print(f"Is Biased: {analysis['is_biased']}")
print(f"Bias Score: {analysis['overall_bias_score']:.2%}")
print(f"Gender Bias: {analysis['gender_bias']['has_gender_bias']}")
print(f"Direction: {analysis['gender_bias']['bias_direction']}")
print(f"Confidence: {analysis['gender_bias']['confidence']:.2%}")

# Generate detailed report
report = detector.generate_bias_report(analysis)
print(report)

# Batch processing multiple texts
texts = ["text 1", "text 2", "text 3"]
results = detector.batch_analysis(texts)
```

### BERTopicModeler
```python
from analysis.topic_modeler import BERTopicModeler

modeler = BERTopicModeler(config)
modeler.train(documents)
topics = modeler.get_topic_summary()
```

### DocumentClusterer
```python
from models.clustering import DocumentClusterer

clusterer = DocumentClusterer(config)
clusterer.fit(documents)
metrics = clusterer.evaluate()
```

### DecisionEngine
```python
from models.decision_engine import DecisionEngine

engine = DecisionEngine(config)
decision = engine.analyze_bias_results(analysis)
guidance = engine.generate_guidance(text, analyses)
```

## 📈 Example Output

### Bias Detection
- Overall bias score (0.0 - 1.0) with confidence metrics
- Gender bias direction (male/female/balanced/mixed)
- Discriminatory language categories (age, race, disability, appearance)
- Evidence windows showing where bias was detected
- Semantic similarity scores for each bias pattern

### Topic Analysis
- Topics identified with keywords
- Document-topic distribution
- Topic prevalence percentages

### Clustering Results
- Cluster assignments
- Evaluation metrics
- Cluster-specific keywords
- Distribution statistics

## 🔧 Configuration

Customize analysis behavior:

```python
from config.settings import PipelineConfig, BiasDetectionConfig, TopicModelingConfig

config = PipelineConfig(
    bias_detection=BiasDetectionConfig(
        model_name="bert-base-uncased",
        device='cpu',  # Auto-detects GPU if available, falls back to CPU
        batch_size=32
    ),
    topic_modeling=TopicModelingConfig(
        min_topic_size=5,
        nr_topics=10
    ),
    random_state=42
)

pipeline = TextMiningPipeline(config)
```

### Device Auto-Detection
The pipeline automatically detects available hardware:
- Uses CUDA GPU if NVIDIA GPU is available
- Falls back to CPU if GPU is not available
- Optimizes model loading for detected device

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Full Documentation](PROJECT_DOCUMENTATION.md)** - Complete API reference
- **[Example Notebook](notebooks/text_mining_analysis.ipynb)** - Interactive examples

## ✨ Highlights

### Comprehensive Analysis
- Multi-component pipeline processing
- Configurable analysis parameters
- Real-time progress reporting
- Automatic result aggregation

### Machine Learning
- Multiple classification algorithms
- Unsupervised clustering
- Advanced topic modeling
- Feature importance analysis

### User-Friendly
- Simple Python API
- Example scripts included
- Jupyter notebook examples
- JSON/CSV export options
- Human-readable reports

### Production-Ready
- Scalable architecture
- Configurable components
- Error handling
- Model persistence
- Result caching

## 🎓 Use Cases

1. **Research Project Analysis**: Analyze job postings and research descriptions for bias
2. **Academic Paper Review**: Detect bias in academic writing
3. **Content Moderation**: Identify discriminatory language in user-generated content
4. **Policy Analysis**: Analyze policy documents for bias and fairness
5. **Hiring Analysis**: Review job postings for discriminatory language
6. **Pattern Discovery**: Extract topics and themes from document collections

## 🛠️ Development

### Running Tests
```bash
python example_analysis.py
```

### Generating Reports
```python
pipeline = TextMiningPipeline()
results = pipeline.run_full_pipeline(documents)
pipeline.export_results('report.json', format='json')
```

### Extending Functionality
Each module is designed to be extensible:
- Custom bias detection rules
- Additional classification algorithms
- Domain-specific topic modeling
- Custom clustering parameters

## 🔐 Data Privacy

- Processes data locally
- No external API calls required
- No personal data storage
- Anonymized reporting

## 📝 Requirements

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- CUDA optional (for GPU acceleration)

See `requirements.txt` for full dependency list.

## 🤝 Contributing

Areas for enhancement:
- Additional language support
- Custom bias detection rules
- Web interface development
- Performance optimization
- Additional NLP models

## 📞 Support & Issues

For questions or issues:
1. Check QUICKSTART.md
2. Review PROJECT_DOCUMENTATION.md
3. Check example_analysis.py
4. Review Jupyter notebook examples

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Interactive web dashboard
- [ ] Real-time streaming analysis
- [ ] Custom model fine-tuning
- [ ] API service deployment
- [ ] Advanced visualization tools
- [ ] Benchmark datasets

## 📄 License

Proof of Concept - Educational Use

## 🙏 Acknowledgments

Built with:
- [BERTopic](https://github.com/MaartenGr/BERTopic) - Topic Modeling
- [Transformers](https://huggingface.co/transformers/) - NLP Models
- [scikit-learn](https://scikit-learn.org/) - ML Algorithms
- [spaCy](https://spacy.io/) - NLP
- [NLTK](https://www.nltk.org/) - Text Processing

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Status**: Active Development