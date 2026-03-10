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
- Sentiment and tone analysis
- Toxic language detection using transformer models
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

### Basic Usage
```python
from pipelines.text_mining_pipeline import TextMiningPipeline

# Your documents
documents = ["Research project text 1...", "Research project text 2..."]

# Run analysis
pipeline = TextMiningPipeline()
results = pipeline.run_full_pipeline(documents)

# Generate report
print(pipeline.generate_report())

# Export results
pipeline.export_results('results.json', format='json')
```

### Run Examples
```bash
python example_analysis.py
jupyter notebook notebooks/text_mining_analysis.ipynb
```

## 📊 Project Structure

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

detector = BiasDetector()
analysis = detector.comprehensive_bias_analysis("text")
print(detector.generate_bias_report(analysis))
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
- Overall bias score (0.0 - 1.0)
- Gender bias direction (male/female/balanced)
- Discriminatory language categories
- Sentiment analysis
- Toxic language detection

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
        model_name="bert-base-cased",
        batch_size=32
    ),
    topic_modeling=TopicModelingConfig(
        min_topic_size=5,
        nr_topics=10
    )
)

pipeline = TextMiningPipeline(config)
```

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