# Text Mining and Bias Detection Pipeline

A comprehensive machine learning pipeline for analyzing textual content in research projects to identify patterns, trends, and potential bias issues, including gender bias and discriminatory language detection.

## 📋 Overview

This project implements a sophisticated text mining and natural language processing (NLP) solution that combines multiple machine learning techniques:

- **Classification**: Logistic Regression, Random Forest, SVM for bias classification
- **Regression**: Support for trend analysis in textual patterns
- **Clustering**: K-means, Agglomerative, DBSCAN algorithms for document grouping
- **Topic Modeling**: BERTopic for identifying thematic patterns
- **NLP Technologies**: BERT, Transformers, TensorFlow for deep learning-based analysis
- **Bias Detection**: Gender bias and discriminatory language identification
- **Decision Engine**: Personalized guidance and actionable recommendations

## 🎯 Key Features

### 1. **Text Preprocessing**
- Tokenization and lemmatization using spaCy and NLTK
- Stopword removal and text cleaning
- POS tagging and named entity recognition
- N-gram extraction

### 2. **Bias Detection**
- Gender bias detection (male/female keyword analysis)
- Discriminatory language identification (age, race, disability, appearance)
- ML-based zero-shot classification for bias patterns

### 3. **Topic Modeling**
- BERTopic for semantic topic extraction
- Topic distribution analysis
- Representative document identification
- Topic-keyword relationships

### 4. **Clustering Analysis**
- Multiple clustering algorithms (KMeans, Agglomerative, DBSCAN)
- Cluster evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Cluster keyword extraction
- Document grouping by similarity

### 5. **Classification Models**
- Multiple model training (Logistic Regression, Random Forest, SVM)
- Cross-validation and performance evaluation
- Feature importance analysis
- ROC-AUC scoring for binary classification

### 6. **Decision Engine**
- Confidence-based decision making
- Personalized recommendations
- Priority-based action items
- Comprehensive guidance generation

## 📦 Project Structure

```
poc_text_mining/
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration classes
├── utils/
│   ├── __init__.py
│   └── preprocessing.py            # Text preprocessing utilities
├── analysis/
│   ├── __init__.py
│   ├── bias_detector.py           # Bias detection analysis
│   └── topic_modeler.py           # BERTopic integration
├── models/
│   ├── __init__.py
│   ├── clustering.py              # Clustering algorithms
│   ├── classification.py          # Classification models
│   └── decision_engine.py         # Decision-making logic
├── pipelines/
│   ├── __init__.py
│   └── text_mining_pipeline.py   # Main orchestration pipeline
├── notebooks/
│   └── text_mining_analysis.ipynb # Interactive Jupyter notebook
├── data/                          # Data storage directory
├── output/                        # Results and reports
├── example_analysis.py            # Example usage script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository** (if using git):
```bash
cd poc_text_mining
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download required models**:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

## 💻 Usage

### Quick Start - Example Script

```bash
python example_analysis.py
```

This will:
1. Load sample research project descriptions
2. Run the complete analysis pipeline
3. Generate bias detection reports
4. Create topic models
5. Perform clustering analysis
6. Export results to `output/` directory

### Using the Pipeline Programmatically

```python
from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import PipelineConfig

# Initialize pipeline
config = PipelineConfig()
pipeline = TextMiningPipeline(config)

# Load documents
documents = ["Your research project text here...", "Another project..."]

# Run complete analysis
results = pipeline.run_full_pipeline(documents)

# Generate report
report = pipeline.generate_report()
print(report)

# Export results
pipeline.export_results('results.json', format='json')
```

### Interactive Jupyter Notebook

```bash
jupyter notebook notebooks/text_mining_analysis.ipynb
```

The notebook provides interactive examples for:
- Loading and exploring data
- Bias detection visualization
- Topic modeling analysis
- Clustering evaluation
- Report generation

### Custom Configuration

```python
from config.settings import (
    PipelineConfig, 
    BiasDetectionConfig,
    TopicModelingConfig,
    ClusteringConfig
)

# Create custom configuration
config = PipelineConfig(
    bias_detection=BiasDetectionConfig(
        model_name="bert-base-cased",
        batch_size=32
    ),
    topic_modeling=TopicModelingConfig(
        min_topic_size=5,
        nr_topics=10
    ),
    clustering=ClusteringConfig(
        n_clusters=5,
        algorithm="kmeans"
    )
)

# Use custom config
pipeline = TextMiningPipeline(config)
```

## 🔍 Analysis Components

### 1. Text Preprocessing (`utils/preprocessing.py`)

**TextPreprocessor** class provides:
- Text cleaning (URL, email, HTML removal)
- Lowercase conversion
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
- N-gram extraction
- POS tagging
- Named entity recognition

```python
from utils.preprocessing import TextPreprocessor
from config.settings import TextPreprocessingConfig

preprocessor = TextPreprocessor(TextPreprocessingConfig())
cleaned_text = preprocessor.preprocess("Your text here...")
```

### 2. Bias Detection (`analysis/bias_detector.py`)

**BiasDetector** class provides:
- Gender bias detection
- Discriminatory language identification
- Zero-shot classification for bias patterns
- Comprehensive bias reports

```python
from analysis.bias_detector import BiasDetector

detector = BiasDetector()
analysis = detector.comprehensive_bias_analysis("Research project text...")
report = detector.generate_bias_report(analysis)
```

### 3. Topic Modeling (`analysis/topic_modeler.py`)

**BERTopicModeler** class provides:
- BERTopic model training
- Topic extraction and summarization
- Document-topic assignment
- Topic distribution analysis
- Representative document identification
- Topic searching and visualization

```python
from analysis.topic_modeler import BERTopicModeler

modeler = BERTopicModeler(config)
modeler.train(documents)
topics = modeler.get_topics()
distribution = modeler.get_topic_distribution()
```

### 4. Clustering (`models/clustering.py`)

**DocumentClusterer** class provides:
- Multiple clustering algorithms
- Feature extraction (TF-IDF)
- Cluster evaluation
- Keyword extraction per cluster
- Dynamic cluster assignment

```python
from models.clustering import DocumentClusterer
from config.settings import ClusteringConfig

clusterer = DocumentClusterer(ClusteringConfig(algorithm='kmeans'))
clusterer.fit(documents)
metrics = clusterer.evaluate()
keywords = clusterer.get_cluster_keywords(cluster_id=0)
```

### 5. Classification (`models/classification.py`)

**TextClassifier** class provides:
- Multiple model training
- Performance evaluation
- Feature importance analysis
- Prediction with probability scores
- Classification reports

```python
from models.classification import TextClassifier

classifier = TextClassifier(config)
classifier.prepare_data(texts, labels)
classifier.train()
evaluation = classifier.evaluate()
predictions = classifier.predict(new_texts)
```

### 6. Decision Engine (`models/decision_engine.py`)

**DecisionEngine** class provides:
- Confidence-based decisions
- Personalized recommendations
- Priority-based action items
- Comprehensive guidance generation
- Multi-source analysis aggregation

```python
from models.decision_engine import DecisionEngine

engine = DecisionEngine(config)
decision = engine.analyze_bias_results(bias_analysis)
guidance = engine.generate_guidance(text, all_analyses)
```

## 📊 Output and Results

The pipeline generates comprehensive outputs including:

### 1. **JSON Results** (`output/analysis_results.json`)
- Structured analysis results
- All metrics and scores
- Detailed findings per component
- Timestamp and metadata

### 2. **Text Report** (`output/analysis_report.txt`)
- Human-readable summary
- Key findings
- Recommendations
- Statistical summary

### 3. **Console Output**
- Real-time progress updates
- Summary statistics
- Actionable insights

## 🎨 Visualization and Reports

The pipeline generates:
- Bias distribution charts
- Topic word clouds
- Cluster similarity matrices
- Performance evaluation plots
- Recommendation prioritization

(Visualizations available in Jupyter notebook)

## 📈 Performance Metrics

### Bias Detection
- Overall bias score (0.0 - 1.0)
- Gender bias direction (male/female/balanced)
- Discriminatory language categories
- Sentiment polarity

### Clustering
- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

### Classification
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (for binary classification)

### Topic Modeling
- Number of topics discovered
- Topic coherence
- Document-topic distribution

## 🔧 Configuration Options

Key configuration parameters:

```python
TextPreprocessingConfig:
  - lowercase: bool
  - remove_punctuation: bool
  - remove_stopwords: bool
  - min_word_length: int
  - language: str

BiasDetectionConfig:
  - model_name: str (BERT variant)
  - batch_size: int
  - threshold: float

TopicModelingConfig:
  - min_topic_size: int
  - nr_topics: Optional[int]
  - embedding_model: str

ClusteringConfig:
  - n_clusters: int
  - algorithm: str (kmeans, agglomerative, dbscan)

DecisionEngineConfig:
  - confidence_threshold: float
  - recommendation_count: int
```

## 🤝 Integration with NLP Models

The pipeline integrates with state-of-the-art models:

### Transformers
- **BERT**: For contextual embeddings and zero-shot classification
- **GPT**: For language understanding (optional)
- **SentenceTransformers**: For semantic similarity

### Deep Learning
- **TensorFlow/Keras**: For potential custom model training
- **PyTorch**: Underlying library for transformers
- **PyTorch Lightning**: For structured training workflows

### Topic Modeling
- **BERTopic**: State-of-the-art topic modeling
- **UMAP**: Dimensionality reduction
- **HDBSCAN**: Clustering for topics

## 📚 Example Use Cases

### 1. Research Project Analysis
Analyze job postings, research descriptions, or grant proposals for bias:

```python
project_descriptions = [...list of texts...]
results = pipeline.run_full_pipeline(project_descriptions)
```

### 2. Gender Bias Detection
Identify and quantify gender bias in academic papers:

```python
papers = [...academic texts...]
bias_detector = BiasDetector()
for paper in papers:
    analysis = bias_detector.comprehensive_bias_analysis(paper)
    if analysis['is_biased']:
        print(f"Gender bias detected: {analysis['gender_bias']['bias_direction']}")
```

### 3. Pattern Discovery
Discover topics and patterns in large document collections:

```python
documents = [...large collection...]
modeler = BERTopicModeler(config)
modeler.train(documents)
topics = modeler.get_topics()
```

## 🛠️ Troubleshooting

### CUDA Out of Memory
```python
config = PipelineConfig()
config.bias_detection.device = "cpu"
pipeline = TextMiningPipeline(config)
```

### Slow Performance
- Reduce batch_size in BiasDetectionConfig
- Use CPU instead of GPU for small datasets
- Reduce max_features in vectorizer

### Model Download Issues
```bash
python -m spacy download en_core_web_sm
python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"
```

## 📖 Documentation

Each module includes comprehensive docstrings. Key classes:

- `TextMiningPipeline` - Main orchestrator
- `TextPreprocessor` - Text cleaning
- `BiasDetector` - Bias analysis
- `BERTopicModeler` - Topic extraction
- `DocumentClusterer` - Document grouping
- `TextClassifier` - ML-based classification
- `DecisionEngine` - Personalized recommendations

## 🔐 Data Privacy

The pipeline:
- Processes data locally
- Does not store personal information
- Can work offline (except first model download)
- Generates anonymized reports

## 📝 License

This project is part of a proof-of-concept for text mining and bias detection.

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- Additional language support
- Custom bias detection rules
- Visualization improvements
- Performance optimization
- Integration with additional NLP models
- Domain-specific bias detection

## ✉️ Support

For issues, questions, or improvements:
1. Check example notebooks for usage
2. Review configuration options
3. Consult module docstrings
4. Run diagnostics with example_analysis.py

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Custom bias detection rules
- [ ] Interactive web interface
- [ ] Real-time streaming analysis
- [ ] Advanced visualization dashboard
- [ ] Custom model fine-tuning
- [ ] API service deployment
- [ ] Benchmark datasets

---

**Last Updated**: March 2026
**Version**: 1.0.0
