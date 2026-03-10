# Architecture Overview

## System Architecture

The Text Mining and Bias Detection Pipeline is built with a modular, layered architecture designed for scalability, maintainability, and extensibility.

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  (example_analysis.py, Jupyter Notebooks)                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Pipeline Orchestration Layer                   │
│  (TextMiningPipeline - Main Coordinator)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
┌───────▼────┐ ┌──────▼─────┐ ┌─────▼──────┐ ┌────▼─────┐
│ Analysis   │ │   Models   │ │ Decision   │ │ Config  │
│ Components │ │ Components │ │ Engine     │ │Management│
└───────┬────┘ └──────┬─────┘ └─────┬──────┘ └────┬─────┘
        │             │             │             │
        └─────────────┼─────────────┼─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼──────┐ ┌───▼──────┐ ┌────▼──────┐
│ Preprocessing│ │ Utilities │ │  Utils    │
│  (NLTK,      │ │(Data Mgmt)│ │(Analysis) │
│  spaCy)      │ └──────────┘ └───────────┘
└──────────────┘
```

## Component Breakdown

### 1. Configuration Layer (`config/`)

**Purpose**: Centralized configuration management

**Components**:
- `settings.py` - Configuration dataclasses for all pipeline components

**Key Classes**:
- `PipelineConfig` - Main configuration container
- `TextPreprocessingConfig` - Preprocessing settings
- `BiasDetectionConfig` - Bias detection parameters
- `TopicModelingConfig` - Topic modeling settings
- `ClusteringConfig` - Clustering algorithm parameters
- `ClassificationConfig` - Classification model settings
- `DecisionEngineConfig` - Decision-making parameters

**Responsibilities**:
- Provide sensible defaults
- Allow configuration overrides
- Validate configuration parameters
- Centralize tuning parameters

### 2. Preprocessing Layer (`utils/preprocessing.py`)

**Purpose**: Text cleaning and feature extraction

**Key Class**: `TextPreprocessor`

**Main Methods**:
- `clean_text()` - Remove URLs, emails, HTML
- `remove_punctuation()` - Punctuation removal
- `lowercase()` - Text normalization
- `tokenize()` - Word-level tokenization (spaCy)
- `remove_stopwords()` - Stopword filtering
- `lemmatize()` - Word lemmatization
- `preprocess()` - Full pipeline
- `preprocess_batch()` - Process multiple texts
- `extract_sentences()` - Sentence tokenization
- `extract_ngrams()` - N-gram extraction
- `extract_pos_tags()` - POS tagging
- `extract_entities()` - Named entity recognition

**Dependencies**:
- NLTK (tokenization, lemmatization, stopwords)
- spaCy (advanced NLP, POS tagging, NER)
- TextBlob (sentiment analysis auxiliary)

**Output**: Cleaned, normalized text for downstream processing

### 3. Analysis Layer (`analysis/`)

#### 3.1 Bias Detector (`bias_detector.py`)

**Purpose**: Detect gender bias and discriminatory language

**Key Class**: `BiasDetector`

**Main Methods**:
- `detect_gender_bias()` - Gender bias analysis
- `detect_discriminatory_language()` - Discriminatory term detection
- `sentiment_and_tone_analysis()` - Sentiment scoring
- `toxic_language_detection()` - Toxic content detection
- `ml_based_bias_detection()` - Zero-shot classification
- `comprehensive_bias_analysis()` - Full pipeline analysis
- `batch_analysis()` - Process multiple texts
- `generate_bias_report()` - Human-readable report

**Detection Methods**:
1. **Keyword-based**: Manual keyword lists for gender and discriminatory terms
2. **Sentiment Analysis**: Using transformer models
3. **Toxic Language**: NSFW text classifier
4. **Zero-shot Classification**: BART for pattern matching

**Output**: Bias scores, detected patterns, confidence levels

#### 3.2 Topic Modeler (`topic_modeler.py`)

**Purpose**: Extract semantic topics from documents

**Key Class**: `BERTopicModeler`

**Main Methods**:
- `train()` - Train BERTopic model
- `get_topics()` - Get topic information
- `get_topic_summary()` - Summarized topic data
- `predict()` - Predict topics for new documents
- `search_topics()` - Find similar topics
- `get_topic_distribution()` - Topic prevalence
- `get_representative_documents()` - Key exemplars per topic
- `generate_topic_report()` - Report generation

**Technology Stack**:
- BERTopic (topic modeling framework)
- Sentence Transformers (embedding models)
- UMAP (dimensionality reduction)
- HDBSCAN (clustering)
- CountVectorizer (feature extraction)

**Output**: Topics, keywords, document assignments

### 4. Models Layer (`models/`)

#### 4.1 Clustering (`clustering.py`)

**Purpose**: Unsupervised document grouping

**Key Class**: `DocumentClusterer`

**Supported Algorithms**:
- KMeans (configurable k)
- Agglomerative (hierarchical)
- DBSCAN (density-based)

**Main Methods**:
- `fit()` - Fit clustering model
- `predict()` - Assign clusters to new documents
- `get_cluster_distribution()` - Cluster statistics
- `get_cluster_documents()` - Documents per cluster
- `get_cluster_keywords()` - Cluster-specific keywords
- `evaluate()` - Quality metrics
- `generate_clustering_report()` - Report generation

**Evaluation Metrics**:
- Silhouette Score (internal validity)
- Davies-Bouldin Index (cluster separation)
- Calinski-Harabasz Score (cluster density)

**Output**: Cluster assignments, metrics, keywords

#### 4.2 Classification (`classification.py`)

**Purpose**: ML-based bias classification

**Key Class**: `TextClassifier`

**Supported Models**:
- Logistic Regression (baseline)
- Random Forest (ensemble)
- SVM (support vector machine)

**Main Methods**:
- `prepare_data()` - Data splitting and vectorization
- `train()` - Model training
- `evaluate()` - Performance metrics
- `predict()` - Label prediction with probabilities
- `get_feature_importance()` - Important features
- `get_coefficients()` - Model coefficients
- `generate_classification_report()` - Report generation

**Output**: Predictions, probabilities, performance metrics

#### 4.3 Decision Engine (`decision_engine.py`)

**Purpose**: Transform analysis results into actionable decisions

**Key Class**: `DecisionEngine`

**Main Methods**:
- `analyze_bias_results()` - Bias-based decisions
- `analyze_classification_results()` - Model-based decisions
- `analyze_topic_results()` - Topic-based decisions
- `generate_guidance()` - Comprehensive guidance
- `_format_decision()` - Output formatting

**Decision Logic**:
1. **Confidence Thresholding**: Minimum confidence for decisions
2. **Multi-source Aggregation**: Combine multiple analyses
3. **Priority Assignment**: High/Medium/Low based on severity
4. **Recommendation Generation**: Context-aware suggestions

**Output**: Actionable recommendations, priority rankings

### 5. Pipeline Layer (`pipelines/`)

**Purpose**: Orchestrate all components into a unified workflow

**Key Class**: `TextMiningPipeline`

**Main Methods**:
- `load_documents()` - Input data loading
- `preprocess()` - Text preprocessing
- `detect_bias()` - Bias analysis
- `topic_modeling()` - Topic extraction
- `clustering()` - Document clustering
- `classification()` - ML classification (optional)
- `decision_making()` - Decision generation
- `run_full_pipeline()` - Complete workflow
- `generate_report()` - Comprehensive reporting
- `export_results()` - Result export (JSON/TXT)

**Workflow**:
```
1. Load → 2. Preprocess → 3. Detect Bias
    │
    └─→ 4. Topic Modeling
    └─→ 5. Clustering
    └─→ 6. Classification (optional)
    │
    └─→ 7. Decision Making → 8. Report
```

### 6. Utilities Layer (`utils/`)

**Purpose**: Support functions for data and analysis management

**Key Classes**:
- `TextPreprocessor` - Text preprocessing
- `DataLoader` - Load various data formats
- `Visualizer` - Create charts and plots
- `ReportGenerator` - Generate reports
- `ResultsManager` - Save/load results
- `TextAnalysisUtils` - Text statistics and filtering

**Responsibilities**:
- Data I/O (CSV, JSON, TXT)
- Visualization generation
- Report generation
- Result persistence
- Text statistics

## Data Flow

### Single Document Analysis
```
Raw Text
   ↓
Preprocessing (tokenize, clean, lemmatize)
   ↓
Bias Detection (keyword matching, models)
   ↓
Decision Engine (confidence, recommendations)
   ↓
Report Generation
```

### Batch Analysis
```
Raw Documents (List)
   ↓
Vectorization (TF-IDF)
   ↓
├─→ Bias Detection (per document)
├─→ Topic Modeling (aggregate)
├─→ Clustering (similarity)
├─→ Classification (ML models)
   ↓
Aggregation & Analysis
   ↓
Decision Making
   ↓
Pipeline Report
```

## Key Design Patterns

### 1. Configuration Pattern
- Dataclass-based configuration objects
- Hierarchical configuration (global → component-specific)
- Easy overrides and customization

### 2. Fluent Interface Pattern
- Method chaining for pipeline operations
- `pipeline.load_documents().preprocess().detect_bias()`

### 3. Strategy Pattern
- Multiple algorithms for clustering
- Pluggable model implementations
- Easy to swap components

### 4. Factory Pattern
- Component creation in pipeline
- Configuration-driven instantiation

### 5. Aggregator Pattern
- Results aggregation in decision engine
- Multi-source decision making

## Dependencies and Integration

### External Libraries
```
Visualization:
  - Matplotlib, Seaborn, Plotly

NLP/Text:
  - NLTK, spaCy, TextBlob
  - Transformers, Sentence-Transformers

Machine Learning:
  - scikit-learn (clustering, classification)
  - BERTopic (topic modeling)
  - UMAP, HDBSCAN (for BERTopic)

Deep Learning:
  - PyTorch, TensorFlow, PyTorch Lightning

Data Handling:
  - Pandas, NumPy, SciPy
```

### Model Dependencies
```
Bias Detection:
  - BERT models (sentiment, toxic text)
  - Zero-shot classification models

Topic Modeling:
  - Sentence Transformers (embeddings)
  - UMAP (dimensionality reduction)
  - HDBSCAN (clustering)

NLP Processing:
  - spaCy (en_core_web_sm)
  - NLTK data (punkt, stopwords, wordnet)
```

## Extensibility Points

### 1. Custom Bias Detection Rules
- Add to `gender_bias_keywords` and `discriminatory_keywords` dictionaries
- Implement custom detection methods in `BiasDetector`

### 2. Additional Classification Models
- Add to `_create_models()` in `TextClassifier`
- Implement scikit-learn compatible interface

### 3. Custom Clustering Algorithms
- Extend `fit_*()` methods in `DocumentClusterer`
- Support sklearn-compatible algorithms

### 4. Domain-Specific Configuration
- Create custom `PipelineConfig` subclasses
- Override component configurations

### 5. Custom Visualizations
- Extend `Visualizer` class
- Add domain-specific plots

## Performance Considerations

### Memory Optimization
- Batch processing for large datasets
- Configurable batch sizes
- Feature dimensionality reduction

### Speed Optimization
- GPU support (CUDA) for transformers
- Vectorized operations (NumPy, Pandas)
- Model caching
- Parallel processing where applicable

### Scalability
- Streaming document processing
- Distributed clustering support
- Model optimization techniques

## Error Handling

### Graceful Degradation
- Fallback to CPU if GPU unavailable
- Default models when custom unavailable
- Error messages guide users to solutions

### Validation
- Input validation in preprocessing
- Configuration validation
- Output verification

## Testing Strategy

### Unit Testing
- Individual component testing
- Mock external dependencies

### Integration Testing
- Pipeline end-to-end
- Component interaction
- Data flow verification

### Performance Testing
- Scalability testing
- Resource usage profiling
- Speed benchmarking

## Future Architecture Enhancements

1. **Microservices**: Separate components into services
2. **Message Queue**: Asynchronous processing
3. **Caching Layer**: Results memoization
4. **Monitoring**: Performance and accuracy tracking
5. **API Gateway**: RESTful API interface
6. **Distributed Processing**: Spark/Ray integration

---

This architecture provides a solid foundation for text mining and bias detection while maintaining flexibility for future enhancements and customizations.
