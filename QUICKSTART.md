# Quick Start Guide

Get started with the Text Mining and Bias Detection Pipeline in minutes.

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

### 3. Run Example
```bash
python example_analysis.py
```

## Basic Usage

### Analyze Your Own Data

```python
from pipelines.text_mining_pipeline import TextMiningPipeline

# Your documents
documents = [
    "Text of research project 1...",
    "Text of research project 2...",
    # ... more documents
]

# Create and run pipeline
pipeline = TextMiningPipeline()
results = pipeline.run_full_pipeline(documents)

# View results
print(pipeline.generate_report())

# Export results
pipeline.export_results('results.json', format='json')
```

## Common Tasks

### 1. Detect Bias in Single Document
```python
from analysis.bias_detector import BiasDetector

detector = BiasDetector()
analysis = detector.comprehensive_bias_analysis("Your text here...")
print(detector.generate_bias_report(analysis))
```

### 2. Extract Topics
```python
from analysis.topic_modeler import BERTopicModeler

modeler = BERTopicModeler(config=None)
modeler.train(your_documents)
topics = modeler.get_topic_summary()
```

### 3. Cluster Documents
```python
from models.clustering import DocumentClusterer
from config.settings import ClusteringConfig

clusterer = DocumentClusterer(ClusteringConfig(n_clusters=5))
clusterer.fit(your_documents)
distribution = clusterer.get_cluster_distribution()
```

### 4. Train Classification Model
```python
from models.classification import TextClassifier

classifier = TextClassifier(config=None)
classifier.prepare_data(texts, labels)
classifier.train()
evaluation = classifier.evaluate()
```

## Working with Data

### Load Data from CSV
```python
from utils.analysis_utils import DataLoader

df = DataLoader.load_csv('data.csv')
documents = df['text'].tolist()

# Run pipeline
pipeline = TextMiningPipeline()
results = pipeline.run_full_pipeline(documents)
```

### Load from Text File
```python
from utils.analysis_utils import DataLoader

# Load documents separated by double newlines
texts = DataLoader.load_txt('documents.txt', delimiter='\n\n')
```

## Configuration

### Customize Pipeline
```python
from config.settings import (
    PipelineConfig, 
    BiasDetectionConfig,
    TopicModelingConfig,
    ClusteringConfig
)

config = PipelineConfig(
    bias_detection=BiasDetectionConfig(batch_size=32),
    topic_modeling=TopicModelingConfig(nr_topics=10),
    clustering=ClusteringConfig(n_clusters=5, algorithm='kmeans')
)

pipeline = TextMiningPipeline(config)
results = pipeline.run_full_pipeline(documents)
```

## Visualization

### Using Matplotlib
```python
from utils.analysis_utils import Visualizer

# Plot bias distribution
fig = Visualizer.plot_bias_distribution(bias_scores, threshold=0.5)
plt.show()

# Plot cluster distribution
fig = Visualizer.plot_cluster_distribution(cluster_dist)
plt.show()
```

## Web Usage with Jupyter

```bash
jupyter notebook notebooks/text_mining_analysis.ipynb
```

The notebook provides:
- Interactive data exploration
- Visualization of results
- Step-by-step analysis
- Export functionality

## Output Examples

### Bias Detection Output
```
Gender Bias Analysis:
  - Direction: MALE
  - Male Keywords: 4
  - Female Keywords: 1
  - Confidence: 0.80

Overall Bias Score: 0.65%
Is Biased: True
```

### Topic Modeling Output
```
Topics identified: 5

Topic 0: Keywords - research, data, analysis, findings
Topic 1: Keywords - team, leadership, management, project
Topic 2: Keywords - innovation, technology, development, system
...
```

### Clustering Output
```
Cluster Distribution:
  Cluster 0: 15 documents (50%)
  Cluster 1: 10 documents (33.33%)
  Cluster 2: 5 documents (16.67%)

Silhouette Score: 0.5234
Davies Bouldin Score: 0.8765
```

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Use CPU instead
```python
config = PipelineConfig()
config.bias_detection.device = "cpu"
```

### Issue: Slow Processing
**Solution**: Reduce batch size and document count
```python
config.bias_detection.batch_size = 16
documents = documents[:100]  # Process subset first
```

### Issue: Missing Model Files
**Solution**: Download models
```bash
python -m spacy download en_core_web_sm
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

## Next Steps

1. **Explore the notebook**: `notebooks/text_mining_analysis.ipynb`
2. **Read full docs**: `PROJECT_DOCUMENTATION.md`
3. **Check examples**: `example_analysis.py`
4. **Experiment with your data**: Create a Python script similar to `example_analysis.py`

## Performance Tips

- **GPU Usage**: Processing is faster with CUDA-enabled GPU
- **Large Datasets**: Process in batches of 100-1000 documents
- **Memory**: Reduce `batch_size` in configuration if running out of memory
- **Cache**: Results are cached automatically in the pipeline

## Getting Help

- Check example scripts
- Review docstrings in source code
- Consult Jupyter notebook for interactive examples
- Refer to configuration options in `config/settings.py`

## What's Next?

- Customize bias detection rules
- Add domain-specific keywords
- Fine-tune models for your data
- Integrate with your systems
- Build custom visualizations

Happy analyzing! 🎉
