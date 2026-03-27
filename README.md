# Text Mining and Bias Detection — Proof of Concept

A proof-of-concept pipeline for detecting **gender bias** and **discriminatory language** in textual content (research projects, job postings, academic papers, PDFs). It supports **English** and **Spanish** and offers two complementary detection approaches:

- **Keyword-based** (`BiasDetector`) — fast, no GPU, pattern/keyword matching.
- **ML-based** (`MLBiasDetector`) — combines a fine-tuned transformer (distilroberta-bias), zero-shot classification (BART / XLM-R), and keyword analysis.

## Project Structure

```
poc_text_mining/
├── config/
│   └── settings.py              # Dataclass-based configuration for all components
├── utils/
│   ├── preprocessing.py         # TextPreprocessor — NLTK + spaCy text cleaning
│   └── analysis_utils.py        # DataLoader, Visualizer, ReportGenerator, ResultsManager
├── analysis/
│   ├── bias_keywords.py         # Shared keyword dictionaries & patterns (EN + ES)
│   ├── bias_detector.py         # BiasDetector  — lightweight keyword-based detection
│   ├── ml_bias_detector.py      # MLBiasDetector — transformer + zero-shot detection
│   └── topic_modeler.py         # BERTopicModeler — semantic topic extraction
├── models/
│   ├── clustering.py            # DocumentClusterer (KMeans, Agglomerative, DBSCAN)
│   ├── classification.py        # TextClassifier (LogisticRegression, RF, SVM)
│   └── decision_engine.py       # DecisionEngine — recommendations from analysis results
├── pipelines/
│   └── text_mining_pipeline.py  # TextMiningPipeline — full workflow orchestrator
├── sample_code/
│   ├── keyword_analysis.py      # CLI: keyword-based bias detection on a PDF
│   ├── ml_analysis.py           # CLI: ML-based bias detection on a PDF
│   └── generate_report.py       # Compare keyword vs ML results, generate charts
├── data/                        # Input PDFs for analysis
├── output/                      # Generated results (JSON) and charts (PNG)
└── requirements.txt             # Python dependencies
```

## Setup

### 1. Create a virtual environment

```bash
cd poc_text_mining
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLP models

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm
```

## Usage

### Keyword-based bias detection on a PDF

```bash
python sample_code/keyword_analysis.py path-to-file.pdf
OR
python sample_code/keyword_analysis.py path-to-file.pdf --language spanish (IF text is in Spanish)
```

Analyses each page for gender bias and discriminatory language using keyword matching and regex patterns. Results are saved to `output/`.

### ML-based bias detection on a PDF

```bash
python sample_code/ml_analysis.py path-to-file.pdf
OR
python sample_code/ml_analysis.py path-to-file.pdf --language spanish (IF text is in Spanish)
```

Combines a fine-tuned bias classifier, zero-shot categorisation, and keyword analysis. Results are saved to `output/`.


### Full pipeline (programmatic)

```python
from pipelines.text_mining_pipeline import TextMiningPipeline

pipeline = TextMiningPipeline()
results = pipeline.run_full_pipeline(["Your document text...", "Another document..."])

print(pipeline.generate_report())
pipeline.export_results('results.json', format='json')
```

### Interactive notebooks

```bash
jupyter notebook notebooks/text_mining_analysis.ipynb
```

## Key Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `analysis/bias_detector.py` | `BiasDetector` | Keyword/pattern gender bias & discriminatory language detection |
| `analysis/ml_bias_detector.py` | `MLBiasDetector` | Transformer + zero-shot + keyword combined detection |
| `analysis/bias_keywords.py` | — | Shared keyword dictionaries, patterns, severity levels (EN + ES) |
| `analysis/topic_modeler.py` | `BERTopicModeler` | BERTopic semantic topic extraction |
| `models/clustering.py` | `DocumentClusterer` | KMeans / Agglomerative / DBSCAN document grouping |
| `models/classification.py` | `TextClassifier` | Logistic Regression / Random Forest / SVM classification |
| `models/decision_engine.py` | `DecisionEngine` | Converts analysis results into actionable recommendations |
| `pipelines/text_mining_pipeline.py` | `TextMiningPipeline` | End-to-end orchestrator (preprocess → detect → model → report) |
| `utils/preprocessing.py` | `TextPreprocessor` | Text cleaning, tokenisation, lemmatisation, NER, POS tagging |
| `utils/analysis_utils.py` | `DataLoader`, `Visualizer`, `ReportGenerator`, `ResultsManager` | I/O, charts, reports, result persistence |
| `config/settings.py` | `PipelineConfig` | Centralised configuration with sensible defaults |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Set `config.bias_detection.device = "cpu"` or use `BiasDetector` (keyword-only) instead |
| Slow performance | Reduce `batch_size`, use CPU for small datasets |
| Model download fails | Run `python -m spacy download en_core_web_sm` and check internet connection |
| Missing NLTK data | Run the NLTK download commands from the setup section |

## Requirements

- Python 3.8+
- 4 GB RAM minimum (8 GB+ recommended)
- CUDA optional (for GPU-accelerated transformer inference)
