# Text Mining for Bias Detection

A pipeline for detecting **gender bias** and **discriminatory language** in PDF research project documents. It supports **English** and **Spanish**, offers two complementary detection approaches, and ships with a **Streamlit web app** for uploading and analysing documents from the browser.

- **Keyword-based** (`BiasDetector`): fast, no models to download, pattern/keyword matching.
- **ML-based** (`MLBiasDetector`): combines a fine-tuned transformer classifier ([`himel7/bias-detector`](https://huggingface.co/himel7/bias-detector)), zero-shot classification (BART / XLM-R), and keyword analysis.

The web app is built on the **GENDERVISION-AI** project, conducted at the Instituto Universitario de Estudios de las Mujeres (IUEM), Universidad de La Laguna.

## Project Structure

```
poc_text_mining/
├── .streamlit/
│   └── config.toml              # Streamlit theme (light base, ULL primary colour)
├── config/
│   └── settings.py              # Dataclass-based configuration for all components
├── utils/
│   ├── preprocessing.py         # TextPreprocessor: NLTK + spaCy text cleaning
│   └── bias_analysis.py         # Web app helpers: PDF extraction, language detection, detection runners
├── analysis/
│   ├── bias_keywords.py         # Shared keyword dictionaries & patterns (EN + ES)
│   ├── bias_detector.py         # BiasDetector: lightweight keyword-based detection
│   └── ml_bias_detector.py      # MLBiasDetector: transformer + zero-shot detection
├── webapp/
│   └── app.py                   # Streamlit web app
├── images/                      # IUEM and ULL logos used in the web app header/footer
├── sample_code/
│   ├── keyword_analysis.py      # CLI: keyword-based bias detection on a PDF
│   └── ml_analysis.py           # CLI: ML-based bias detection on a PDF
├── data/                        # Input PDFs for analysis (git-ignored)
├── output/                      # Generated JSON results from the CLI scripts (git-ignored)
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
python -m spacy download es_core_news_sm
```

Both spaCy models are needed: the web app picks the language automatically, so it may load either one.

## Usage

All commands are run from the repository root.

### Web app

```bash
streamlit run webapp/app.py
```

Streamlit opens the app in your browser (by default at http://localhost:8501). To analyse a document:

1. Upload a PDF.
2. Click **🗝️ Keyword based analysis** or **🧠 ML Model based analysis**.
3. Review the results:
   - **Summary metrics**: detected language, total pages, biased pages, neutral pages and overall bias rate.
   - **Page breakdown**: one expandable entry per page with its status (biased / neutral), severity, gender bias direction, matched male/female keywords, discriminatory terms by category (keyword mode) or the model's flag and primary bias category (ML mode), plus a text snippet.

Notes:

- **Language is detected automatically.** The app samples the first 3 pages with `langdetect`. Spanish documents are analysed as Spanish; any other language falls back to English.
- **The ML model runs on CPU** and is cached in memory per language, so only the first ML analysis for each language is slow. The first run also downloads the transformer models from Hugging Face (several GB, mostly the zero-shot model).
- Uploading a different file clears the previous results.
- Results are shown in the browser only. The web app does not write files to `output/`.

### Keyword-based bias detection on a PDF (CLI)

```bash
python sample_code/keyword_analysis.py path-to-file.pdf
python sample_code/keyword_analysis.py path-to-file.pdf --language spanish   # Spanish text
```

Analyses each page for gender bias and discriminatory language using keyword matching and regex patterns. Results are saved to `output/<pdf_name> - keyword_bias_results.json`.

### ML-based bias detection on a PDF (CLI)

```bash
python sample_code/ml_analysis.py path-to-file.pdf
python sample_code/ml_analysis.py path-to-file.pdf --language spanish   # Spanish text
```

Combines a fine-tuned bias classifier, zero-shot categorisation, and keyword analysis. Results are saved to `output/<pdf_name> - ml_bias_results.json`.

Unlike the web app, the CLI scripts do **not** detect the language: pass `--language spanish` for Spanish documents (the default is English).

## How It Works

Each PDF page is treated as an independent document (empty pages are skipped):

**PDF → per-page text → clean (URLs, emails, HTML removed) → detect → score → severity**

- Each signal produces a score between 0 and 1. The signals are combined in a weighted sum (capped at 1.0), and a page is flagged as biased when the score reaches the threshold (default `0.25`).
  - Keyword detector weights: gender 0.6, discriminatory language 0.4.
  - ML detector weights: fine-tuned classifier 0.30, zero-shot 0.30, gender keywords 0.20, discriminatory keywords 0.20.
- Positive-context phrases (e.g. "gender equality") lower the score, so text that merely _discusses_ gender is less likely to be flagged.
- For Spanish, the fine-tuned classifier is skipped (it is English-only); zero-shot uses XLM-R (`joeddav/xlm-roberta-large-xnli`) instead of BART (`facebook/bart-large-mnli`).

## Key Modules

| Module                         | Class              | Purpose                                                                    |
| ------------------------------ | ------------------ | -------------------------------------------------------------------------- |
| `webapp/app.py`                | —                  | Streamlit UI: upload, run analysis, display summary and per-page results   |
| `utils/bias_analysis.py`       | —                  | PDF text extraction, language auto-detection, keyword/ML detection runners |
| `analysis/bias_detector.py`    | `BiasDetector`     | Keyword/pattern gender bias & discriminatory language detection            |
| `analysis/ml_bias_detector.py` | `MLBiasDetector`   | Transformer + zero-shot + keyword combined detection                       |
| `analysis/bias_keywords.py`    | —                  | Shared keyword dictionaries, patterns, severity levels (EN + ES)           |
| `utils/preprocessing.py`       | `TextPreprocessor` | Text cleaning, tokenisation, lemmatisation, NER, POS tagging               |
| `config/settings.py`           | `PipelineConfig`   | Centralised configuration with sensible defaults                           |

## Troubleshooting

| Problem                                             | Solution                                                                                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| "No extractable text was found in the provided PDF" | The PDF is likely scanned/image-only. Run it through OCR first; text extraction does not do OCR.             |
| First ML analysis takes a long time                 | The transformer models are downloading and loading. Later runs for the same language reuse the cached model. |
| Wrong language detected                             | Detection uses the first 3 pages. For the CLI, set `--language` explicitly.                                  |
| `OSError: Can't find model 'es_core_news_sm'`       | Run `python -m spacy download es_core_news_sm` (or `en_core_web_sm` for English).                            |
| Terminal shows `Examining the path of transformers... ModuleNotFoundError: No module named 'torchvision'` while the web app works | Harmless: Streamlit's auto-reload file watcher probes every loaded module, which makes `transformers` try to import optional image models. Run `streamlit run webapp/app.py --server.fileWatcherType none`, or add `[server]` / `fileWatcherType = "none"` to `.streamlit/config.toml`. Either way the app no longer reloads when you save a file, so restart it after editing code. |
| Web app doesn't use the purple theme                | Run Streamlit from the repository root so `.streamlit/config.toml` is picked up.                             |
| CUDA out of memory                                  | The web app and CLI already use CPU. If you instantiate `MLBiasDetector` yourself, pass `device="cpu"`.      |
| Slow performance                                    | Use the keyword-based analysis, which loads no models.                                                       |
| Model download fails                                | Check your internet connection; Hugging Face and spaCy models are downloaded on first use.                   |
| Missing NLTK data                                   | Run the NLTK download commands from the setup section.                                                       |

## Requirements

- Python 3.8+
- 4 GB RAM minimum (8 GB+ recommended for the ML-based analysis)
- Several GB of free disk space for the Hugging Face model cache
- Internet access on first run (model downloads)
- CUDA optional: everything runs on CPU by default
