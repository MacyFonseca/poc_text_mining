# Text Mining for Bias Detection

A pipeline for detecting **gender bias** and **discriminatory language** in PDF research project documents. It supports **English** and **Spanish**, offers two complementary detection approaches, and ships with a **Streamlit web app** for uploading and analysing documents from the browser.

- **Keyword-based** (`BiasDetector`): fast, no models to download, pattern/keyword matching.
- **ML-based** (`MLBiasDetector`): combines a fine-tuned transformer classifier ([`himel7/bias-detector`](https://huggingface.co/himel7/bias-detector)), zero-shot classification (BART / XLM-R), and keyword analysis.
- **LLM review** (`LLMBiasDetector`, optional): a local LLM served by [Ollama](https://ollama.com) quotes each problematic passage, explains it and proposes an inclusive rewrite. Nothing leaves the machine.

Every detector reports **findings**: the exact passage, its type (e.g. generic masculine, gender stereotype, outdated term), an explanation and an inclusive-language suggestion.

Improvement work is tracked in [`ROADMAP.md`](ROADMAP.md).

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
│   ├── nlp.py                   # Cached spaCy pipeline per language
│   ├── pdf_extraction.py        # pypdf extraction, header/footer removal, stops at the bibliography
│   └── bias_analysis.py         # PDF -> per-page analysis (per-page language detection), shared by app and CLI
├── analysis/
│   ├── bias_keywords.py         # Shared keyword dictionaries, patterns, sentence-level rules (EN + ES)
│   ├── bias_detector.py         # BiasDetector: lightweight keyword-based detection
│   ├── ml_bias_detector.py      # MLBiasDetector: transformer + zero-shot detection
│   └── llm_bias_detector.py     # LLMBiasDetector: local LLM review via Ollama
├── eval/
│   ├── evaluate.py              # Precision / recall / F1 of a detector on data/gold
│   ├── fit_weights.py           # Learn ML signal weights on the dev split
│   └── agreement.py             # Inter-annotator agreement (Cohen's kappa)
├── tests/                       # pytest regression and contrast tests
├── webapp/
│   └── app.py                   # Streamlit web app
├── images/                      # IUEM and ULL logos used in the web app header/footer
├── sample_code/
│   ├── keyword_analysis.py      # CLI: keyword-based bias detection on a PDF
│   └── ml_analysis.py           # CLI: ML-based bias detection on a PDF
├── data/                        # Input PDFs (git-ignored)
│   └── gold/                    # Labelled sentences for evaluation (tracked; see its README)
├── ROADMAP.md                   # Improvement plan and results log
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
python -m spacy download en_core_web_md
python -m spacy download es_core_news_md
```

Both spaCy models are needed: the language is detected per page, so either may be loaded. The `md` models give better lemmas and grammatical gender (used for generic-masculine detection); the `sm` models still work as a fallback.

### 4. Optional: OCR for scanned PDFs

```bash
sudo apt install ocrmypdf tesseract-ocr-spa tesseract-ocr-eng
```

When some pages have no text layer and `ocrmypdf` is installed, those pages are OCRed (in Spanish and English) and analysed. The summary and the report list them, since recognition errors are possible. Without it, such pages are reported as pages without text.

### 5. Optional: local LLM review

Install [Ollama](https://ollama.com) and pull a model: `ollama pull llama3.1:8b` (set `OLLAMA_MODEL` to use another one). The **LLM review** button is disabled while Ollama is not running.

## Usage

All commands are run from the repository root.

### Web app

```bash
streamlit run webapp/app.py
```

Streamlit opens the app in your browser (by default at http://localhost:8501). To analyse a document:

1. Upload a PDF.
2. Click **🗝️ Keyword based analysis**, **🧠 ML Model based analysis** or **💬 LLM review**.
3. Review the results:
   - **Summary metrics**: detected language, total pages, biased pages, neutral pages and overall bias rate.
   - **Page breakdown**: one expandable entry per page with its status (biased / neutral), severity, gender bias direction, matched male/female keywords, discriminatory terms by category (keyword mode) or the model's flag and primary bias category (ML mode), plus a text snippet.

Notes:

- **Language is detected per page** with `langdetect`, so a Spanish proposal with an English abstract is analysed correctly. Short pages inherit the document language. Any language other than Spanish falls back to English.
- **Findings are highlighted** in the page text; hover to see the explanation and suggestion. Use 👍 / 👎 to save a finding as a labelled example in `data/feedback/feedback.jsonl` (git-ignored; review it and move confirmed records into `data/gold/`).
- **The bibliography is skipped**, and running headers/footers are removed.
- Results can be downloaded as JSON.
- **The ML model runs on CPU** and is cached in memory per language, so only the first ML analysis for each language is slow. The first run also downloads the transformer models from Hugging Face (several GB, mostly the zero-shot model).
- Uploading a different file clears the previous results.
- The web app does not write files to `output/` (only the feedback file above).

### Keyword-based bias detection on a PDF (CLI)

```bash
python sample_code/keyword_analysis.py path-to-file.pdf
python sample_code/keyword_analysis.py path-to-file.pdf --language spanish   # Spanish text
```

Analyses each page for gender bias and discriminatory language using keyword matching and regex patterns. Results are saved to `output/<pdf_name> - keyword_bias_results.json`, and a recommendation report grouping every finding by type (with pages and suggested rewrites) to `output/<pdf_name> - keyword_bias_report.md`.

### ML-based bias detection on a PDF (CLI)

```bash
python sample_code/ml_analysis.py path-to-file.pdf
python sample_code/ml_analysis.py path-to-file.pdf --language spanish   # Spanish text
```

Combines a fine-tuned bias classifier, zero-shot categorisation, and keyword analysis. Results are saved to `output/<pdf_name> - ml_bias_results.json`, and the recommendation report to `output/<pdf_name> - ml_bias_report.md`.

The CLI scripts detect the language per page by default (`--language auto`); pass `--language spanish` or `english` to force one.

### Tests and evaluation

```bash
python -m pytest tests/ -q                                    # regression and contrast tests
python eval/evaluate.py --detector keyword --split test       # P / R / F1 on held-out sentences
python eval/evaluate.py --detector ml --language spanish --show-errors
```

See [`data/gold/README.md`](data/gold/README.md) for the labelled-data format and annotation guideline, and the results log in [`ROADMAP.md`](ROADMAP.md).

## How It Works

Each PDF page is analysed separately, sentence by sentence:

**PDF → per-page text (headers, footers, bibliography removed) → page language → sentences → findings → page score → severity**

- **Keyword rules** (lemma-based, so inflected forms match) find explicit discriminatory requirements, gender stereotypes (a trait or generalisation next to a gendered reference), outdated or pejorative terms, negative generalisations about groups, and non-inclusive language (Spanish generic masculine, androcentric forms such as *el hombre* or *mankind*, gendered job titles, generic *he*). A neutral mention of a group (e.g. *población inmigrante*) is not scored.
- Findings are combined per page with noisy-OR; non-inclusive language is capped so it cannot dominate. Phrases that promote equality or study bias discount stereotype evidence in the same sentence only.
- Each signal produces a score between 0 and 1. The signals are combined in a weighted sum (capped at 1.0), and a page is flagged as biased when the score reaches the threshold (default `0.25`).
  - Keyword detector weights: gender 0.6, discriminatory language 0.4.
  - ML detector weights: learned per language by `eval/fit_weights.py` (`config/ml_weights.json`), default fine-tuned 0.30, zero-shot 0.30, gender keywords 0.20, discriminatory keywords 0.20.
- Zero-shot uses one hypothesis per category written in the document's language, scored with BART (`facebook/bart-large-mnli`) for English and XLM-R (`joeddav/xlm-roberta-large-xnli`) for Spanish. The fine-tuned classifier is English-only; for Spanish its weight goes to the other signals.

## Key Modules

| Module                         | Class              | Purpose                                                                    |
| ------------------------------ | ------------------ | -------------------------------------------------------------------------- |
| `webapp/app.py`                | —                  | Streamlit UI: upload, run analysis, display summary and per-page results   |
| `utils/bias_analysis.py`       | —                  | Per-page language detection and detection runners for app and CLI          |
| `utils/pdf_extraction.py`      | —                  | PDF text extraction and clean-up                                           |
| `analysis/llm_bias_detector.py`| `LLMBiasDetector`  | Local LLM review (Ollama) with rewrite suggestions                         |
| `analysis/bias_detector.py`    | `BiasDetector`     | Keyword/pattern gender bias & discriminatory language detection            |
| `analysis/ml_bias_detector.py` | `MLBiasDetector`   | Transformer + zero-shot + keyword combined detection                       |
| `analysis/bias_keywords.py`    | —                  | Shared keyword dictionaries, patterns, severity levels (EN + ES)           |
| `utils/preprocessing.py`       | `TextPreprocessor` | Text cleaning, tokenisation, lemmatisation, NER, POS tagging               |
| `config/settings.py`           | `PipelineConfig`   | Centralised configuration with sensible defaults                           |

## Troubleshooting

| Problem                                             | Solution                                                                                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| "No extractable text was found in the provided PDF" | The PDF is likely scanned/image-only. Install `ocrmypdf` (see setup step 4); if it is installed, OCR found no text either. |
| First ML analysis takes a long time                 | The transformer models are downloading and loading. Later runs for the same language reuse the cached model. |
| Wrong language detected                             | Detection runs per page; pages under 200 characters use the language of the first 3 pages. For the CLI, set `--language` explicitly. |
| `OSError: No spaCy model installed for 'spanish'`   | Run `python -m spacy download es_core_news_md` (or `en_core_web_md` for English).                            |
| LLM review button is disabled                       | Start Ollama (`ollama serve`) and pull the model (`ollama pull llama3.1:8b`).                                |
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
