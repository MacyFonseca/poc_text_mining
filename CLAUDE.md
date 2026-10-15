# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Proof-of-concept pipeline that detects **gender bias** and **discriminatory language** in PDF research project documents, in **English and Spanish**. It offers two parallel detection approaches — lightweight keyword-based and heavier ML-based — plus a Streamlit web UI.

## Commands

A virtual environment already exists at `.venv/`.

```bash
source .venv/bin/activate

# Install deps (if setting up fresh)
pip install -r requirements.txt
# First-run NLP data (NLTK corpora + spaCy models — both languages)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_md   # md preferred (better lemmas/morphology); sm is the fallback
python -m spacy download es_core_news_md

# Web app (CPU-only; caches the ML model per language in RAM)
streamlit run webapp/app.py

# CLI, keyword-based (fast, no models); language defaults to auto (per page)
python sample_code/keyword_analysis.py path/to/doc.pdf [--language spanish]

# CLI, ML-based (downloads transformer models on first run)
python sample_code/ml_analysis.py path/to/doc.pdf [--language spanish]

# Tests and evaluation
python -m pytest tests/ -q
python eval/evaluate.py --detector keyword|ml|llm --split test [--show-errors] [--tag name]
python eval/fit_weights.py --language spanish [--save]   # learn ML weights on the dev split
```

CLI scripts write the full result dict to `output/<pdf_name> - {keyword,ml}_bias_results.json` and the grouped recommendation report (`utils/report.py`) to `output/<pdf_name> - {keyword,ml}_bias_report.md`.

No linter or build step. Sample PDFs live in `data/` (git-ignored); labelled sentences in `data/gold/` (tracked).

## Roadmap

`ROADMAP.md` tracks the improvement plan (task IDs P0.1-P4.6, done/pending, results log). Use `/roadmap status|next|<id>|phase <n>` (`.claude/skills/roadmap/`). When finishing a roadmap task, tick it in ROADMAP.md and add a results-log row if scores changed. Never tune rules or weights on `split: test` gold items.

## Architecture

The pipeline is **PDF → per-page text → preprocess → detect → score → severity**. Each PDF page is treated as an independent document; empty pages are skipped.

### Two detectors with an identical interface

`BiasDetector` (`analysis/bias_detector.py`, keyword-only), `MLBiasDetector` (`analysis/ml_bias_detector.py`, transformers) and `LLMBiasDetector` (`analysis/llm_bias_detector.py`, local Ollama LLM) expose the same contract:
- `comprehensive_bias_analysis(text) -> dict` with keys `overall_bias_score`, `is_biased`, `severity`, `gender_bias`, `discriminatory_language`, `findings` (ML adds `ml_detection`, `ml_categorization`, `signal_scores`; LLM adds `llm_review`).
- Each finding: `type, category, subtype, match, start, end, sentence, score, explanation, suggestion` (offsets index into the analysed text).
- `batch_analysis`, `generate_bias_report`.

This interface symmetry is deliberate — callers can swap detectors freely. Preserve it when editing either class.

**Scoring model** (all detectors): each signal contributes a score in `[0,1]`, combined via a weighted sum capped at 1.0, then `is_biased = overall_bias_score >= overall_bias_threshold` (default 0.25), and `severity` via `score_to_severity`. This is threshold-based, **not** OR-gating — a change made specifically to cut false positives on text that merely *discusses* gender. Keyword detector weights: `(gender 0.6, discriminatory 0.4)`. ML detector default weights: `(fine_tuned 0.30, zero_shot 0.30, gender_kw 0.20, discrim_kw 0.20)`; a missing signal's weight (fine-tuned for Spanish) is redistributed; if `config/ml_weights.json` exists (from `eval/fit_weights.py`) its learned weights and threshold replace the defaults. LLM detector weights: `(llm 0.6, gender_kw 0.2, discrim_kw 0.2)`.

Inside the keyword module, per-sentence findings are combined per page with noisy-OR (`1 - prod(1 - s)`); non-inclusive language (generic masculine, androcentric, gendered terms) is capped at 0.6 per page.

### Shared keyword module is the single source of truth

`analysis/bias_keywords.py` holds all EN/ES dictionaries, regex patterns, positive-context phrases, scores and `score_to_severity`. Every detector delegates keyword/pattern logic here — never duplicate keyword data into a detector. Analysis runs **per sentence** (spaCy, lemma-based, cached via `analyse_text`). It deliberately separates:
- **Gendered references** (neutral alone; used for direction and co-occurrence).
- **Stereotyped traits / markers / generalisations** (biased only next to a gendered reference in the same sentence).
- **Explicit discriminatory patterns** (regex for requirements; never discounted).
- **Discriminatory terms** (outdated/pejorative) and **negative generalisations** about groups; a *neutral* group mention is recorded under `mentions` and scores 0.
- **Non-inclusive language**: Spanish generic masculine (`GENERIC_MASCULINE_ES`, `EPICENE_NOUNS_ES`), androcentric forms, gendered job titles, generic "he". Each carries an inclusive suggestion.
- **Positive-context phrases** that discount stereotype evidence in the same sentence only.

### ML detector signal sources

1. Fine-tuned binary classifier — **`himel7/bias-detector`** (`_FINETUNED_MODEL`; trained on news media bias). English only. Scored per sentence; page confidence = highest sentence.
2. Zero-shot NLI — BART (`facebook/bart-large-mnli`) for English, XLM-R (`joeddav/xlm-roberta-large-xnli`) for Spanish. One hypothesis per category **in the document's language** (`_HYPOTHESES`), scored independently with the full 3-way entailment probability over windows of <=200 tokens; category named only when >= 0.5. (The pipeline's `multi_label` mode was dropped: it ignores the NLI neutral class and scores unrelated text high. mDeBERTa-v3-base-xnli was tested and gave near-zero entailment.)
3. Keyword/pattern analysis from the shared module.

For `language='spanish'` the fine-tuned classifier is skipped (returns a stub with `available: False`).

### Language handling

`config/settings.py` maps language → preferred spaCy models (md, then sm); `utils/nlp.get_nlp` caches one pipeline per language. Language defaults to `auto` everywhere: `utils/bias_analysis.detect_page_languages` detects it **per page** with `langdetect` (seeded), pages under 200 chars inherit the document language (first 3 pages), and everything non-Spanish collapses to english. Each page is analysed by the detector for its own language.

### Web layer

`webapp/app.py` (Streamlit) is the active development surface (branch `webapp`). It calls `utils/bias_analysis.py` (`run_keyword_bias_detection`, `run_ml_bias_detection`, `run_llm_bias_detection`, all via `run_bias_detection`), which wraps extraction (`utils/pdf_extraction.py`: pypdf, header/footer removal, de-hyphenation, stops at the bibliography) + cleaning + detection and returns a `{summary, pages}` dict; pages carry `findings` for highlighting. `utils/report.build_report` groups a whole document's findings by subtype and wording (count, pages, suggestion, example) and `report_to_markdown` renders it for the "Download report" button and the CLIs. The ML model is cached per language via `@st.cache_resource` and pinned to CPU. Reviewer 👍/👎 feedback is appended to `data/feedback/feedback.jsonl` (git-ignored) in gold-set format.

### Preprocessing

`utils/preprocessing.py` (`TextPreprocessor`, NLTK + spaCy) handles cleaning, tokenisation, lemmatisation, stopword removal, POS, NER. Only `clean_text()` (URL/email/HTML removal) runs before detection in the current pipelines.

## Conventions

- Scripts and the web app insert the project root into `sys.path` so imports are absolute from the repo root (`from analysis...`, `from utils...`). Run from the repo root.
- `language` is always the full word `"english"` / `"spanish"`, not a code.
