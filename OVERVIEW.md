# Project Overview

## Purpose

This proof of concept was created to explore automated bias detection in textual content — specifically research projects. The goal is to identify gender bias and discriminatory language so that reviewers and organisations can take corrective action before publication or hiring decisions.

The key research question: **can we reliably detect subtle bias in text, and how do lightweight keyword methods compare to ML-based approaches?**

## Development Process

The project was built incrementally, with each phase adding a new capability on top of the previous one.

### Phase 1 — Text Preprocessing

The foundation: a `TextPreprocessor` class wrapping NLTK and spaCy to handle cleaning (URL/email/HTML removal), tokenisation, lemmatisation, stopword removal, POS tagging, and named entity recognition. This layer feeds clean text to every downstream component.

### Phase 2 — Keyword-Based Bias Detection

The first detection approach (`BiasDetector`) uses curated keyword dictionaries and regex patterns to find:
- **Gender bias** — imbalances in male/female keyword counts, stereotypical role associations (e.g. leader→male, nurse→female), explicit discriminatory patterns.
- **Discriminatory language** — categorised by type: age, race, disability, appearance.

Dictionaries were built for both **English and Spanish**, stored in a shared `bias_keywords.py` module. A positive-context mechanism was added to discount false positives when phrases like "gender equality" or "diversity and inclusion" appear nearby.

### Phase 3 — ML-Based Bias Detection

The second approach (`MLBiasDetector`) combines three signal sources:
1. **Fine-tuned classifier** — `valurank/distilroberta-bias`, a DistilRoBERTa model fine-tuned for binary bias detection (English only).
2. **Zero-shot classification** — BART (`facebook/bart-large-mnli`) for English, XLM-RoBERTa (`joeddav/xlm-roberta-large-xnli`) for Spanish, categorising text against bias labels.
3. **Keyword/pattern analysis** — the same shared dictionaries from Phase 2, ensuring consistency between approaches.

The final bias score is a weighted combination of all three signals.

### Phase 4 — Sample Scripts & Reporting

Two CLI scripts were created to demonstrate real-world usage on PDF documents:
- `keyword_analysis.py` — runs keyword detection per page
- `ml_analysis.py` — runs ML detection per page

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Dual approach (keyword + ML)** | Keywords are fast, transparent, and need no GPU; ML catches subtler patterns. Comparing both reveals detection coverage and agreement. |
| **distilroberta-bias as fine-tuned model** | Purpose-built for bias detection, lightweight enough for PoC use, good accuracy on English text. |
| **BART + XLM-R for zero-shot** | BART performs well on English zero-shot; XLM-R extends coverage to Spanish without fine-tuning. |
| **Shared bias_keywords module** | Single source of truth for keyword dictionaries ensures consistency between BiasDetector and MLBiasDetector. |
| **Positive-context discounting** | Reduces false positives when bias-related words appear in anti-bias contexts (e.g. "promoting gender equality"). |
| **PDF per-page analysis** | Many real documents are PDFs; per-page granularity lets reviewers pinpoint where bias occurs. |

## Obstacles Encountered

### Multilingual model availability

The fine-tuned bias classifier (`distilroberta-bias`) only works for English. For Spanish, we had to rely on zero-shot classification with XLM-RoBERTa, which is less precise for domain-specific bias categories. Building separate keyword dictionaries for each language was also labour-intensive and would need native-speaker validation.

### False positive handling

Early versions flagged any text mentioning gender-related terms as biased — including passages actively promoting equality. The positive-context phrase mechanism was introduced to address this, discounting scores by up to 60% when anti-bias language is present. Tuning these discount thresholds required iterative testing on real documents.

### Keyword vs ML disagreement

The two detection approaches don't always agree. Keywords catch explicit patterns (e.g., "only men should apply") that ML sometimes under-weights, while ML catches implicit bias that keywords miss entirely.

### GPU and memory constraints

Transformer models (especially BART-large) require significant memory. The pipeline was designed to fall back to CPU gracefully, and the lightweight `BiasDetector` was maintained as a no-GPU alternative. Batch size configuration allows tuning for available hardware.

### PDF text extraction quality

PDF text extraction (via PyPDF2) produces inconsistent results depending on document formatting — scanned PDFs, multi-column layouts, and embedded tables all degrade extraction quality. Per-page analysis helps contain these issues, but preprocessing still needs to handle noisy input gracefully.

### Severity threshold calibration

Mapping continuous bias scores to discrete severity levels (none / low / medium / high / critical) required experimentation. Thresholds that worked well for English texts didn't generalise to Spanish texts, leading to language-aware severity mapping in the shared keywords module.

## Current State

The PoC is functional and demonstrates both detection approaches on real PDF documents. The sample scripts produce per-page analysis with JSON exports.

## Future Directions

- Additional language support beyond English and Spanish.
- Fine-tuning a bias classifier on Spanish data.
- Interactive web dashboard for non-technical reviewers.
- Benchmark datasets for systematic accuracy evaluation.
- Domain-specific keyword dictionaries (e.g., medical, legal).
