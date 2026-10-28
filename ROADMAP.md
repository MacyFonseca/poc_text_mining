# Detection improvement roadmap

Task tracker for improving bias detection (focus: Spanish). It is the source of truth for **what is done**.
The shared document explains **why** and **how** for each task:
https://claude.ai/code/artifact/85396407-4b04-49db-a147-bb2a65fc5015

Run a task from Claude Code with `/roadmap <task-id>`, a phase with `/roadmap phase <n>`,
the next available task with `/roadmap next`, and the overview with `/roadmap status`.

Status: `[x]` done · `[~]` code ready, needs data or a decision · `[ ]` pending.
`needs:` lists prerequisites (tasks or external resources).

Abbreviations used below:

| Abbreviation | Meaning |
| --- | --- |
| EN / ES | English / Spanish |
| P / R / F1 | Precision / recall / F1 score (harmonic mean of precision and recall) |
| PR-AUC | Area under the precision-recall curve |
| CI | Confidence interval (95%, bootstrap) |
| FP / FN | False positive / false negative |
| ML | Machine learning (the transformer-based detector) |
| LLM | Large language model (the Ollama-based reviewer) |
| NLI | Natural language inference (used by the zero-shot classifier) |
| XLM-R | XLM-RoBERTa, a multilingual transformer model |
| OCR | Optical character recognition |
| ONNX | Open Neural Network Exchange (model format) |
| int8 | 8-bit integer quantisation |
| CSV / JSON / DOCX / PDF | Comma-separated values / JavaScript Object Notation / Word document / Portable Document Format |
| CLI | Command-line interface |
| IUEM | Instituto Universitario de Estudios de las Mujeres (ULL) |
| ULL | Universidad de La Laguna |
| APA | American Psychological Association |
| EXIST | sEXism Identification in Social neTworks (shared task) |
| DETESTS | DETEction and classification of racial STereotypes in Spanish (shared task) |
| BABE | Bias Annotations By Experts (dataset) |

## Phase 0: Measure before changing anything

- [x] **P0.1** Gold-set format, annotation guideline, 95 dev + 60 test sentences. `data/gold/`
- [ ] **P0.2** Expert annotation: 600+ sentences per language from real proposals, two annotators, kappa >= 0.7. needs: IUEM annotators, 30-50 proposals
- [x] **P0.3** Regression, contrast, hard-negative and gender-swap tests. `tests/test_keyword_detector.py`
- [x] **P0.4** Evaluation script with dev/test splits and saved runs. `eval/evaluate.py`
- [x] **P0.5** Inter-annotator agreement (Cohen's kappa). `eval/agreement.py`
- [ ] **P0.6** Public datasets as secondary benchmarks ([EXIST](https://nlp.uned.es/exist2024/), [DETESTS](https://detestsiberlef.wixsite.com/detests/corpus), [DETESTS-Dis](https://huggingface.co/datasets/CLiC-UB/DETESTS-Dis), [MeTwo](https://github.com/franciscorodriguez92/MeTwo), [BABE](https://github.com/Media-Bias-Group/Neural-Media-Bias-Detection-Using-Distant-Supervision-With-BABE)): write `eval/import_<dataset>.py`. needs: dataset access (registration)
- [x] **P0.7** Side-by-side comparison of the three detectors: bootstrap 95% confidence intervals, recall per category with counts, pairwise McNemar, per-sentence error overlap. `eval/compare.py`
- [~] **P0.8** Realistic `real` split from PDFs: pool sentences any detector flags + a random sample of the rest, label them in a CSV, import with sampling weights. `eval/pool_sentences.py`. needs: real research proposals in `data/proposals/` (see P0.2), then labels

## Phase 1: Keyword detector

- [x] **P1.1** Sentence-level analysis with character offsets. `analysis/bias_keywords.py`
- [x] **P1.2** Lemma matching; spaCy `md` models cached per language. `utils/nlp.py`
- [x] **P1.3** Spanish generic-masculine detector (37 nouns + 10 epicene) with alternatives.
- [x] **P1.4** Androcentric forms, gendered job titles, generic "he", "señorita".
- [x] **P1.5** Graded scoring: neutral group mentions score 0, noisy-OR, non-inclusive cap 0.6.
- [x] **P1.6** Positive-context discount per sentence; never discounts explicit patterns.
- [x] **P1.7** Domain-aware patterns (age 18 ignored; hiring context for "male engineer").
- [ ] **P1.8** Expand word lists from published guides (Instituto de las Mujeres, ULL Unidad de Igualdad, Gaucher et al. 2011, APA), add a general `-man/-men` rule. Then create a NEW test split. needs: P0.2 recommended
- [x] **P1.9** Inclusive-language suggestion on every finding.

## Phase 2: Models

- [x] **P2.1** Zero-shot fix: hypotheses in the document language, each category scored independently with 3-way entailment, threshold 0.5. Model comparison done: XLM-R-large kept (mDeBERTa-v3-base gave near-zero entailment). `analysis/ml_bias_detector.py`
- [ ] **P2.2** Dedicated Spanish classifier: SetFit few-shot (`paraphrase-multilingual-mpnet-base-v2`) or fine-tune `BSC-LT/roberta-base-bne` on EXIST-ES + DETESTS + gold dev. Plug in as the `fine_tuned` signal for Spanish. needs: P0.2 (or P0.6)
- [x] **P2.3** Sentence-level fine-tuned scoring (page = highest sentence), weight redistribution when a signal is missing.
- [x] **P2.4** Local LLM reviewer via Ollama, same interface, quotes verified against the text. `analysis/llm_bias_detector.py`
- [x] **P2.4a** Evaluate the LLM reviewer: `python eval/evaluate.py -d llm --split test --tag llama3.1-8b`. Done 2026-10-09: F1 0.80 EN / 0.88 ES at 18-24 s per sentence. Misses English gendered terms and 2 of 5 Spanish generic masculines; flags 2 of 7 Spanish hard negatives.
- [ ] **P2.4b** Better LLM: try a stronger local model (e.g. Qwen 2.5/3 14B, Gemma, BSC Salamandra) or an approved hosted API; compare with `eval/evaluate.py -d llm`. needs: decision on model and data policy
- [~] **P2.5** Learned signal weights (logistic regression on dev, converted to weights + threshold). `eval/fit_weights.py`, `config/ml_weights.json`. Currently fitted on Claude-written dev data: refit after P0.2.
- [ ] **P2.6** Replace or complement the English fine-tuned model (`himel7/bias-detector` is trained on news media bias, not gender bias). Candidate: classifier fine-tuned on EXIST-EN. needs: P0.6

## Phase 3: PDF extraction and pipeline

- [x] **P3.1** `pypdf` instead of PyPDF2; real page numbers kept. `utils/pdf_extraction.py`
- [x] **P3.2** Running header/footer removal.
- [x] **P3.3** Hyphenated line-break re-joining.
- [x] **P3.4** Bibliography excluded from analysis.
- [x] **P3.5** OCR for scanned pages: when pages have no text and `ocrmypdf` is installed, the PDF is OCRed (`--skip-text -l spa+eng`) and extracted again; OCRed pages are reported as `ocr_pages`. `utils/pdf_extraction.py`, `tests/test_pdf_extraction.py`
- [x] **P3.6** Per-page language detection (short pages inherit the document language). `utils/bias_analysis.py`
- [x] **P3.7** Token-based zero-shot windows (200 tokens) and batched inference.
- [ ] **P3.8** CPU speed-up: ONNX export + int8 quantisation via `optimum[onnxruntime]`; check F1 drops < 0.01.

## Phase 4: Web app and recommendations

- [x] **P4.1** Highlighted findings with explanation/suggestion tooltips. `webapp/app.py`
- [x] **P4.2** Suggestions list per page.
- [x] **P4.3** Reviewer feedback buttons -> `data/feedback/feedback.jsonl` (gold format, git-ignored; move reviewed records to `data/gold/`).
- [x] **P4.4** LLM review button (disabled when Ollama is not running).
- [x] **P4.5** JSON download, mixed-language / bibliography / empty-page notes, neutral-page toggle.
- [x] **P4.6** Document-level recommendation report: findings grouped by type and wording, one explanation + rewrite + pages + example each; Markdown download and preview in the web app, written by both CLIs. `utils/report.py`. Not done: DOCX export, LLM rewrites of whole flagged sentences (optional).

## Results log

Held-out **test** split (60 Claude-written sentences, never used for tuning). F1 / precision / recall.
Dev-split scores are not listed: the rules were written against them.

| Date       | Commit    | Detector | English F1 (P / R)  | Spanish F1 (P / R)  | Notes                                                                           |
| ---------- | --------- | -------- | ------------------- | ------------------- | ------------------------------------------------------------------------------- |
| 2026-10-08 | d3f180a   | keyword  | 0.125 (1.00 / 0.07) | 0.118 (1.00 / 0.06) | Original proof of concept                                                       |
| 2026-10-08 | d3f180a   | ml       | 0.538 (0.64 / 0.47) | 0.111 (0.50 / 0.06) | Original; 44% / 14% hard-negative false positives                               |
| 2026-10-08 | phase 1-4 | keyword  | 0.500 (1.00 / 0.33) | 0.857 (1.00 / 0.75) | 0 hard-negative false positives                                                 |
| 2026-10-08 | phase 1-4 | ml       | 0.636 (1.00 / 0.47) | 0.897 (1.00 / 0.81) | Learned weights; PR-AUC 0.82 / 1.00                                             |
| 2026-10-09 | phase 1-4 | llm      | 0.800 (1.00 / 0.67) | 0.875 (0.88 / 0.88) | llama3.1:8b; PR-AUC 0.88 / 0.90; 29% Spanish hard-negative FP; 18-24 s/sentence |

2026-10-28 comparison (`eval/compare.py --split test`, commit 374d418) reproduces the rows above and adds 95% confidence intervals (CI).
F1 pooled over both languages: keyword 0.71 [0.55-0.83], ml 0.78 [0.65-0.90], llm 0.84 [0.72-0.93].
No pairwise difference is significant (McNemar test, p >= 0.06): the 60-item test split cannot rank the detectors.
