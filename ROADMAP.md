# Detection improvement roadmap

Task tracker for improving bias detection (focus: Spanish). It is the source of truth for **what is done**.
The shared document explains **why** and **how** for each task:
https://claude.ai/code/artifact/85396407-4b04-49db-a147-bb2a65fc5015

Run a task from Claude Code with `/roadmap <task-id>`, a phase with `/roadmap phase <n>`,
the next available task with `/roadmap next`, and the overview with `/roadmap status`.

Status: `[x]` done · `[~]` code ready, needs data or a decision · `[ ]` pending.
`needs:` lists prerequisites (tasks or external resources).

## Phase 0: Measure before changing anything

- [x] **P0.1** Gold-set format, annotation guideline, 95 dev + 60 test sentences. `data/gold/`
- [ ] **P0.2** Expert annotation: 600+ sentences per language from real proposals, two annotators, kappa >= 0.7. needs: IUEM annotators, 30-50 proposals
- [x] **P0.3** Regression, contrast, hard-negative and gender-swap tests. `tests/test_keyword_detector.py`
- [x] **P0.4** Evaluation script with dev/test splits and saved runs. `eval/evaluate.py`
- [x] **P0.5** Inter-annotator agreement (Cohen's kappa). `eval/agreement.py`
- [ ] **P0.6** Public datasets as secondary benchmarks (EXIST, DETESTS, MeTwo, BABE): write `eval/import_<dataset>.py`. needs: dataset access (registration)

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
- [x] **P2.4a** Evaluate the LLM reviewer: `python eval/evaluate.py -d llm --split test --tag llama3.1-8b` (~1 min/sentence on CPU). The first run was stopped for low memory; run it with no ML model loaded.
- [ ] **P2.4b** Better LLM: try a stronger local model (e.g. Qwen 2.5/3 14B, Gemma, BSC Salamandra) or an approved hosted API; compare with `eval/evaluate.py -d llm`. needs: decision on model and data policy
- [~] **P2.5** Learned signal weights (logistic regression on dev, converted to weights + threshold). `eval/fit_weights.py`, `config/ml_weights.json`. Currently fitted on Claude-written dev data: refit after P0.2.
- [ ] **P2.6** Replace or complement the English fine-tuned model (`himel7/bias-detector` is trained on news media bias, not gender bias). Candidate: classifier fine-tuned on EXIST-EN. needs: P0.6

## Phase 3: PDF extraction and pipeline

- [x] **P3.1** `pypdf` instead of PyPDF2; real page numbers kept. `utils/pdf_extraction.py`
- [x] **P3.2** Running header/footer removal.
- [x] **P3.3** Hyphenated line-break re-joining.
- [x] **P3.4** Bibliography excluded from analysis.
- [x] **P3.5** OCR for scanned pages with `ocrmypdf --skip-text -l spa+eng`. needs: `sudo apt install ocrmypdf tesseract-ocr-spa tesseract-ocr-eng`
- [x] **P3.6** Per-page language detection (short pages inherit the document language). `utils/bias_analysis.py`
- [x] **P3.7** Token-based zero-shot windows (200 tokens) and batched inference.
- [ ] **P3.8** CPU speed-up: ONNX export + int8 quantisation via `optimum[onnxruntime]`; check F1 drops < 0.01.

## Phase 4: Web app and recommendations

- [x] **P4.1** Highlighted findings with explanation/suggestion tooltips. `webapp/app.py`
- [x] **P4.2** Suggestions list per page.
- [x] **P4.3** Reviewer feedback buttons -> `data/feedback/feedback.jsonl` (gold format, git-ignored; move reviewed records to `data/gold/`).
- [x] **P4.4** LLM review button (disabled when Ollama is not running).
- [x] **P4.5** JSON download, mixed-language / bibliography / empty-page notes, neutral-page toggle.
- [ ] **P4.6** Document-level recommendation report: group findings by type, one explanation + rewrite + page list each, downloadable as Markdown (later DOCX). needs: none

## Results log

Held-out **test** split (60 Claude-written sentences, never used for tuning). F1 / precision / recall.
Dev-split scores are not listed: the rules were written against them.

| Date       | Commit    | Detector | English F1 (P / R)  | Spanish F1 (P / R)  | Notes                                             |
| ---------- | --------- | -------- | ------------------- | ------------------- | ------------------------------------------------- |
| 2026-10-08 | d3f180a   | keyword  | 0.125 (1.00 / 0.07) | 0.118 (1.00 / 0.06) | Original proof of concept                         |
| 2026-10-08 | d3f180a   | ml       | 0.538 (0.64 / 0.47) | 0.111 (0.50 / 0.06) | Original; 44% / 14% hard-negative false positives |
| 2026-10-08 | phase 1-4 | keyword  | 0.500 (1.00 / 0.33) | 0.857 (1.00 / 0.75) | 0 hard-negative false positives                   |
| 2026-10-08 | phase 1-4 | ml       | 0.636 (1.00 / 0.47) | 0.897 (1.00 / 0.81) | Learned weights; PR-AUC 0.82 / 1.00               |
| —          | —         | llm      | not measured        | not measured        | See P2.4a                                         |

Not yet checked end to end: `sample_code/ml_analysis.py` on a PDF (stopped for low memory; the keyword CLI and the web app were checked).
