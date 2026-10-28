# Gold set: labelled sentences for evaluation

One JSON (JavaScript Object Notation) object per line (`*.jsonl`, JSON Lines). `eval/evaluate.py` loads every file in this folder.

| Field | Meaning |
| --- | --- |
| `id` | Unique id, e.g. `es-001` |
| `language` | `english` or `spanish` |
| `split` | `dev` (allowed for tuning rules and weights), `test` (never tune on it) or `real` (sentences from real research proposals, never tune on it) |
| `text` | One sentence as it appears in a document |
| `label` | `1` = biased or non-inclusive, `0` = acceptable |
| `category` | `gender_stereotype`, `explicit_discrimination`, `generic_masculine`, `androcentric`, `gendered_term`, `other_discrimination`, or `none` |
| `hard_negative` | `true` for acceptable sentences that mention gender or social groups (the false-positive trap) |
| `source` | Who labelled it: `seed-claude`, `holdout-claude`, `expert-<initials>`, `feedback-webapp` |
| `weight` | Optional, `real` split only: how many document sentences this item stands for (see below) |
| `document`, `page` | Optional, `real` split only: where the sentence comes from |

## Current files

- `seed_*.jsonl` (dev): 95 sentences written while building the Phase 1 rules. Scores on them are optimistic.
- `holdout_*.jsonl` (test): 60 sentences written afterwards and not used for tuning.

Both were written by Claude, not by domain experts. They are for development only. Real accuracy needs roadmap task P0.2 (expert annotation).

- `real_*.jsonl` (real, once labelled): sentences from real research proposals, built with `eval/pool_sentences.py`. None yet.

## Real split: pooling sentences from research proposals

Use real research proposals in PDF (Portable Document Format), the kind of document the tool is meant to
review. Put them in `data/proposals/` (git-ignored); demo or unrelated documents give misleading numbers.

```
python eval/pool_sentences.py pool data/proposals/*.pdf  # -> data/annotation/pool_<language>.csv
# label the CSV (comma-separated values) file: label 1/0, category, hard_negative true/false, optional note
python eval/pool_sentences.py import data/annotation/pool_spanish.csv --annotator ab
python eval/compare.py --split real
```

The pool keeps every sentence that any detector flags, plus a random sample of the unflagged ones (50 per
language by default). The CSV does not show which detector flagged what, so the flags do not influence the
labels. A sampled unflagged sentence gets `weight = unflagged / sampled`, so weighted precision and recall
estimate the rates over the whole documents, including bias that all detectors miss. `data/annotation/`
is git-ignored; `real_*.jsonl` contains sentences from the source documents, so check they may be shared
before committing them.

## Annotation guideline (for roadmap task P0.2, expert annotation)

Label each sentence on its own, as it would read in a research proposal.

1. **Biased (1)** if the sentence
   - sets a requirement or prescription based on gender, age, appearance, origin or disability;
   - attributes a trait or ability to a gender or group (stereotype), even mildly;
   - uses an outdated or pejorative term for a group;
   - uses generic masculine, androcentric forms ("el hombre", "mankind") or gendered job titles when a neutral form exists.
2. **Acceptable (0)** if the sentence
   - reports data about a group ("las mujeres representan el 38 %");
   - discusses, studies or criticises bias ("el estudio analiza los estereotipos...");
   - refers to a specific, named person with their own gender ("la doctora López").
3. When unsure, label your best guess and add `"note": "..."`.

Two annotators label the same 100-sentence sample first. Then compute agreement:

```
python eval/agreement.py data/gold/annotator_a.jsonl data/gold/annotator_b.jsonl
```

Aim for Cohen's kappa of at least 0.7 before annotating the rest. Below that, refine this guideline and repeat.
