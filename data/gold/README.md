# Gold set: labelled sentences for evaluation

One JSON object per line (`*.jsonl`). `eval/evaluate.py` loads every file in this folder.

| Field | Meaning |
| --- | --- |
| `id` | Unique id, e.g. `es-001` |
| `language` | `english` or `spanish` |
| `split` | `dev` (allowed for tuning rules and weights) or `test` (never tune on it) |
| `text` | One sentence as it appears in a document |
| `label` | `1` = biased or non-inclusive, `0` = acceptable |
| `category` | `gender_stereotype`, `explicit_discrimination`, `generic_masculine`, `androcentric`, `gendered_term`, `other_discrimination`, or `none` |
| `hard_negative` | `true` for acceptable sentences that mention gender or social groups (the false-positive trap) |
| `source` | Who labelled it: `seed-claude`, `holdout-claude`, `expert-<initials>`, `feedback-webapp` |

## Current files

- `seed_*.jsonl` (dev): 95 sentences written while building the Phase 1 rules. Scores on them are optimistic.
- `holdout_*.jsonl` (test): 60 sentences written afterwards and not used for tuning.

Both were written by Claude, not by domain experts. They are for development only. Real accuracy needs P0.2.

## Annotation guideline (for P0.2, expert annotation)

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
