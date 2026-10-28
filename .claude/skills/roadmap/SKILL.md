---
name: roadmap
description: Start, resume or check the bias-detection improvement roadmap (ROADMAP.md). Use for "/roadmap status", "/roadmap next", "/roadmap P2.2", "/roadmap phase 3".
argument-hint: "status | next | <task-id, e.g. P2.2> | phase <0-4>"
---

# Roadmap runner

Arguments: `$ARGUMENTS`

`ROADMAP.md` (repo root) is the source of truth for what is done. The shared document linked at its
top explains the reasoning and the "How to do" steps for every pending task. Read both before acting:
`ROADMAP.md` with Read, the document with the Claude Docs connector if it is available (otherwise
rely on ROADMAP.md and CLAUDE.md).

## Modes

- **status** (or no argument): print each phase with done / code-ready / pending counts, list pending
  tasks with their `needs:`, and the latest Results log row. Do not change anything.
- **next**: pick the first `[ ]` or `[~]` task whose `needs:` are satisfied (check the repo, installed
  tools and data, e.g. `which ocrmypdf`, files in `data/gold/` with `source` starting `expert-`).
  Say which task you picked and why the earlier ones are blocked, then implement it as below.
- **<task-id>** (e.g. `P2.2`): implement that task. If its `needs:` are not met, explain what is
  missing and ask the user how to proceed (e.g. provide data, install a tool, choose a model).
- **phase <n>**: implement every pending task in that phase in order, skipping blocked ones and
  reporting them at the end.

## Implementing a task

1. Follow the "How to do" steps for the task in the shared document and the conventions in CLAUDE.md
   (detector interface symmetry; keyword data only in `analysis/bias_keywords.py`; `language` is
   `"english"` / `"spanish"`).
2. Measure before and after on the **test** split:
   `python eval/evaluate.py --detector <keyword|ml|llm> --split test --tag <task-id>`.
   Never tune rules, word lists or weights on `split: test` items. If a task requires looking at test
   errors (e.g. new word lists), create a new test split first and say so.
3. Add or update tests in `tests/` and run `python -m pytest tests/ -q`. Everything must pass.
4. Update `ROADMAP.md`: tick the task (`[x]`, or `[~]` if it needs data to be useful), and add a row to
   the Results log when scores changed.
5. Update the shared document: set the task's Status dropdown and replace its "How to do" text with a
   short "How it was implemented" note. If the connector is unavailable, tell the user which rows to
   update by hand.
6. Report: what changed, the before/after scores, anything skipped or still blocked.

Do not commit unless the user asks.
