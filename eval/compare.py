"""Compare the keyword, ML and LLM detectors on the same gold items.

Runs every detector over one split, then writes a Markdown report with:
metrics and bootstrap 95% confidence intervals, recall per category (with
counts), pairwise McNemar tests, and which sentences each detector misses.

Usage:
    python eval/compare.py --split test
    python eval/compare.py --split test --detectors keyword,ml
    python eval/compare.py --from-json eval/results/<stamp>-compare-test.json   # re-render only

Results go to ``eval/results/<stamp>-compare-<split>.{json,md}``.  The JSON
keeps every prediction, so the report can be rebuilt without re-running the
(slow) LLM.
"""
import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from math import comb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.evaluate import (GOLD_DIR, LANGUAGES, RESULTS_DIR, SPLITS, _git_commit,
                           build_detector, compute_metrics, load_gold,
                           metrics_from_rows, predict)

DETECTORS = ['keyword', 'ml', 'llm']
BOOTSTRAP_SAMPLES = 2000


# -- statistics -------------------------------------------------------------

def bootstrap_ci(rows: list[dict], samples: int = BOOTSTRAP_SAMPLES, seed: int = 0) -> dict:
    """95% percentile intervals for precision, recall and F1 (items resampled)."""
    rng = random.Random(seed)
    values = {'precision': [], 'recall': [], 'f1': []}
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        m = compute_metrics([r['label'] for r in sample], [r['pred'] for r in sample],
                            [r['score'] for r in sample], [r.get('weight', 1.0) for r in sample])
        for k in values:
            values[k].append(m[k])
    ci = {}
    for k, v in values.items():
        v.sort()
        ci[k] = [round(v[int(0.025 * samples)], 3), round(v[int(0.975 * samples) - 1], 3)]
    return ci


def mcnemar(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Exact McNemar test on per-item correctness of two detectors."""
    b = sum(1 for x, y in zip(rows_a, rows_b) if x['pred'] == x['label'] and y['pred'] != y['label'])
    c = sum(1 for x, y in zip(rows_a, rows_b) if x['pred'] != x['label'] and y['pred'] == y['label'])
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return {'only_a_right': b, 'only_b_right': c, 'p_value': round(p, 4)}


# -- running ----------------------------------------------------------------

def run(detectors: list[str], languages: list[str], split: str, gold_dir: str) -> dict:
    """Predictions of every detector on every item, one language at a time."""
    predictions = {}
    for language in languages:
        items = load_gold(language, gold_dir, split)
        if not items:
            continue
        predictions[language] = {}
        for name in detectors:
            print(f"  {language:<8} {name:<8} {len(items)} items ...", flush=True)
            detector = build_detector(name, language)
            if name == 'llm' and not detector.is_available():
                sys.exit("Ollama is not running or the model is missing: start it with `ollama serve`.")
            predictions[language][name] = predict(detector, items)
            del detector
            gc.collect()        # free the transformer models before loading the next ones
    return predictions


def groups(predictions: dict) -> dict:
    """Predictions per language, plus 'all' (languages pooled) when there are several."""
    out = dict(predictions)
    if len(predictions) > 1:
        out['all'] = {}
        for per_detector in predictions.values():
            for name, rows in per_detector.items():
                out['all'].setdefault(name, []).extend(rows)
    return out


def summarise(predictions: dict) -> dict:
    """Metrics, intervals and pairwise tests per group (language or 'all')."""
    summary = {}
    for group, per_detector in groups(predictions).items():
        names = list(per_detector)
        entry = {'metrics': {}, 'ci': {}, 'mcnemar': {}}
        for name in names:
            entry['metrics'][name] = metrics_from_rows(per_detector[name])
            entry['ci'][name] = bootstrap_ci(per_detector[name])
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                entry['mcnemar'][f'{a} vs {b}'] = mcnemar(per_detector[a], per_detector[b])
        summary[group] = entry
    return summary


# -- report -----------------------------------------------------------------

def _fmt(value, ci=None):
    if value is None:
        return '-'
    return f"{value:.2f} [{ci[0]:.2f}-{ci[1]:.2f}]" if ci else f"{value:.2f}"


def render_markdown(run_info: dict) -> str:
    predictions, summary = run_info['predictions'], run_info['summary']
    names = run_info['detectors']
    lines = [
        f"# Detector comparison: {run_info['split']} split",
        '',
        f"Commit `{run_info['commit']}`, {run_info['timestamp']}. "
        f"Values in brackets are bootstrap 95% confidence intervals ({BOOTSTRAP_SAMPLES} resamples).",
        '',
        "Abbreviations: ML = machine learning (transformer-based detector); "
        "LLM = large language model (Ollama-based reviewer); F1 = harmonic mean of precision and recall; "
        "PR-AUC = area under the precision-recall curve; FP = false positive; FN = false negative; "
        "s/item = seconds per sentence; n = number of sentences.",
        '',
    ]
    weighted = any(r.get('weight', 1.0) != 1.0
                   for per in predictions.values() for rows in per.values() for r in rows)
    if weighted:
        lines += ["Metrics are weighted to undo pool sampling (see `eval/pool_sentences.py`); "
                  "recall by category and the error table are unweighted.", '']

    for group, per_detector in groups(predictions).items():
        entry = summary[group]
        title = 'All languages (pooled)' if group == 'all' else group.capitalize()
        any_rows = per_detector[names[0]]
        n_pos = sum(r['label'] for r in any_rows)
        n_hard = sum(1 for r in any_rows if r['hard_negative'])
        lines += [f"## {title}", '',
                  f"{len(any_rows)} items: {n_pos} biased, {len(any_rows) - n_pos} acceptable "
                  f"({n_hard} hard negatives).", '',
                  '| Detector | Precision | Recall | F1 | PR-AUC | Hard-negative FP rate | s/item |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
        for name in names:
            m, ci = entry['metrics'][name], entry['ci'][name]
            lines.append(f"| {name} | {_fmt(m['precision'], ci['precision'])} | "
                         f"{_fmt(m['recall'], ci['recall'])} | {_fmt(m['f1'], ci['f1'])} | "
                         f"{_fmt(m['pr_auc'])} | {_fmt(m['hard_negative_fp_rate'])} | "
                         f"{m['seconds_per_item']:.2f} |")

        # Recall per category, with counts so tiny categories are visible
        cats = sorted({r['category'] for r in any_rows if r['label'] == 1})
        lines += ['', '| Category (recall) | n | ' + ' | '.join(names) + ' |',
                  '| --- | --- |' + ' --- |' * len(names)]
        for cat in cats:
            pos = [r for r in any_rows if r['label'] == 1 and r['category'] == cat]
            cells = []
            for name in names:
                hits = sum(r['pred'] for r in per_detector[name] if r['label'] == 1 and r['category'] == cat)
                cells.append(f"{hits}/{len(pos)}")
            lines.append(f"| {cat} | {len(pos)} | " + ' | '.join(cells) + ' |')

        if entry['mcnemar']:
            lines += ['', '| Pair (McNemar test) | Only first right | Only second right | p-value |',
                      '| --- | --- | --- | --- |']
            for pair, t in entry['mcnemar'].items():
                lines.append(f"| {pair} | {t['only_a_right']} | {t['only_b_right']} | {t['p_value']:.3f} |")
        lines.append('')

    # Per-item error overlap (per language, not pooled)
    lines += ['## Errors by sentence', '',
              'FN = biased sentence missed, FP = acceptable sentence flagged. '
              '✓ = detector right, ✗ = wrong.', '']
    for language, per_detector in predictions.items():
        rows = list(zip(*[per_detector[n] for n in names]))
        wrong = [rs for rs in rows if any(r['pred'] != r['label'] for r in rs)]
        all_wrong = sum(1 for rs in wrong if all(r['pred'] != r['label'] for r in rs))
        lines += [f"### {language.capitalize()}", '',
                  f"{len(wrong)} of {len(rows)} sentences have at least one error; "
                  f"{all_wrong} are wrong for every detector.", '',
                  '| Kind | Category | ' + ' | '.join(names) + ' | Sentence |',
                  '| --- | --- |' + ' --- |' * len(names) + ' --- |']
        for rs in sorted(wrong, key=lambda rs: (-sum(r['pred'] != r['label'] for r in rs), rs[0]['id'])):
            r0 = rs[0]
            kind = 'FN' if r0['label'] == 1 else ('FP (hard)' if r0['hard_negative'] else 'FP')
            marks = ' | '.join('✓' if r['pred'] == r['label'] else '✗' for r in rs)
            text = r0['text'].replace('|', '\\|')
            lines.append(f"| {kind} | {r0['category']} | {marks} | {text} |")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare detectors on the gold set")
    parser.add_argument('--detectors', default=','.join(DETECTORS),
                        help="Comma-separated subset of: " + ', '.join(DETECTORS))
    parser.add_argument('--language', '-l', choices=LANGUAGES + ['all'], default='all')
    parser.add_argument('--split', choices=SPLITS, default='test')
    parser.add_argument('--gold-dir', default=GOLD_DIR)
    parser.add_argument('--tag', default='')
    parser.add_argument('--from-json', help="Re-render the report from a saved comparison run")
    args = parser.parse_args()

    if args.from_json:
        with open(args.from_json, encoding='utf-8') as f:
            run_info = json.load(f)
        run_info['summary'] = summarise(run_info['predictions'])
        base = os.path.splitext(args.from_json)[0]
    else:
        detectors = [d.strip() for d in args.detectors.split(',') if d.strip()]
        unknown = set(detectors) - set(DETECTORS)
        if unknown:
            parser.error(f"unknown detector(s): {', '.join(sorted(unknown))}")
        languages = LANGUAGES if args.language == 'all' else [args.language]
        print(f"Comparing {', '.join(detectors)} on the {args.split} split")
        predictions = run(detectors, languages, args.split, args.gold_dir)
        if not predictions:
            sys.exit("No gold data for this split.")
        run_info = {
            'detectors': detectors, 'split': args.split, 'tag': args.tag,
            'commit': _git_commit(), 'timestamp': datetime.now().isoformat(timespec='seconds'),
            'predictions': predictions,
        }
        run_info['summary'] = summarise(predictions)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        base = os.path.join(RESULTS_DIR, f"{stamp}-compare-{args.split}" + (f"-{args.tag}" if args.tag else ''))
        with open(base + '.json', 'w', encoding='utf-8') as f:
            json.dump(run_info, f, indent=2, ensure_ascii=False)

    report = render_markdown(run_info)
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(report)
    print('\n' + report)
    print(f"\nSaved to {os.path.relpath(base, PROJECT_ROOT)}.{{json,md}}")


if __name__ == '__main__':
    main()
