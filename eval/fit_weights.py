"""Learn the ML detector's signal weights and threshold from the gold set (ROADMAP P2.5).

Runs every signal source once per gold sentence, caches the scores, fits a
logistic regression on the ``dev`` split and reports both splits.  The
logistic model is converted back into the detector's weighted-sum form:

    w . x + b >= 0   <=>   (w / sum(w)) . x >= -b / sum(w)

so the detector keeps its ``weights`` + ``overall_bias_threshold`` contract.

Usage:
    python eval/fit_weights.py --language spanish            # report only
    python eval/fit_weights.py --language spanish --save     # write config/ml_weights.json

Only use --save once the dev split holds expert-labelled data (P0.2).
"""
import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.evaluate import GOLD_DIR, RESULTS_DIR, compute_metrics, load_gold

NAMES = ['fine_tuned', 'zero_shot', 'gender_keyword', 'discriminatory_keyword']
WEIGHTS_FILE = os.path.join(PROJECT_ROOT, 'config', 'ml_weights.json')


def extract_signals(language: str, refresh: bool = False) -> list[dict]:
    """Signal scores for every gold item of *language* (cached on disk)."""
    cache = os.path.join(RESULTS_DIR, f'signals-{language}.json')
    items = load_gold(language, GOLD_DIR, 'all')
    if not refresh and os.path.exists(cache):
        with open(cache, encoding='utf-8') as f:
            cached = {r['id']: r for r in json.load(f)}
        if all(i['id'] in cached and cached[i['id']]['text'] == i['text'] for i in items):
            return [cached[i['id']] for i in items]

    from analysis.ml_bias_detector import MLBiasDetector
    detector = MLBiasDetector(device='cpu', language=language)
    rows = []
    for item in items:
        s = detector.signal_scores(item['text'])['scores']
        rows.append({'id': item['id'], 'text': item['text'], 'label': item['label'],
                     'split': item.get('split', 'dev'), 'available_ft': language == 'english', **s})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    return rows


def score_with(rows, weights, threshold):
    scores = [min(sum(r[n] * w for n, w in zip(NAMES, weights)), 1.0) for r in rows]
    return scores, [int(s >= threshold) for s in scores]


def default_weights(language: str):
    from analysis.ml_bias_detector import DEFAULT_WEIGHTS
    w = list(DEFAULT_WEIGHTS)
    if language != 'english':          # fine-tuned model unavailable: redistribute
        missing, w[0] = w[0], 0.0
        total = sum(w[1:])
        w = [0.0] + [v + missing * v / total for v in w[1:]]
    return w, 0.25


def fit(rows):
    from sklearn.linear_model import LogisticRegression
    X = [[r[n] for n in NAMES] for r in rows]
    y = [r['label'] for r in rows]
    model = LogisticRegression(C=1.0, class_weight='balanced').fit(X, y)
    coef = [max(c, 0.0) for c in model.coef_[0]]
    total = sum(coef) or 1.0
    weights = [c / total for c in coef]
    threshold = max(0.0, min(1.0, -model.intercept_[0] / total))
    return weights, threshold


def report(name, rows, weights, threshold):
    for split in ('dev', 'test'):
        part = [r for r in rows if r['split'] == split]
        if not part:
            continue
        scores, preds = score_with(part, weights, threshold)
        m = compute_metrics([r['label'] for r in part], preds, scores)
        print(f"  {name:<9} {split:<5} n={m['n']:>3}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  PR-AUC={m['pr_auc'] if m['pr_auc'] is not None else '-'}")


def main():
    parser = argparse.ArgumentParser(description="Fit ML detector weights on the dev split")
    parser.add_argument('--language', '-l', choices=['english', 'spanish'], required=True)
    parser.add_argument('--refresh', action='store_true', help="Recompute cached signals")
    parser.add_argument('--save', action='store_true', help="Write config/ml_weights.json")
    args = parser.parse_args()

    rows = extract_signals(args.language, args.refresh)
    dev = [r for r in rows if r['split'] == 'dev']
    if len({r['label'] for r in dev}) < 2:
        sys.exit("Dev split needs both labels to fit weights.")

    print(f"\n{args.language}: signals for {len(rows)} sentences")
    w0, t0 = default_weights(args.language)
    print(f"  default weights {[round(x, 3) for x in w0]}  threshold {t0}")
    report('default', rows, w0, t0)

    w, t = fit(dev)
    print(f"  learned weights {[round(x, 3) for x in w]}  threshold {t:.3f}")
    report('learned', rows, w, t)

    if args.save:
        data = {}
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, encoding='utf-8') as f:
                data = json.load(f)
        data[args.language] = {'weights': [round(x, 4) for x in w], 'threshold': round(t, 4),
                               'n_dev': len(dev)}
        with open(WEIGHTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved to {os.path.relpath(WEIGHTS_FILE, PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
