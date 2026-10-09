"""Evaluate a bias detector against the labelled sentences in ``data/gold``.

Every detector exposes ``comprehensive_bias_analysis(text)``; this script runs
it on each gold sentence and compares ``is_biased`` with the human label.

Usage:
    python eval/evaluate.py                          # keyword detector, both languages
    python eval/evaluate.py --detector ml --language spanish
    python eval/evaluate.py --detector keyword --tag baseline

Results are printed as a table and saved to ``eval/results/`` as JSON so that
runs can be compared over time (see ROADMAP.md, "Results log").
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

GOLD_DIR = os.path.join(PROJECT_ROOT, 'data', 'gold')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'eval', 'results')
LANGUAGES = ['english', 'spanish']


def load_gold(language: str, gold_dir: str = GOLD_DIR, split: str = 'all') -> list[dict]:
    """Load gold items for *language* (and *split*: dev, test or all) from *gold_dir*."""
    items = []
    for path in sorted(glob.glob(os.path.join(gold_dir, '*.jsonl'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item['language'] == language and split in ('all', item.get('split', 'dev')):
                        items.append(item)
    return items


def build_detector(name: str, language: str):
    """Instantiate the detector called *name* for *language*."""
    if name == 'keyword':
        from analysis.bias_detector import BiasDetector
        return BiasDetector(language=language)
    if name == 'ml':
        from analysis.ml_bias_detector import MLBiasDetector
        return MLBiasDetector(device='cpu', language=language)
    if name == 'llm':
        from analysis.llm_bias_detector import LLMBiasDetector
        return LLMBiasDetector(language=language)
    raise ValueError(f"Unknown detector: {name}")


def compute_metrics(labels: list[int], preds: list[int], scores: list[float]) -> dict:
    """Precision, recall, F1, accuracy and average precision (PR-AUC)."""
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    try:
        from sklearn.metrics import average_precision_score
        pr_auc = float(average_precision_score(labels, scores)) if len(set(labels)) > 1 else None
    except ImportError:
        pr_auc = None
    return {
        'n': len(labels), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'accuracy': round((tp + tn) / len(labels), 4) if labels else 0.0,
        'pr_auc': round(pr_auc, 4) if pr_auc is not None else None,
    }


def evaluate(detector_name: str, language: str, gold_dir: str = GOLD_DIR,
             split: str = 'all') -> dict:
    """Run *detector_name* over the gold set for *language* and return metrics."""
    items = load_gold(language, gold_dir, split)
    if not items:
        return {}
    detector = build_detector(detector_name, language)

    start = time.time()
    rows = []
    for item in items:
        analysis = detector.comprehensive_bias_analysis(item['text'])
        rows.append({
            'id': item['id'],
            'text': item['text'],
            'label': item['label'],
            'category': item['category'],
            'hard_negative': item.get('hard_negative', False),
            'pred': int(bool(analysis['is_biased'])),
            'score': float(analysis['overall_bias_score']),
        })
    elapsed = time.time() - start

    labels = [r['label'] for r in rows]
    preds = [r['pred'] for r in rows]
    scores = [r['score'] for r in rows]
    metrics = compute_metrics(labels, preds, scores)

    # Recall per bias category (how many of each kind the detector catches)
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        if r['label'] == 1:
            by_cat[r['category']][1] += 1
            by_cat[r['category']][0] += r['pred']
    metrics['recall_by_category'] = {c: round(hit / tot, 4) for c, (hit, tot) in sorted(by_cat.items())}

    # False-positive rate on hard negatives (neutral text that talks about gender or groups)
    hard = [r for r in rows if r['hard_negative']]
    metrics['hard_negative_fp_rate'] = (
        round(sum(r['pred'] for r in hard) / len(hard), 4) if hard else None
    )
    metrics['seconds_per_item'] = round(elapsed / len(rows), 3)

    return {
        'metrics': metrics,
        'errors': [r for r in rows if r['label'] != r['pred']],
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=PROJECT_ROOT, text=True,
        ).strip()
    except Exception:
        return 'unknown'


def main():
    parser = argparse.ArgumentParser(description="Evaluate a bias detector on the gold set")
    parser.add_argument('--detector', '-d', choices=['keyword', 'ml', 'llm'], default='keyword')
    parser.add_argument('--language', '-l', choices=LANGUAGES + ['all'], default='all')
    parser.add_argument('--gold-dir', default=GOLD_DIR)
    parser.add_argument('--split', choices=['dev', 'test', 'all'], default='all',
                        help="dev = sentences rules were tuned on; test = held-out sentences")
    parser.add_argument('--tag', default='', help="Label saved with the run, e.g. 'baseline'")
    parser.add_argument('--show-errors', action='store_true', help="Print misclassified sentences")
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    languages = LANGUAGES if args.language == 'all' else [args.language]
    run = {
        'detector': args.detector,
        'tag': args.tag,
        'split': args.split,
        'commit': _git_commit(),
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'results': {},
    }

    print(f"\nDetector: {args.detector}   split: {args.split}   commit: {run['commit']}   tag: {args.tag or '-'}\n")
    print(f"{'language':<9} {'n':>4} {'P':>6} {'R':>6} {'F1':>6} {'PR-AUC':>7} {'hardFP':>7} {'s/item':>7}")
    for language in languages:
        result = evaluate(args.detector, language, args.gold_dir, args.split)
        if not result:
            print(f"{language:<9} no gold data")
            continue
        run['results'][language] = result
        m = result['metrics']
        pr_auc = f"{m['pr_auc']:.3f}" if m['pr_auc'] is not None else '-'
        hard_fp = f"{m['hard_negative_fp_rate']:.3f}" if m['hard_negative_fp_rate'] is not None else '-'
        print(f"{language:<9} {m['n']:>4} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
              f"{pr_auc:>7} {hard_fp:>7} {m['seconds_per_item']:>7.3f}")
        print(f"          recall by category: {m['recall_by_category']}")
        if args.show_errors:
            for e in result['errors']:
                kind = 'FN' if e['label'] == 1 else 'FP'
                print(f"          {kind} {e['score']:.3f} [{e['category']}] {e['text']}")

    if not args.no_save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        suffix = f"-{args.split}" + (f"-{args.tag}" if args.tag else '')
        path = os.path.join(RESULTS_DIR, f"{stamp}-{args.detector}{suffix}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(run, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {os.path.relpath(path, PROJECT_ROOT)}")


if __name__ == '__main__':
    main()
