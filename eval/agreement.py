"""Inter-annotator agreement (Cohen's kappa) between two annotation files.

Usage:
    python eval/agreement.py annotator_a.jsonl annotator_b.jsonl

Both files use the gold-set format (see data/gold/README.md); items are
matched by ``id``.  Reports kappa for the binary label and for the category.
"""
import json
import sys


def load(path):
    with open(path, encoding='utf-8') as f:
        return {item['id']: item for item in (json.loads(l) for l in f if l.strip())}


def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def main(path_a, path_b):
    a, b = load(path_a), load(path_b)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("No shared ids between the two files.")
    for field in ('label', 'category'):
        xa = [a[i][field] for i in common]
        xb = [b[i][field] for i in common]
        print(f"{field:<9} kappa = {cohen_kappa(xa, xb):.3f}   (n = {len(common)})")
    disagreements = [i for i in common if a[i]['label'] != b[i]['label']]
    if disagreements:
        print(f"\n{len(disagreements)} label disagreements:")
        for i in disagreements:
            print(f"  {i}: A={a[i]['label']} B={b[i]['label']}  {a[i]['text']}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
