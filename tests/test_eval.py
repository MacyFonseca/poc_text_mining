"""Tests for the evaluation helpers (metrics, comparison statistics, pooling filter)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.compare import bootstrap_ci, mcnemar
from eval.evaluate import compute_metrics
from eval.pool_sentences import _is_prose


def _rows(labels, preds, weights=None):
    weights = weights or [1.0] * len(labels)
    return [{'label': y, 'pred': p, 'score': float(p), 'weight': w}
            for y, p, w in zip(labels, preds, weights)]


def test_unweighted_metrics():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert (m['tp'], m['fp'], m['fn'], m['tn']) == (1, 1, 1, 1)
    assert m['precision'] == m['recall'] == 0.5


def test_weight_scales_missed_positives():
    # A missed sentence that stands for 4 others drops recall to 1/5
    m = compute_metrics([1, 1], [1, 0], [1.0, 0.0], [1.0, 4.0])
    assert m['recall'] == 0.2
    assert m['precision'] == 1.0


def test_mcnemar_identical_detectors():
    rows = _rows([1, 0, 1, 0], [1, 0, 0, 1])
    assert mcnemar(rows, rows) == {'only_a_right': 0, 'only_b_right': 0, 'p_value': 1.0}


def test_mcnemar_counts_and_p_value():
    labels = [1] * 6
    a = _rows(labels, [1] * 6)
    b = _rows(labels, [0] * 6)
    t = mcnemar(a, b)
    assert (t['only_a_right'], t['only_b_right']) == (6, 0)
    assert t['p_value'] == round(2 / 2 ** 6, 4)


def test_bootstrap_ci_contains_point_estimate():
    rows = _rows([1, 1, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 1, 0, 1, 0])
    ci = bootstrap_ci(rows, samples=500)
    f1 = compute_metrics([r['label'] for r in rows], [r['pred'] for r in rows], [0] * 8)['f1']
    assert ci['f1'][0] <= f1 <= ci['f1'][1]


def test_prose_filter():
    assert _is_prose("Los investigadores deberán presentar su currículum antes del plazo.")
    assert _is_prose("- Poner voz a las mujeres que dialogan y trabajan.")
    assert not _is_prose("Still too many for a systematic")           # line-break fragment
    assert not _is_prose("publicación del espacio online y para la posibilidad.")
    assert not _is_prose("Tabla 3.")                                    # too short
