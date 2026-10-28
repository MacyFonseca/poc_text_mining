"""Tests for the document-level recommendation report (ROADMAP P4.6)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.bias_detector import BiasDetector
from utils.report import build_report, format_pages, report_to_markdown


def _finding(match, subtype, score, start=0, suggestion=None, explanation='Why.', sentence=None):
    return {'type': 'non_inclusive', 'category': 'gender', 'subtype': subtype, 'match': match,
            'start': start, 'end': start + len(match), 'sentence': sentence, 'score': score,
            'explanation': explanation, 'suggestion': suggestion}


def _results(pages):
    return {'summary': {'total_pages': len(pages), 'biased_pages': 0, 'page_languages': {'spanish': len(pages)},
                        'references_from_page': 9, 'empty_pages': []},
            'pages': [{'page_number': n, 'full_text': text, 'findings': findings} for n, text, findings in pages]}


def test_format_pages_compacts_runs():
    assert format_pages([7, 2, 3, 4, 4]) == '2-4, 7'
    assert format_pages([5]) == '5'
    assert format_pages([]) == ''


def test_repeated_matches_merge_across_pages_and_case():
    gm = 'generic_masculine'
    results = _results([
        (2, 'Los investigadores...', [_finding('Los investigadores', gm, 0.3, suggestion='el personal investigador')]),
        (3, 'los investigadores...', [_finding('los investigadores', gm, 0.3, suggestion='el personal investigador')]),
        (5, 'los investigadores...', [_finding('los investigadores', gm, 0.3, suggestion='el personal investigador'),
                                      _finding('los alumnos', gm, 0.3, start=30, suggestion='el alumnado')]),
    ])
    report = build_report(results, 'keyword', 'prop.pdf')
    assert report['total_findings'] == 4
    [section] = report['sections']
    assert section['subtype'] == gm and section['count'] == 4 and section['pages'] == [2, 3, 5]
    first = section['items'][0]
    assert first['count'] == 3 and first['pages'] == [2, 3, 5]
    assert first['suggestion'] == 'el personal investigador'


def test_sections_ordered_by_severity_then_count():
    results = _results([(1, 'x' * 50, [
        _finding('los alumnos', 'generic_masculine', 0.3),
        _finding('los padres', 'generic_masculine', 0.3, start=12),
        _finding('only men may apply', 'explicit_requirement', 0.9, start=24),
    ])])
    report = build_report(results, 'keyword')
    assert [s['subtype'] for s in report['sections']] == ['explicit_requirement', 'generic_masculine']
    assert report['sections'][0]['level'] == 'high'


def test_example_falls_back_to_page_context_without_sentence():
    text = 'Intro. ' + 'The chairman will decide.' + ' Outro.'
    f = _finding('chairman', 'gendered_term', 0.3, start=text.index('chairman'))
    report = build_report(_results([(4, text, [f])]), 'llm')
    assert 'The chairman will decide.' in report['sections'][0]['items'][0]['example']


def test_markdown_escapes_pipes_and_lists_notes():
    f = _finding('a | b', 'gendered_term', 0.3, sentence='Uses a | b here.', suggestion='c')
    md = report_to_markdown(build_report(_results([(1, 'Uses a | b here.', [f])]), 'keyword', 'doc.pdf'))
    assert md.startswith('# Bias review: doc.pdf')
    assert 'a \\| b' in md
    assert 'References from page 9 onwards were not analysed.' in md
    assert '1 finding of 1 type on 1 of 1 page analysed.' in md


def test_empty_report():
    md = report_to_markdown(build_report(_results([(1, 'Neutral text.', [])]), 'keyword'))
    assert 'No findings in the 1 page analysed.' in md
    assert '## Overview' not in md


def test_report_from_keyword_detector_output():
    detector = BiasDetector(language='spanish')
    texts = ['Los investigadores presentarán los resultados.', 'Los investigadores y los alumnos asistirán.']
    pages = [(i + 1, t, detector.comprehensive_bias_analysis(t)['findings']) for i, t in enumerate(texts)]
    report = build_report(_results(pages), 'keyword', 'prop.pdf')
    items = {i['match'].lower(): i for s in report['sections'] for i in s['items']}
    assert items['investigadores']['count'] == 2
    assert items['investigadores']['pages'] == [1, 2]
    assert items['investigadores']['suggestion']
