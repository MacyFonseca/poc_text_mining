"""Document-level recommendation report (ROADMAP P4.6).

Groups every finding in an analysed PDF by type and by matched wording, so a
reviewer gets one entry per issue ("los investigadores", 14 times, pages
2-7) with its explanation, suggested rewrite and an example sentence,
instead of the same finding repeated page after page.

``build_report`` works on the ``{summary, pages}`` dict returned by
``utils.bias_analysis.run_bias_detection`` (any detector);
``report_to_markdown`` renders it for download.
"""
from collections import Counter
from datetime import date
from typing import Any, Dict, List, Optional

SUBTYPE_LABELS = {
    'generic_masculine': 'Generic masculine',
    'androcentric': 'Androcentric language',
    'gendered_term': 'Gendered job titles and terms',
    'generic_he': 'Generic "he"',
    'asymmetric_treatment': 'Asymmetric treatment',
    'gender_stereotype': 'Gender stereotypes',
    'explicit_requirement': 'Explicit gender requirements',
    'explicit_discrimination': 'Explicit discrimination',
    'age_requirement': 'Age requirements',
    'appearance_requirement': 'Physical appearance',
    'outdated_term': 'Outdated or pejorative terms',
    'negative_generalisation': 'Negative generalisations about groups',
    # LLM categories
    'age': 'Age', 'race': 'Race and ethnicity', 'disability': 'Disability', 'appearance': 'Physical appearance',
}

DETECTOR_LABELS = {'keyword': 'keyword rules', 'ml': 'ML models + keyword rules',
                   'llm': 'local LLM review + keyword rules'}
LEVEL_ORDER = {'high': 0, 'medium': 1, 'low': 2}
EXAMPLE_CONTEXT_CHARS = 120
MAX_CELL_CHARS = 200


def finding_level(score: float) -> str:
    """Severity of a single finding: high >= 0.6, medium >= 0.35, else low."""
    return 'high' if score >= 0.6 else 'medium' if score >= 0.35 else 'low'


def subtype_label(subtype: str) -> str:
    return SUBTYPE_LABELS.get(subtype, subtype.replace('_', ' ').capitalize())


def format_pages(pages: List[int]) -> str:
    """Compact page list: [2, 3, 4, 7] -> '2-4, 7'."""
    pages = sorted(set(pages))
    runs, start = [], None
    for i, p in enumerate(pages):
        if start is None:
            start = p
        if i + 1 == len(pages) or pages[i + 1] != p + 1:
            runs.append(str(start) if start == p else f"{start}-{p}")
            start = None
    return ', '.join(runs)


def _example(page: Dict[str, Any], finding: Dict[str, Any]) -> str:
    """The finding's sentence, or the text around it when the detector gives none."""
    if finding.get('sentence'):
        return finding['sentence']
    text = page.get('full_text', '')
    lo = max(0, finding['start'] - EXAMPLE_CONTEXT_CHARS)
    hi = min(len(text), finding['end'] + EXAMPLE_CONTEXT_CHARS)
    return ('…' if lo else '') + text[lo:hi].strip() + ('…' if hi < len(text) else '')


def build_report(results: Dict[str, Any], detector: str = 'keyword',
                 filename: Optional[str] = None) -> Dict[str, Any]:
    """Group the findings of *results* by subtype and wording.

    Returns ``{'filename', 'detector', 'date', 'summary', 'total_findings',
    'sections': [{'subtype', 'label', 'level', 'count', 'explanation',
    'pages', 'items': [{'match', 'count', 'pages', 'level', 'suggestion',
    'example'}]}]}``, sections and items ordered by severity then frequency.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for page in results.get('pages', []):
        for f in page.get('findings', []):
            key = (f['subtype'], ' '.join(f['match'].lower().split()))
            g = groups.setdefault(key, {'match': f['match'], 'count': 0, 'pages': [], 'max_score': 0.0,
                                        'suggestions': Counter(), 'explanations': Counter(),
                                        'example': _example(page, f)})
            g['count'] += 1
            g['pages'].append(page['page_number'])
            if f['score'] > g['max_score']:
                g['max_score'] = f['score']
            if f.get('suggestion'):
                g['suggestions'][f['suggestion']] += 1
            if f.get('explanation'):
                g['explanations'][f['explanation']] += 1

    by_subtype: Dict[str, List[Dict[str, Any]]] = {}
    for (subtype, _), g in groups.items():
        by_subtype.setdefault(subtype, []).append(g)

    sections = []
    for subtype, items in by_subtype.items():
        explanations = sum((g['explanations'] for g in items), Counter())
        rendered = sorted(({
            'match': g['match'],
            'count': g['count'],
            'pages': sorted(set(g['pages'])),
            'level': finding_level(g['max_score']),
            'suggestion': g['suggestions'].most_common(1)[0][0] if g['suggestions'] else None,
            'example': None if g['example'] == g['match'] else g['example'],
        } for g in items), key=lambda i: (LEVEL_ORDER[i['level']], -i['count'], i['match'].lower()))
        sections.append({
            'subtype': subtype,
            'label': subtype_label(subtype),
            'level': min((i['level'] for i in rendered), key=LEVEL_ORDER.get),
            'count': sum(i['count'] for i in rendered),
            'explanation': explanations.most_common(1)[0][0] if explanations else '',
            'pages': sorted({p for i in rendered for p in i['pages']}),
            'items': rendered,
        })
    sections.sort(key=lambda s: (LEVEL_ORDER[s['level']], -s['count']))

    return {
        'filename': filename,
        'detector': detector,
        'date': date.today().isoformat(),
        'summary': results.get('summary', {}),
        'total_findings': sum(s['count'] for s in sections),
        'sections': sections,
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ('' if n == 1 else 's')


def _cell(text: Optional[str]) -> str:
    """Make *text* safe inside a Markdown table cell, cut to MAX_CELL_CHARS."""
    if not text:
        return '—'
    text = ' '.join(str(text).split())
    if len(text) > MAX_CELL_CHARS:
        text = text[:MAX_CELL_CHARS].rsplit(' ', 1)[0] + ' …'
    return text.replace('|', '\\|')


def report_to_markdown(report: Dict[str, Any]) -> str:
    """Render a report from ``build_report`` as Markdown."""
    s = report['summary']
    title = f"Bias review: {report['filename']}" if report.get('filename') else 'Bias review'
    lines = [f"# {title}", '',
             f"{report['date']} · {DETECTOR_LABELS.get(report['detector'], report['detector'])}", '']

    if not report['sections']:
        lines.append(f"No findings in the {_plural(s.get('total_pages', 0), 'page')} analysed.")
    else:
        pages_with = len({p for sec in report['sections'] for p in sec['pages']})
        lines.append(f"{_plural(report['total_findings'], 'finding')} of {_plural(len(report['sections']), 'type')} "
                     f"on {pages_with} of {_plural(s.get('total_pages', 0), 'page')} analysed. "
                     f"{_plural(s.get('biased_pages', 0), 'page')} above the bias threshold.")
    lines.append('')

    notes = []
    if s.get('page_languages'):
        notes.append('Languages: ' + ', '.join(f"{lang} ({n} pages)" for lang, n in s['page_languages'].items()) + '.')
    if s.get('references_from_page'):
        notes.append(f"References from page {s['references_from_page']} onwards were not analysed.")
    if s.get('empty_pages'):
        notes.append(f"Pages without extractable text, possibly scanned: {format_pages(s['empty_pages'])}.")
    notes.append('Findings are automatic suggestions for a reviewer, not verdicts. '
                 'Check each one in context before rewriting.')
    lines += [f"- {n}" for n in notes] + ['']

    if report['sections']:
        lines += ['## Overview', '', '| Type | Severity | Occurrences | Pages |', '| --- | --- | --- | --- |']
        lines += [f"| {sec['label']} | {sec['level']} | {sec['count']} | {format_pages(sec['pages'])} |"
                  for sec in report['sections']]
        lines.append('')

    for sec in report['sections']:
        lines += [f"## {sec['label']}", '']
        if sec['explanation']:
            lines += [sec['explanation'], '']
        lines += ['| Wording | Times | Pages | Suggested rewrite | Example |', '| --- | --- | --- | --- | --- |']
        lines += [f"| {_cell(i['match'])} | {i['count']} | {format_pages(i['pages'])} | "
                  f"{_cell(i['suggestion'])} | {_cell(i['example'])} |" for i in sec['items']]
        lines.append('')

    return '\n'.join(lines)
