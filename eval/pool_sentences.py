"""Build a realistic test set from real PDFs by pooling detector output.

Step 1, pool: split the PDFs into sentences, run the detectors, and keep
every sentence that any detector flags plus a random sample of the rest.

    python eval/pool_sentences.py pool data/proposals/*.pdf [--detectors keyword,ml,llm] [--random 50]

Use real research proposals (the kind of document the tool is meant for); put them in
``data/proposals/`` (git-ignored).

This writes, per language, ``data/annotation/pool_<language>.csv`` (for the
annotator: id, text and empty label columns, in random order, without the
detector flags so they do not anchor the labels) and ``pool_meta.json``
(document, page, which detectors flagged each sentence, sampling weight).

Step 2, label the CSV following data/gold/README.md: ``label`` 1/0,
``category``, ``hard_negative`` true/false, optional ``note``.

Step 3, import the labelled CSV as split ``real``:

    python eval/pool_sentences.py import data/annotation/pool_spanish.csv --annotator ab

which writes ``data/gold/real_<language>.jsonl``; then
``python eval/compare.py --split real``.

Sampling weights: flagged sentences are all kept (weight 1); an unflagged
sentence drawn at random from N stands for N / n_sampled sentences, so the
weighted precision and recall estimate the rates over whole documents.
The LLM reviews whole pages (one call per page instead of per sentence);
a sentence counts as LLM-flagged when one of its quotes falls inside it.
"""
import argparse
import csv
import gc
import json
import os
import random
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from eval.evaluate import GOLD_DIR, build_detector

ANNOTATION_DIR = os.path.join(PROJECT_ROOT, 'data', 'annotation')
META_PATH = os.path.join(ANNOTATION_DIR, 'pool_meta.json')
DETECTORS = ['keyword', 'ml', 'llm']
CATEGORIES = ['gender_stereotype', 'explicit_discrimination', 'generic_masculine', 'androcentric',
              'gendered_term', 'other_discrimination', 'none']
MIN_WORDS, MAX_WORDS = 6, 80
CSV_FIELDS = ['id', 'text', 'label', 'category', 'hard_negative', 'note']


def _is_prose(text: str) -> bool:
    """Skip table rows, reference fragments and other non-sentences."""
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    letters = sum(c.isalpha() for c in text)
    if letters / len(text) < 0.6:
        return False
    # Line-break fragments start mid-sentence or stop without punctuation
    body = text.lstrip('-•–* ')
    return (body[:1].isupper() or body[:1].isdigit() or body[:1] in '¿¡"“«(') \
        and text.rstrip()[-1:] in '.!?:;)"”»'


def extract_sentences(pdf_paths: list[str]) -> list[dict]:
    """Prose sentences of every PDF page, in the page's detected language."""
    from config.settings import TextPreprocessingConfig
    from utils.bias_analysis import detect_page_languages
    from utils.nlp import get_nlp
    from utils.pdf_extraction import extract_pages
    from utils.preprocessing import TextPreprocessor

    sentences, pages, seen = [], [], set()
    for path in pdf_paths:
        extraction = extract_pages(path)
        texts = [p['text'] for p in extraction['pages']]
        for page, language in zip(extraction['pages'], detect_page_languages(texts)):
            text = TextPreprocessor(TextPreprocessingConfig(language=language)).clean_text(page['text'])
            page_index = len(pages)
            pages.append({'document': os.path.basename(path), 'page': page['page_number'],
                          'language': language, 'text': text})
            for sent in get_nlp(language)(text).sents:
                s = re.sub(r'\s+', ' ', sent.text).strip()
                key = s.lower()
                if not _is_prose(s) or key in seen:
                    continue
                seen.add(key)
                sentences.append({'text': s, 'language': language, 'page_index': page_index,
                                  'start': sent.start_char, 'end': sent.end_char,
                                  'document': os.path.basename(path), 'page': page['page_number'],
                                  'flags': {}})
    return sentences, pages


def flag_sentences(sentences: list[dict], pages: list[dict], detectors: list[str]) -> None:
    """Fill ``sentence['flags'][detector]`` with 0/1, one detector at a time."""
    for name in detectors:
        for language in sorted({s['language'] for s in sentences}):
            group = [s for s in sentences if s['language'] == language]
            print(f"  {language:<8} {name:<8} {len(group)} sentences ...", flush=True)
            detector = build_detector(name, language)
            if name == 'llm':
                if not detector.is_available():
                    sys.exit("Ollama is not running or the model is missing: start it with `ollama serve`.")
                spans = {}
                for i in sorted({s['page_index'] for s in group}):
                    spans[i] = [(f['start'], f['end']) for f in detector.review(pages[i]['text'])['findings']]
                for s in group:
                    s['flags'][name] = int(any(a < s['end'] and b > s['start'] for a, b in spans[s['page_index']]))
            else:
                for s in group:
                    s['flags'][name] = int(bool(detector.comprehensive_bias_analysis(s['text'])['is_biased']))
            del detector
            gc.collect()


def pool(args) -> None:
    detectors = [d.strip() for d in args.detectors.split(',') if d.strip()]
    sentences, pages = extract_sentences(args.pdfs)
    print(f"{len(sentences)} prose sentences from {len(args.pdfs)} PDFs")
    flag_sentences(sentences, pages, detectors)

    rng = random.Random(args.seed)
    os.makedirs(ANNOTATION_DIR, exist_ok=True)
    meta = {'detectors': detectors, 'pdfs': [os.path.basename(p) for p in args.pdfs],
            'seed': args.seed, 'languages': {}, 'items': {}}
    for language in sorted({s['language'] for s in sentences}):
        group = [s for s in sentences if s['language'] == language]
        flagged = [s for s in group if any(s['flags'].values())]
        rest = [s for s in group if not any(s['flags'].values())]
        sampled = rng.sample(rest, min(args.random, len(rest)))
        weight = len(rest) / len(sampled) if sampled else 1.0
        chosen = [(s, 1.0) for s in flagged] + [(s, weight) for s in sampled]
        rng.shuffle(chosen)

        prefix = 'real-en' if language == 'english' else 'real-es'
        path = os.path.join(ANNOTATION_DIR, f'pool_{language}.csv')
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for i, (s, w) in enumerate(chosen, 1):
                item_id = f'{prefix}-{i:04d}'
                writer.writerow({'id': item_id, 'text': s['text']})
                meta['items'][item_id] = {'language': language, 'document': s['document'],
                                          'page': s['page'], 'flags': s['flags'],
                                          'weight': round(w, 4), 'pooled_by': 'flag' if any(s['flags'].values()) else 'random'}
        meta['languages'][language] = {'sentences': len(group), 'flagged': len(flagged),
                                       'unflagged': len(rest), 'sampled': len(sampled),
                                       'unflagged_weight': round(weight, 4)}
        print(f"  {language}: {len(group)} sentences, {len(flagged)} flagged, "
              f"{len(sampled)} of {len(rest)} unflagged sampled -> {os.path.relpath(path, PROJECT_ROOT)}")

    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Label the CSVs (see data/gold/README.md), then run the 'import' step.")


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in ('1', 'true', 'yes', 'y', 'si', 'sí')


def import_labels(args) -> None:
    with open(META_PATH, encoding='utf-8') as f:
        meta = json.load(f)['items']
    with open(args.csv, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))

    items, problems = [], []
    for row in rows:
        label = row.get('label', '').strip()
        if label == '':
            continue                       # not labelled yet
        m = meta.get(row['id'])
        category = (row.get('category') or '').strip() or ('none' if label == '0' else '')
        if m is None or label not in ('0', '1') or category not in CATEGORIES \
                or (label == '0') != (category == 'none'):
            problems.append(row['id'])
            continue
        item = {'id': row['id'], 'language': m['language'], 'split': 'real', 'text': row['text'],
                'label': int(label), 'category': category,
                'hard_negative': label == '0' and _parse_bool(row.get('hard_negative', '')),
                'source': f'expert-{args.annotator}', 'document': m['document'], 'page': m['page'],
                'weight': m['weight']}
        if (row.get('note') or '').strip():
            item['note'] = row['note'].strip()
        items.append(item)

    if problems:
        sys.exit(f"Fix these rows first (unknown id, label not 0/1, or category not matching the label "
                 f"/ not one of {', '.join(CATEGORIES)}): {', '.join(problems)}")
    if not items:
        sys.exit("No labelled rows found.")
    language = items[0]['language']
    out = os.path.join(GOLD_DIR, f'real_{language}.jsonl')
    with open(out, 'w', encoding='utf-8') as f:
        for item in sorted(items, key=lambda i: i['id']):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"{len(items)} of {len(rows)} rows labelled -> {os.path.relpath(out, PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Pool PDF sentences for annotation, or import labels")
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('pool', help="Extract and pool sentences from PDFs")
    p.add_argument('pdfs', nargs='+')
    p.add_argument('--detectors', default=','.join(DETECTORS))
    p.add_argument('--random', type=int, default=50, help="Unflagged sentences sampled per language")
    p.add_argument('--seed', type=int, default=0)
    p.set_defaults(func=pool)
    i = sub.add_parser('import', help="Turn a labelled pool CSV into data/gold/real_<language>.jsonl")
    i.add_argument('csv')
    i.add_argument('--annotator', required=True, help="Initials, stored as source 'expert-<initials>'")
    i.set_defaults(func=import_labels)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
