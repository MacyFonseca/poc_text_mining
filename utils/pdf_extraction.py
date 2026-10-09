"""PDF text extraction with clean-up for bias analysis (ROADMAP Phase 3).

- Uses ``pypdf`` (PyPDF2 is deprecated).
- Re-joins words hyphenated across line breaks ("investiga-\\nción").
- Removes running headers and footers (lines repeated on most pages).
- Stops at the bibliography: references quote titles that mention gender or
  groups and would otherwise be flagged.
- Keeps real page numbers, and reports pages with no extractable text
  (likely scanned; see P3.5 for OCR).
"""
import re
import shutil
from collections import Counter
from typing import Dict, List

from pypdf import PdfReader

_REFERENCES_HEADING = re.compile(
    r'^\s*(\d+(\.\d+)*\.?\s*)?(references|bibliography|works\s+cited|referencias(\s+bibliográficas)?|'
    r'bibliografía|bibliografia)\s*:?\s*$', re.I)
_HYPHEN_BREAK = re.compile(r'(\w)[-­]\n\s*(\w)')


def _normalise_line(line: str) -> str:
    """Line key for header/footer detection: digits removed, spaces collapsed."""
    return re.sub(r'\s+', ' ', re.sub(r'\d+', '#', line)).strip().lower()


def ocr_available() -> bool:
    """True when ``ocrmypdf`` is installed (needed for scanned PDFs)."""
    return shutil.which('ocrmypdf') is not None


def extract_pages(pdf_file, strip_headers: bool = True,
                  stop_at_references: bool = True) -> Dict:
    """Extract cleaned text per page.

    Returns ``{'pages': [{'page_number', 'text'}], 'empty_pages': [...],
    'references_from_page': int | None, 'total_pages': int}``.  ``pages``
    only holds pages with text, before the references section.
    """
    if hasattr(pdf_file, 'seek'):
        pdf_file.seek(0)
    reader = PdfReader(pdf_file)
    raw = []
    for page in reader.pages:
        text = page.extract_text() or ''
        text = _HYPHEN_BREAK.sub(r'\1\2', text)
        raw.append([l.strip() for l in text.split('\n') if l.strip()])

    # Running headers/footers: among the first/last 2 lines of each page,
    # those repeated on at least half the pages (min. 3 pages).
    repeated = set()
    if strip_headers and len(raw) >= 3:
        counts = Counter()
        for lines in raw:
            edge = {_normalise_line(l) for l in lines[:2] + lines[-2:]}
            counts.update(edge)
        repeated = {k for k, c in counts.items() if c >= max(3, len(raw) // 2) and k}

    pages, empty, references_from = [], [], None
    for number, lines in enumerate(raw, start=1):
        if references_from is not None:
            break
        kept = []
        for i, line in enumerate(lines):
            if (i < 2 or i >= len(lines) - 2) and _normalise_line(line) in repeated:
                continue
            if stop_at_references and _REFERENCES_HEADING.match(line):
                references_from = number
                break
            kept.append(line)
        text = ' '.join(' '.join(kept).split())
        if text:
            pages.append({'page_number': number, 'text': text})
        elif references_from is None:
            empty.append(number)

    return {
        'pages': pages,
        'empty_pages': empty,
        'references_from_page': references_from,
        'total_pages': len(raw),
    }
