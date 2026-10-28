"""PDF text extraction with clean-up for bias analysis (ROADMAP Phase 3).

- Uses ``pypdf`` (PyPDF2 is deprecated).
- Re-joins words hyphenated across line breaks ("investiga-\\nción").
- Removes running headers and footers (lines repeated on most pages).
- Stops at the bibliography: references quote titles that mention gender or
  groups and would otherwise be flagged.
- Keeps real page numbers, and reports pages with no extractable text.
- OCRs scanned pages with ``ocrmypdf`` when it is installed (P3.5). Only
  pages without a text layer are OCRed (``--skip-text``).
"""
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from typing import Dict, List, Optional

from pypdf import PdfReader

_REFERENCES_HEADING = re.compile(
    r'^\s*(\d+(\.\d+)*\.?\s*)?(references|bibliography|works\s+cited|referencias(\s+bibliográficas)?|'
    r'bibliografía|bibliografia)\s*:?\s*$', re.I)
_HYPHEN_BREAK = re.compile(r'(\w)[-­]\n\s*(\w)')
OCR_LANGUAGES = 'spa+eng'
OCR_TIMEOUT_SECONDS = 600

logger = logging.getLogger(__name__)


def _normalise_line(line: str) -> str:
    """Line key for header/footer detection: digits removed, spaces collapsed."""
    return re.sub(r'\s+', ' ', re.sub(r'\d+', '#', line)).strip().lower()


def ocr_available() -> bool:
    """True when ``ocrmypdf`` is installed (needed for scanned PDFs)."""
    return shutil.which('ocrmypdf') is not None


def _read_bytes(pdf_file) -> bytes:
    if isinstance(pdf_file, (str, os.PathLike)):
        with open(pdf_file, 'rb') as f:
            return f.read()
    pdf_file.seek(0)
    return pdf_file.read()


def ocr_pdf(pdf_bytes: bytes) -> Optional[bytes]:
    """Add a text layer to the pages that have none; None if OCR fails."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, 'in.pdf'), os.path.join(tmp, 'out.pdf')
        with open(src, 'wb') as f:
            f.write(pdf_bytes)
        try:
            subprocess.run(['ocrmypdf', '--skip-text', '--output-type', 'pdf', '-l', OCR_LANGUAGES,
                            '--quiet', src, dst],
                           check=True, capture_output=True, timeout=OCR_TIMEOUT_SECONDS)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            stderr = getattr(e, 'stderr', b'') or b''
            logger.warning("OCR failed: %s %s", e, stderr.decode(errors='replace')[-500:])
            return None
        with open(dst, 'rb') as f:
            return f.read()


def extract_pages(pdf_file, strip_headers: bool = True,
                  stop_at_references: bool = True, ocr: bool = True) -> Dict:
    """Extract cleaned text per page.

    Returns ``{'pages': [{'page_number', 'text'}], 'empty_pages': [...],
    'ocr_pages': [...], 'references_from_page': int | None,
    'total_pages': int}``.  ``pages`` only holds pages with text, before the
    references section.  When some pages have no text and *ocr* is set and
    ``ocrmypdf`` is installed, the PDF is OCRed and extracted again;
    ``ocr_pages`` lists the pages that got their text from OCR.
    """
    result = _extract(pdf_file, strip_headers, stop_at_references)
    if ocr and result['empty_pages'] and ocr_available():
        ocred = ocr_pdf(_read_bytes(pdf_file))
        if ocred is not None:
            before = set(result['empty_pages'])
            result = _extract(io.BytesIO(ocred), strip_headers, stop_at_references)
            result['ocr_pages'] = sorted(before - set(result['empty_pages']))
    return result


def _extract(pdf_file, strip_headers: bool, stop_at_references: bool) -> Dict:
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
        'ocr_pages': [],
        'references_from_page': references_from,
        'total_pages': len(raw),
    }
