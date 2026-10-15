"""Tests for PDF extraction, including OCR of scanned pages (ROADMAP P3.5)."""
import io
import os
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.pdf_extraction import extract_pages, ocr_available

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
TEXT_PAGE = 'Este proyecto estudia la participacion de las mujeres en la ciencia.'
SCANNED_LINES = ['Los investigadores presentaran', 'los resultados del proyecto.']


def _text_page(writer: PdfWriter, text: str):
    """A page with a real text layer (Helvetica)."""
    page = writer.add_blank_page(612, 792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject(
        {NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f'BT /F1 12 Tf 72 700 Td ({text}) Tj ET'.encode('latin-1'))
    page[NameObject('/Contents')] = writer._add_object(stream)


def _scanned_page(writer: PdfWriter, lines):
    """An image-only page, as a scanner would produce."""
    img = Image.new('RGB', (1700, 2200), 'white')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 60) if os.path.exists(FONT) else ImageFont.load_default(size=60)
    for i, line in enumerate(lines):
        draw.text((150, 200 + i * 100), line, fill='black', font=font)
    buf = io.BytesIO()
    img.save(buf, format='PDF', resolution=200)
    writer.add_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])


def _mixed_pdf() -> io.BytesIO:
    writer = PdfWriter()
    _text_page(writer, TEXT_PAGE)
    _scanned_page(writer, SCANNED_LINES)
    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out


def test_without_ocr_scanned_page_is_reported_empty():
    result = extract_pages(_mixed_pdf(), ocr=False)
    assert [p['page_number'] for p in result['pages']] == [1]
    assert result['pages'][0]['text'] == TEXT_PAGE
    assert result['empty_pages'] == [2]
    assert result['ocr_pages'] == []


@pytest.mark.skipif(not ocr_available(), reason='ocrmypdf is not installed')
def test_ocr_reads_scanned_page_and_keeps_text_page():
    result = extract_pages(_mixed_pdf())
    pages = {p['page_number']: p['text'] for p in result['pages']}
    assert result['ocr_pages'] == [2]
    assert result['empty_pages'] == []
    assert pages[1] == TEXT_PAGE  # --skip-text leaves pages with a text layer alone
    assert 'investigadores' in pages[2].lower()
    assert 'resultados' in pages[2].lower()


@pytest.mark.skipif(not ocr_available(), reason='ocrmypdf is not installed')
def test_ocr_accepts_a_file_path(tmp_path):
    path = tmp_path / 'scan.pdf'
    path.write_bytes(_mixed_pdf().getvalue())
    assert extract_pages(str(path))['ocr_pages'] == [2]
