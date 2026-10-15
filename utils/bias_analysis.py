"""PDF -> per-page bias analysis, shared by the web app and the CLI scripts.

Each page is analysed in its own detected language, so a Spanish proposal
with an English abstract uses the right rules and models on every page.
"""
from collections import Counter
from typing import Any, Callable, Dict, List

from langdetect import DetectorFactory, detect_langs

from analysis.bias_detector import BiasDetector
from config.settings import TextPreprocessingConfig
from utils.pdf_extraction import extract_pages, ocr_available
from utils.preprocessing import TextPreprocessor

# Set seed for deterministic language detection results
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ['english', 'spanish']
MIN_CHARS_FOR_PAGE_LANGUAGE = 200


def _detect(text: str) -> str:
    """'spanish' if langdetect's top guess is Spanish, else 'english'."""
    try:
        return 'spanish' if detect_langs(text)[0].lang == 'es' else 'english'
    except Exception:
        return 'english'


def detect_document_language(pages: List[str]) -> str:
    """Language of the document from its first 3 pages ('spanish' or 'english')."""
    if not pages:
        return 'english'
    return _detect(' '.join(pages[:3]))


def detect_page_languages(pages: List[str]) -> List[str]:
    """Language per page; short pages inherit the document language."""
    document_language = detect_document_language(pages)
    return [
        _detect(p) if len(p) >= MIN_CHARS_FOR_PAGE_LANGUAGE else document_language
        for p in pages
    ]


def _keyword_detector(language: str):
    return BiasDetector(language=language)


def run_bias_detection(pdf_file, get_detector_fn: Callable[[str], Any],
                       language: str = 'auto') -> Dict[str, Any]:
    """Extract, clean and analyse every page of *pdf_file*.

    ``get_detector_fn(language)`` returns a detector for that language
    (cached by the caller for ML models).  ``language='auto'`` detects the
    language per page; otherwise every page uses the given language.
    """
    extraction = extract_pages(pdf_file)
    pages = extraction['pages']
    if not pages:
        hint = ("OCR found no text either." if ocr_available()
                 else "It may be a scanned document: install ocrmypdf to read it.")
        raise ValueError("No extractable text was found in the provided PDF. " + hint)

    texts = [p['text'] for p in pages]
    languages = detect_page_languages(texts) if language == 'auto' else [language] * len(texts)

    cleaners = {lang: TextPreprocessor(TextPreprocessingConfig(language=lang)) for lang in set(languages)}
    detectors = {lang: get_detector_fn(lang) for lang in set(languages)}

    page_results = []
    for page, lang in zip(pages, languages):
        text = cleaners[lang].clean_text(page['text'])
        analysis = detectors[lang].comprehensive_bias_analysis(text)
        page_results.append({
            'page_number': page['page_number'],
            'language': lang,
            'text_snippet': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text,
            'is_biased': analysis.get('is_biased', False),
            'severity': analysis.get('severity', 'none'),
            'overall_bias_score': analysis.get('overall_bias_score', 0.0),
            'findings': analysis.get('findings', []),
            'ml_detection': analysis.get('ml_detection', {}),
            'ml_categorization': analysis.get('ml_categorization', {}),
            'llm_review': analysis.get('llm_review', {}),
            'signal_scores': analysis.get('signal_scores', {}),
            'gender_bias': analysis.get('gender_bias', {}),
            'discriminatory_language': {
                cat: data['keywords_found']
                for cat, data in analysis.get('discriminatory_language', {}).items()
                if data.get('count', 0) > 0
            },
        })

    total = len(page_results)
    biased = sum(p['is_biased'] for p in page_results)
    language_counts = Counter(languages)
    findings = [f for p in page_results for f in p['findings']]
    return {
        "summary": {
            "total_pages": total,
            "biased_pages": biased,
            "neutral_pages": total - biased,
            "bias_rate": biased / total if total else 0.0,
            "detected_language": language_counts.most_common(1)[0][0],
            "page_languages": dict(language_counts),
            "pdf_pages": extraction['total_pages'],
            "empty_pages": extraction['empty_pages'],
            "ocr_pages": extraction['ocr_pages'],
            "references_from_page": extraction['references_from_page'],
            "findings_by_subtype": dict(Counter(f['subtype'] for f in findings)),
        },
        "pages": page_results,
    }


def run_keyword_bias_detection(pdf_file, language: str = 'auto') -> Dict[str, Any]:
    """Keyword-based lightweight bias detection with per-page language detection."""
    return run_bias_detection(pdf_file, _keyword_detector, language)


def run_ml_bias_detection(pdf_file, get_detector_fn, language: str = 'auto') -> Dict[str, Any]:
    """ML-based bias detection; *get_detector_fn* returns a (cached) MLBiasDetector."""
    return run_bias_detection(pdf_file, get_detector_fn, language)


def run_llm_bias_detection(pdf_file, get_detector_fn, language: str = 'auto') -> Dict[str, Any]:
    """LLM review (Ollama); *get_detector_fn* returns an LLMBiasDetector."""
    return run_bias_detection(pdf_file, get_detector_fn, language)


def format_page_report(page: Dict[str, Any]) -> str:
    """Plain-text summary of one page result, used by the CLI scripts."""
    lines = [f"  Language         : {page['language']}",
             f"  Overall score    : {page['overall_bias_score']:.4f}",
             f"  Severity         : {page['severity']}"]
    gb = page.get('gender_bias', {})
    if gb:
        lines.append(f"  Gender direction : {gb.get('bias_direction', 'n/a')}")
    if page.get('signal_scores'):
        lines.append("  Signals          : " + ', '.join(f"{k}={v:.2f}" for k, v in page['signal_scores'].items()))
    for f in page['findings']:
        line = f"  - [{f['subtype']}] \"{f['match']}\" ({f['score']:.2f})"
        if f.get('suggestion'):
            line += f" -> {f['suggestion']}"
        lines.append(line)
    return '\n'.join(lines)
