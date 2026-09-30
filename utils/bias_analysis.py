import os
from typing import Dict, Any, List, Tuple
from PyPDF2 import PdfReader
from langdetect import detect, DetectorFactory

from analysis.bias_detector import BiasDetector
from utils.preprocessing import TextPreprocessor
from config.settings import TextPreprocessingConfig

# Set seed for deterministic language detection results
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = ['english', 'spanish']


def extract_raw_texts_from_pdf_stream(pdf_file) -> List[str]:
    """Extract raw text per page from an in-memory PDF stream."""
    reader = PdfReader(pdf_file)
    raw_pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            raw_pages.append(' '.join(text.split()))
    return raw_pages


def detect_document_language(pages: List[str]) -> str:
    """
    Detect language across the document pages using langdetect.
    Returns 'spanish' if Spanish is detected; defaults to 'english' for all others.
    """
    if not pages:
        return 'english'

    # Combine up to first 3 pages for reliable language detection
    sample_text = " ".join(pages[:3])

    try:
        lang_code = detect(sample_text)
        if lang_code == 'es':
            return 'spanish'
        return 'english'
    except Exception:
        return 'english'


def run_keyword_bias_detection(pdf_file) -> Dict[str, Any]:
    """Action 1: Keyword-based lightweight bias detection with auto-language detection."""
    # Reset stream pointer to beginning
    pdf_file.seek(0)
    raw_documents = extract_raw_texts_from_pdf_stream(pdf_file)

    if not raw_documents:
        raise ValueError("No extractable text was found in the provided PDF.")

    # Automatically detect language
    detected_language = detect_document_language(raw_documents)

    # Clean text using detected language config
    preprocessor = TextPreprocessor(TextPreprocessingConfig(language=detected_language))
    clean_documents = [preprocessor.clean_text(doc) for doc in raw_documents]

    detector = BiasDetector(language=detected_language)
    biased_count = 0
    page_results = []

    for idx, text in enumerate(clean_documents, start=1):
        analysis = detector.comprehensive_bias_analysis(text)
        if analysis.get('is_biased', False):
            biased_count += 1

        page_results.append({
            'page_number': idx,
            'text_snippet': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text,
            'is_biased': analysis.get('is_biased', False),
            'severity': analysis.get('severity', 'low'),
            'overall_bias_score': analysis.get('overall_bias_score', 0.0),
            'gender_bias': analysis.get('gender_bias', {}),
            'discriminatory_language': {
                cat: data['keywords_found']
                for cat, data in analysis.get('discriminatory_language', {}).items()
                if data.get('count', 0) > 0
            },
        })

    total_pages = len(clean_documents)
    return {
        "summary": {
            "total_pages": total_pages,
            "biased_pages": biased_count,
            "neutral_pages": total_pages - biased_count,
            "bias_rate": (biased_count / total_pages) if total_pages > 0 else 0.0,
            "detected_language": detected_language
        },
        "pages": page_results
    }


def run_ml_bias_detection(pdf_file, get_detector_fn) -> Dict[str, Any]:
    """Action 2: ML-based bias detection with auto-language detection."""
    pdf_file.seek(0)
    raw_documents = extract_raw_texts_from_pdf_stream(pdf_file)

    if not raw_documents:
        raise ValueError("No extractable text was found in the provided PDF.")

    # Automatically detect language
    detected_language = detect_document_language(raw_documents)

    # Fetch cached model corresponding to detected language
    detector_instance = get_detector_fn(detected_language)

    preprocessor = TextPreprocessor(TextPreprocessingConfig(language=detected_language))
    clean_documents = [preprocessor.clean_text(doc) for doc in raw_documents]

    biased_count = 0
    page_results = []

    for idx, text in enumerate(clean_documents, start=1):
        analysis = detector_instance.comprehensive_bias_analysis(text)
        if analysis.get('is_biased', False):
            biased_count += 1

        page_results.append({
            'page_number': idx,
            'text_snippet': text[:200] + "..." if len(text) > 200 else text,
            'full_text': text,
            'is_biased': analysis.get('is_biased', False),
            'severity': analysis.get('severity', 'low'),
            'overall_bias_score': analysis.get('overall_bias_score', 0.0),
            'ml_detection': analysis.get('ml_detection', {}),
            'ml_categorization': analysis.get('ml_categorization', {}),
            'gender_bias': analysis.get('gender_bias', {}),
            'discriminatory_language': {
                cat: data['keywords_found']
                for cat, data in analysis.get('discriminatory_language', {}).items()
                if data.get('count', 0) > 0
            },
        })

    total_pages = len(clean_documents)
    return {
        "summary": {
            "total_pages": total_pages,
            "biased_pages": biased_count,
            "neutral_pages": total_pages - biased_count,
            "bias_rate": (biased_count / total_pages) if total_pages > 0 else 0.0,
            "detected_language": detected_language
        },
        "pages": page_results
    }
