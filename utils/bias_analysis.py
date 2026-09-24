import os
from typing import Dict, Any, List
from PyPDF2 import PdfReader

from analysis.bias_detector import BiasDetector
from utils.preprocessing import TextPreprocessor
from config.settings import TextPreprocessingConfig

SUPPORTED_LANGUAGES = ['english', 'spanish']


def extract_texts_from_pdf_stream(pdf_file, preprocessor: TextPreprocessor) -> List[str]:
    """Extract clean text from each page of an in-memory PDF file/stream."""
    reader = PdfReader(pdf_file)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            clean = ' '.join(text.split())
            clean = preprocessor.clean_text(clean)
            documents.append(clean)

    return documents


def run_keyword_bias_detection(pdf_file, language: str = "english") -> Dict[str, Any]:
    """
    Core function called by Streamlit or external scripts.
    Accepts an uploaded file object or file path, runs bias analysis,
    and returns structured results.
    """
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language '{language}'. Choose from {SUPPORTED_LANGUAGES}")

    # Setup processing pipeline
    preprocessor = TextPreprocessor(TextPreprocessingConfig(language=language))
    documents = extract_texts_from_pdf_stream(pdf_file, preprocessor)

    if not documents:
        raise ValueError("No extractable text was found in the provided PDF.")

    detector = BiasDetector(language=language)

    biased_count = 0
    page_results = []

    for idx, text in enumerate(documents, start=1):
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

    total_pages = len(documents)
    bias_rate = (biased_count / total_pages) if total_pages > 0 else 0.0

    return {
        "summary": {
            "total_pages": total_pages,
            "biased_pages": biased_count,
            "neutral_pages": total_pages - biased_count,
            "bias_rate": bias_rate,
            "language": language
        },
        "pages": page_results
    }
