"""
Example script demonstrating ML-based bias detection on PDF documents.

Uses BiasDetector.ml_based_bias_detection() (facebook/bart-large-mnli)
to perform zero-shot classification on each page extracted from a PDF.
"""
import sys
import os
import json
import argparse
from PyPDF2 import PdfReader

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.bias_detector import BiasDetector

SUPPORTED_LANGUAGES = ['english', 'spanish']

# Default zero-shot categories for bias classification
BIAS_CATEGORIES = [
    "gender bias",
    "age discrimination",
    "racial discrimination",
    "disability discrimination",
    "appearance discrimination",
    "neutral and inclusive",
]


def load_texts_from_pdf(pdf_path: str) -> list[str]:
    """Load and split text from a PDF file into a list of documents.

    Each page of the PDF is treated as a separate document.
    Empty pages are skipped.
    """
    reader = PdfReader(pdf_path)
    documents = []
    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            clean = ' '.join(text.split())
            documents.append(clean)

    if not documents:
        print(f"Error: No text could be extracted from {pdf_path}")
        sys.exit(1)

    return documents


def format_ml_result(result: dict) -> str:
    """Format a single ML bias detection result for display."""
    lines = []
    lines.append(f"  Top category : {result['top_category']}")
    lines.append(f"  Top score    : {result['top_score']:.4f}")
    lines.append("  All scores:")
    for label, score in zip(result['labels'], result['scores']):
        bar = "#" * int(score * 40)
        lines.append(f"    {label:<30s} {score:.4f}  {bar}")
    return "\n".join(lines)


def main(pdf_path: str, language: str = "english"):
    """Run ML-based bias detection on all pages of a PDF."""
    print("\n" + "=" * 70)
    print("ML-BASED BIAS DETECTION - PDF ANALYSIS")
    print(f"Model: facebook/bart-large-mnli")
    print(f"Language: {language.upper()}")
    print("=" * 70 + "\n")

    # Load data from PDF
    print(f"Loading text from PDF: {pdf_path}")
    documents = load_texts_from_pdf(pdf_path)
    print(f"Loaded {len(documents)} documents (pages) from PDF\n")

    # Initialize detector
    print("Initializing BiasDetector (this downloads the model on first run)...\n")
    detector = BiasDetector(language=language)

    biased_count = 0
    results_all = []

    for idx, text in enumerate(documents, start=1):
        print("-" * 70)
        print(f"Page {idx}/{len(documents)}")
        print(f"Text: {text[:120]}...")

        result = detector.ml_based_bias_detection(text, BIAS_CATEGORIES)
        results_all.append(result)

        print(format_ml_result(result))

        is_biased = result['top_category'] != "neutral and inclusive"
        if is_biased:
            biased_count += 1
            print(f"  >> BIAS DETECTED: {result['top_category']}")
        else:
            print("  >> No significant bias detected")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PDF file                 : {pdf_path}")
    print(f"Total pages analysed     : {len(documents)}")
    print(f"Pages flagged biased     : {biased_count}")
    print(f"Pages deemed neutral     : {len(documents) - biased_count}")
    print(f"Bias rate                : {biased_count / len(documents):.1%}")

    # Export results
    output_dir = os.path.join(os.path.dirname(__file__), 'output_from_pdf')
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'ml_bias_pdf_results.json')

    export_data = []
    for text, result in zip(documents, results_all):
        export_data.append({
            'text': text,
            'top_category': result['top_category'],
            'top_score': result['top_score'],
            'all_labels': result['labels'],
            'all_scores': result['scores'],
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_path}")

    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML-Based Bias Detection - PDF Analysis"
    )
    parser.add_argument(
        "pdf_file",
        help="Path to the PDF file to analyze"
    )
    parser.add_argument(
        "--language", "-l",
        choices=SUPPORTED_LANGUAGES,
        default="english",
        help="Language of the PDF content (default: english)"
    )
    args = parser.parse_args()

    main(args.pdf_file, language=args.language)
