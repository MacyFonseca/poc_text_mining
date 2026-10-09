"""Keyword-based bias detection on PDF documents.

Uses BiasDetector (lightweight, no ML models): sentence-level keyword,
lemma and pattern rules from ``analysis/bias_keywords.py``.

Usage:
    python sample_code/keyword_analysis.py path/to/document.pdf
    python sample_code/keyword_analysis.py path/to/document.pdf --language spanish

Language defaults to ``auto`` (detected per page).  Results are written to
``output/<pdf name> - keyword_bias_results.json``.
"""
import argparse
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.bias_analysis import SUPPORTED_LANGUAGES, format_page_report, run_keyword_bias_detection


def main(pdf_path: str, language: str = "auto"):
    """Run keyword bias detection on each page of a PDF."""
    print("\n" + "=" * 70)
    print("KEYWORD-BASED BIAS DETECTION — PDF ANALYSIS")
    print(f"Language: {language.upper()}")
    print("=" * 70 + "\n")

    with open(pdf_path, 'rb') as pdf_file:
        results = run_keyword_bias_detection(pdf_file, language=language)

    for page in results['pages']:
        print("-" * 70)
        print(f"Page {page['page_number']}  {'BIAS DETECTED' if page['is_biased'] else 'no significant bias'}")
        print(f"Text: {page['full_text'][:120]}...")
        print(format_page_report(page))
        print()

    s = results['summary']
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"PDF file                 : {pdf_path}")
    print(f"Pages analysed           : {s['total_pages']} of {s['pdf_pages']}")
    print(f"Page languages           : {s['page_languages']}")
    if s['references_from_page']:
        print(f"References skipped from  : page {s['references_from_page']}")
    if s['empty_pages']:
        print(f"Pages without text (OCR) : {s['empty_pages']}")
    print(f"Pages flagged biased     : {s['biased_pages']}")
    print(f"Bias rate                : {s['bias_rate']:.1%}")
    print(f"Findings by type         : {s['findings_by_subtype']}")

    output_dir = os.path.join(PROJECT_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(output_dir, f'{pdf_name} - keyword_bias_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {json_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword-Based Bias Detection — PDF Analysis")
    parser.add_argument("pdf_file", help="Path to the PDF file to analyse")
    parser.add_argument(
        "--language", "-l",
        choices=['auto'] + SUPPORTED_LANGUAGES,
        default="auto",
        help="Language of the PDF content (default: auto, detected per page)",
    )
    args = parser.parse_args()
    main(args.pdf_file, language=args.language)
