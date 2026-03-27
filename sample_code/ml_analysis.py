"""ML-based bias detection on PDF documents.

Uses MLBiasDetector which combines three signal sources:
1. Fine-tuned binary bias classifier (valurank/distilroberta-bias — English only).
2. Zero-shot categorisation (BART for English, XLM-R for Spanish).
3. Keyword / pattern analysis (shared bias_keywords module).

Usage:
    python sample_code/ml_analysis.py path/to/document.pdf
    python sample_code/ml_analysis.py path/to/document.pdf --language spanish
"""
import sys
import os
import json
import argparse

from PyPDF2 import PdfReader

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analysis.ml_bias_detector import MLBiasDetector

SUPPORTED_LANGUAGES = ['english', 'spanish']


def load_texts_from_pdf(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF.

    Each page is returned as a separate document.  Empty pages are skipped.
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


def format_result(analysis: dict) -> str:
    """Format a single comprehensive ML analysis result for display."""
    lines = []

    # Fine-tuned model result
    ml = analysis['ml_detection']
    lines.append(f"  Fine-tuned model : {ml['label']} (confidence: {ml['confidence']:.4f})")

    # Zero-shot categorisation
    cat = analysis['ml_categorization']
    lines.append(f"  Bias category    : {cat['top_category']} (score: {cat['top_score']:.4f})")
    lines.append("  Category scores:")
    for label, score in zip(cat['labels'], cat['scores']):
        bar = "#" * int(score * 40)
        lines.append(f"    {label:<25s} {score:.4f}  {bar}")

    # Gender bias keywords
    gb = analysis['gender_bias']
    lines.append(f"  Gender direction : {gb['bias_direction']}")
    if gb['male_keywords_found']:
        lines.append(f"  Male keywords    : {', '.join(gb['male_keywords_found'])}")
    if gb['female_keywords_found']:
        lines.append(f"  Female keywords  : {', '.join(gb['female_keywords_found'])}")
    if gb.get('bias_patterns_matched'):
        lines.append(f"  Bias patterns    : {', '.join(gb['bias_patterns_matched'])}")
    if gb.get('positive_context_found'):
        lines.append(f"  Positive context : {', '.join(gb['positive_context_found'])}")

    # Discriminatory language
    for category, data in analysis['discriminatory_language'].items():
        if data['count'] > 0:
            lines.append(f"  Discriminatory ({category}): {', '.join(data['keywords_found'])}")

    # Overall
    lines.append(f"  Overall score    : {analysis['overall_bias_score']:.4f}")
    lines.append(f"  Severity         : {analysis['severity']}")

    return "\n".join(lines)


def main(pdf_path: str, language: str = "english"):
    """Run ML-based bias detection on each page of a PDF."""
    print("\n" + "=" * 70)
    print("ML-BASED BIAS DETECTION — PDF ANALYSIS")
    print(f"Fine-tuned model : valurank/distilroberta-bias {'(English only)' if language != 'english' else ''}")
    print(f"Zero-shot model  : {'facebook/bart-large-mnli' if language == 'english' else 'joeddav/xlm-roberta-large-xnli'}")
    print(f"Language         : {language.upper()}")
    print("=" * 70 + "\n")

    print(f"Loading text from PDF: {pdf_path}")
    documents = load_texts_from_pdf(pdf_path)
    print(f"Loaded {len(documents)} pages from PDF\n")

    print("Initialising MLBiasDetector (downloads models on first run)...\n")
    detector = MLBiasDetector(device="cpu", language=language)

    biased_count = 0
    results_all = []

    for idx, text in enumerate(documents, start=1):
        print("-" * 70)
        print(f"Page {idx}/{len(documents)}")
        print(f"Text: {text[:120]}...")

        analysis = detector.comprehensive_bias_analysis(text)
        results_all.append(analysis)

        print(format_result(analysis))

        if analysis['is_biased']:
            biased_count += 1
            cat = analysis['ml_categorization']['top_category']
            print(f"  >> BIAS DETECTED: {cat} (severity: {analysis['severity']})")
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
    if documents:
        print(f"Bias rate                : {biased_count / len(documents):.1%}")

    # Export results
    output_dir = os.path.join(PROJECT_ROOT, 'output')
    os.makedirs(output_dir, exist_ok=True)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    json_path = os.path.join(output_dir, f'{pdf_name} - ml_bias_results.json')

    export_data = []
    for text, analysis in zip(documents, results_all):
        export_data.append({
            'text': text,
            'is_biased': analysis['is_biased'],
            'severity': analysis['severity'],
            'overall_bias_score': analysis['overall_bias_score'],
            'ml_label': analysis['ml_detection']['label'],
            'ml_confidence': analysis['ml_detection']['confidence'],
            'top_category': analysis['ml_categorization']['top_category'],
            'top_category_score': analysis['ml_categorization']['top_score'],
            'all_labels': analysis['ml_categorization']['labels'],
            'all_scores': analysis['ml_categorization']['scores'],
            'bias_direction': analysis['gender_bias']['bias_direction'],
            'male_keywords': analysis['gender_bias']['male_keywords_found'],
            'female_keywords': analysis['gender_bias']['female_keywords_found'],
            'bias_patterns': analysis['gender_bias'].get('bias_patterns_matched', []),
            'positive_context': analysis['gender_bias'].get('positive_context_found', []),
            'discriminatory_language': {
                cat: data['keywords_found']
                for cat, data in analysis['discriminatory_language'].items()
                if data['count'] > 0
            },
        })

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {json_path}")
    return results_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ML-Based Bias Detection — PDF Analysis",
    )
    parser.add_argument("pdf_file", help="Path to the PDF file to analyse")
    parser.add_argument(
        "--language", "-l",
        choices=SUPPORTED_LANGUAGES,
        default="english",
        help="Language of the PDF content (default: english)",
    )
    args = parser.parse_args()
    main(args.pdf_file, language=args.language)
