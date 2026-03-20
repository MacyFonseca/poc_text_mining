"""
Example script demonstrating the complete text mining pipeline.
Loads sample texts from a PDF file instead of hardcoded data.
"""
import sys
import os
import argparse
from PyPDF2 import PdfReader

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipelines.text_mining_pipeline import TextMiningPipeline
from config.settings import (
    PipelineConfig, TopicModelingConfig, ClusteringConfig
)

SUPPORTED_LANGUAGES = ['english', 'spanish']


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
            # Collapse newlines and extra whitespace from PDF layout
            clean = ' '.join(text.split())
            documents.append(clean)

    if not documents:
        print(f"Error: No text could be extracted from {pdf_path}")
        sys.exit(1)

    return documents


def main(pdf_path: str, language: str = "english"):
    """Run the complete text mining pipeline with PDF data."""
    print("\n" + "=" * 70)
    print("TEXT MINING AND BIAS DETECTION PIPELINE - PDF ANALYSIS")
    print(f"Language: {language.upper()}")
    print("=" * 70 + "\n")

    # Load data from PDF
    print(f"Loading text from PDF: {pdf_path}")
    documents = load_texts_from_pdf(pdf_path)
    print(f"Loaded {len(documents)} documents (pages) from PDF\n")

    # Adjust parameters for small datasets
    n_docs = len(documents)
    umap_neighbors = min(15, max(3, n_docs - 5))  # Reduce for small datasets
    hdbscan_min_size = min(10, max(2, n_docs // 5))  # Scale down for small datasets

    config = PipelineConfig(
        language=language,
        topic_modeling=TopicModelingConfig(
            min_topic_size=5,
            nr_topics=None,  # Auto-detect
            umap_n_neighbors=umap_neighbors,  # Adaptive
            hdbscan_min_cluster_size=hdbscan_min_size
        ),
        clustering=ClusteringConfig(
            n_clusters=min(5, max(2, n_docs // 6)),  # Adaptive cluster count
        )
    )

    pipeline = TextMiningPipeline(config)

    # Run full pipeline
    results = pipeline.run_full_pipeline(documents)

    # Generate and print report
    print(pipeline.generate_report())

    # Print summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    summary = pipeline.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Export results
    output_dir = os.path.join(os.path.dirname(__file__), 'output_from_pdf')
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'pdf_analysis_results.json')
    txt_path = os.path.join(output_dir, 'pdf_analysis_report.txt')

    pipeline.export_results(json_path, format='json')
    pipeline.export_results(txt_path, format='txt')

    print(f"\nResults saved to {output_dir}/")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Text Mining and Bias Detection Pipeline - PDF Analysis"
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

    # Example 1: Run full pipeline with PDF data
    pipeline = main(args.pdf_file, language=args.language)
