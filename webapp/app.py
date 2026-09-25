import os
import sys

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from analysis.ml_bias_detector import MLBiasDetector
from utils.bias_analysis import (
    run_keyword_bias_detection,
    run_ml_bias_detection,
    SUPPORTED_LANGUAGES,
)

st.set_page_config(page_title="AI PDF Bias Analyzer", page_icon="📄", layout="wide")


# Cache the ML Model so it loads into memory once and persists across user interactions
@st.cache_resource
def get_ml_detector(language: str):
    return MLBiasDetector(device="cpu", language=language)


st.title("PDF Bias Analysis Interface")
st.markdown("Upload a PDF document and select either **Keyword Analysis** or **ML Model Analysis**.")

# Configuration sidebar
st.sidebar.header("Configuration")
selected_language = st.sidebar.selectbox("Language", SUPPORTED_LANGUAGES, index=0)

# File Uploader
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])


def render_summary_metrics(summary: dict):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Pages", summary["total_pages"])
    m2.metric("Biased Pages", summary["biased_pages"])
    m3.metric("Neutral Pages", summary["neutral_pages"])
    m4.metric("Overall Bias Rate", f"{summary['bias_rate']:.1%}")


# Action 1: Keyword Detection UI
def handle_keyword_based_detection(pdf_file, language: str):
    results = run_keyword_bias_detection(pdf_file, language=language)
    st.success("Keyword Bias Analysis Complete!")
    render_summary_metrics(results["summary"])
    st.divider()

    st.subheader("Page Breakdown (Keyword Matching)")
    for page in results["pages"]:
        status = "⚠️ BIAS DETECTED" if page["is_biased"] else "✅ Neutral"
        with st.expander(f"Page {page['page_number']} — {status} (Severity: {page['severity'].upper()})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Gender Bias Details**")
                gb = page["gender_bias"]
                st.write(f"- Direction: `{gb.get('bias_direction', 'N/A')}`")
                st.write(f"- Male Keywords: `{', '.join(gb.get('male_keywords_found', [])) or 'None'}`")
                st.write(f"- Female Keywords: `{', '.join(gb.get('female_keywords_found', [])) or 'None'}`")

            with c2:
                st.markdown("**Discriminatory Terms**")
                disc = page["discriminatory_language"]
                if disc:
                    for cat, keywords in disc.items():
                        st.write(f"- **{cat.capitalize()}:** {', '.join(keywords)}")
                else:
                    st.write("None detected.")


# Action 2: ML Detection UI
def handle_ml_based_detection(pdf_file, language: str):
    # Retrieve cached ML model
    with st.spinner("Initializing/Retrieving ML Models in memory..."):
        ml_detector = get_ml_detector(language)

    results = run_ml_bias_detection(pdf_file, ml_detector, language=language)
    st.success("ML Bias Analysis Complete!")
    render_summary_metrics(results["summary"])
    st.divider()

    st.subheader("Page Breakdown (ML Classification & Zero-Shot)")
    for page in results["pages"]:
        status = "⚠️ BIAS DETECTED" if page["is_biased"] else "✅ Neutral"
        with st.expander(f"Page {page['page_number']} — {status} (Severity: {page['severity'].upper()})"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**ML Fine-Tuned Model**")
                ml_info = page["ml_detection"]
                st.write(f"- Classification: `{ml_info.get('label', 'N/A')}`")
                st.write(f"- Confidence: `{ml_info.get('confidence', 0.0):.2%}`")

                st.markdown("**Zero-Shot Categorization**")
                cat_info = page["ml_categorization"]
                st.write(f"- Top Category: **{cat_info.get('top_category', 'N/A')}** (`{cat_info.get('top_score', 0.0):.2%}`)")

            with c2:
                st.markdown("**Category Confidence Distribution**")
                cat_info = page["ml_categorization"]
                if "labels" in cat_info and "scores" in cat_info:
                    # Render category scores as progress bars
                    for label, score in zip(cat_info["labels"], cat_info["scores"]):
                        st.caption(f"{label}: {score:.1%}")
                        st.progress(float(score))

            st.caption(f"Text snippet: {page['text_snippet']}")


# Main App Layout
if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Keyword Analysis", type="primary", use_container_width=True):
            with st.spinner("Running Keyword Detection..."):
                try:
                    handle_keyword_based_detection(uploaded_file, selected_language)
                except Exception as e:
                    st.error(f"Error executing Action 1: {str(e)}")

    with col2:
        if st.button("ML Model Analysis", type="secondary", use_container_width=True):
            with st.spinner("Running Machine Learning Inferences..."):
                try:
                    handle_ml_based_detection(uploaded_file, selected_language)
                except Exception as e:
                    st.error(f"Error executing Action 2: {str(e)}")
