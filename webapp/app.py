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

# Custom CSS for custom button appearance (White background, blue outline, custom sizing)
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #FFFFFF !important;
        color: #1E88E5 !important;
        border: 2px solid #1E88E5 !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background-color: #1E88E5 !important;
        color: #FFFFFF !important;
        border-color: #1E88E5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Cache the ML Model into memory
@st.cache_resource
def get_ml_detector(language: str):
    return MLBiasDetector(device="cpu", language=language)


st.title("📄 AI PDF Bias Analysis Interface")
st.markdown("Upload a PDF document and select an analysis method.")

# Sidebar Configuration
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


def render_keyword_results(results: dict):
    """Full-width view for Keyword-Based analysis."""
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

            st.caption(f"Text snippet: {page['text_snippet']}")


def render_ml_results(results: dict):
    """Simplified, full-width view for ML-Based analysis matching keyword format."""
    st.success("ML Bias Analysis Complete!")
    render_summary_metrics(results["summary"])
    st.divider()

    st.subheader("Page Breakdown (ML Analysis)")
    for page in results["pages"]:
        status = "⚠️ BIAS DETECTED" if page["is_biased"] else "✅ Neutral"
        with st.expander(f"Page {page['page_number']} — {status} (Severity: {page['severity'].upper()})"):
            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**ML Model Classification**")
                ml_info = page.get("ml_detection", {})
                cat_info = page.get("ml_categorization", {})

                st.write(f"- Model Flag: `{ml_info.get('label', 'N/A')}`")
                st.write(f"- Primary Bias Category: **{cat_info.get('top_category', 'N/A')}**")

            with c2:
                st.markdown("**Matched Keywords & Gender Signals**")
                gb = page.get("gender_bias", {})
                st.write(f"- Direction: `{gb.get('bias_direction', 'N/A')}`")
                st.write(f"- Keywords Found: `{', '.join(gb.get('male_keywords_found', []) + gb.get('female_keywords_found', [])) or 'None'}`")

            st.caption(f"Text snippet: {page['text_snippet']}")


# Handle upload and actions
if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")

    # Layout: Using 4 columns so buttons occupy ~25% width each, aligned left
    btn_col1, btn_col2, _, _ = st.columns([1, 1, 1, 1])

    # Session state initialization to hold results for full-width view
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "active_action" not in st.session_state:
        st.session_state.active_action = None

    with btn_col1:
        if st.button("🚀 Action 1: Keyword", use_container_width=True):
            with st.spinner("Running Keyword Detection..."):
                try:
                    st.session_state.analysis_results = run_keyword_bias_detection(uploaded_file, selected_language)
                    st.session_state.active_action = "keyword"
                except Exception as e:
                    st.error(f"Error executing Action 1: {str(e)}")

    with btn_col2:
        if st.button("🧠 Action 2: ML Model", use_container_width=True):
            with st.spinner("Running ML Inferences..."):
                try:
                    detector = get_ml_detector(selected_language)
                    st.session_state.analysis_results = run_ml_bias_detection(uploaded_file, detector, selected_language)
                    st.session_state.active_action = "ml"
                except Exception as e:
                    st.error(f"Error executing Action 2: {str(e)}")

    # Full-Width Output View (Rendered outside column wrappers)
    if st.session_state.analysis_results is not None:
        st.write("")  # Spacing
        if st.session_state.active_action == "keyword":
            render_keyword_results(st.session_state.analysis_results)
        elif st.session_state.active_action == "ml":
            render_ml_results(st.session_state.analysis_results)
