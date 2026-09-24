import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from utils.bias_analysis import run_keyword_bias_detection, SUPPORTED_LANGUAGES

st.set_page_config(page_title="AI PDF Bias Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI PDF Bias Analysis Interface")
st.markdown("Upload a document to run keyword-based bias detection on each page.")

# Configuration controls in sidebar
st.sidebar.header("Configuration")
selected_language = st.sidebar.selectbox("Language", SUPPORTED_LANGUAGES, index=0)

# File Uploader
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

def run_action_1(pdf_file, language: str):
    """Action 1: Calls the bias analysis utility and renders UI components."""
    results = run_keyword_bias_detection(pdf_file, language=language)
    summary = results["summary"]
    page_details = results["pages"]

    st.success("Analysis Complete!")

    # Top Level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Pages", summary["total_pages"])
    m2.metric("Biased Pages", summary["biased_pages"])
    m3.metric("Neutral Pages", summary["neutral_pages"])
    m4.metric("Overall Bias Rate", f"{summary['bias_rate']:.1%}")

    st.divider()

    # Detailed Per-Page View
    st.subheader("Page-by-Page Breakdown")
    for page in page_details:
        header_status = "⚠️ BIAS DETECTED" if page["is_biased"] else "✅ Neutral"

        with st.expander(f"Page {page['page_number']} — {header_status} (Severity: {page['severity'].upper()})"):
            c1, c2 = st.columns([1, 1])

            with c1:
                st.markdown("**Gender Bias Assessment**")
                gb = page["gender_bias"]
                st.write(f"- Direction: `{gb.get('bias_direction', 'N/A')}`")
                st.write(f"- Male Keywords: `{', '.join(gb.get('male_keywords_found', [])) or 'None'}`")
                st.write(f"- Female Keywords: `{', '.join(gb.get('female_keywords_found', [])) or 'None'}`")

            with c2:
                st.markdown("**Discriminatory Language**")
                disc = page["discriminatory_language"]
                if disc:
                    for cat, keywords in disc.items():
                        st.write(f"- **{cat.capitalize()}:** {', '.join(keywords)}")
                else:
                    st.write("No discriminatory terms matched.")

            st.markdown("**Page Sample Text**")
            st.caption(page["text_snippet"])


if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚀 Run Action 1 (Bias Detection)", type="primary", use_container_width=True):
            with st.spinner("Analyzing PDF pages for bias patterns..."):
                try:
                    run_action_1(uploaded_file, selected_language)
                except Exception as e:
                    st.error(f"Error during Action 1 execution: {str(e)}")

    with col2:
        if st.button("🔍 Run Action 2", type="secondary", use_container_width=True):
            st.info("Action 2 placeholder.")
