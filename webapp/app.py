import base64
import io
import os
import sys

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image, ImageChops
from analysis.ml_bias_detector import MLBiasDetector
from utils.bias_analysis import (
    run_keyword_bias_detection,
    run_ml_bias_detection,
)

st.set_page_config(page_title="PDF Bias Analyzer", page_icon="📄", layout="wide")

BRAND_PURPLE = "#5c068c"
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")


@st.cache_data
def load_logo_base64(filename: str, trim: bool = False) -> str:
    """Return a logo as base64 PNG, optionally trimming its surrounding white or transparent margin."""
    image = Image.open(os.path.join(IMAGES_DIR, filename))
    if trim and image.mode == "RGBA":
        bbox = image.getchannel("A").getbbox()
        if bbox:
            image = image.crop(bbox)
    elif trim:
        image = image.convert("RGB")
        background = Image.new("RGB", image.size, (255, 255, 255))
        # Threshold the difference so JPEG noise near-white is not treated as content
        diff = ImageChops.difference(image, background).convert("L").point(lambda px: 255 if px > 24 else 0)
        bbox = diff.getbbox()
        if bbox:
            image = image.crop(bbox)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


st.markdown(
    f"""
    <style>
    /* 1/5 left margin, 3/5 content, 1/5 right margin */
    [data-testid="stMainBlockContainer"], .block-container {{
        max-width: 60% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        padding-top: 2.5rem !important;
        padding-bottom: 1rem !important;
        box-sizing: border-box;
        /* Fill the viewport so the footer can be pushed to the bottom */
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }}
    /* The page-level wrappers above the footer grow to fill the remaining
       height, and the footer's element is pushed to the bottom. The footer's
       own markdown wrappers are left untouched so its layout is unchanged. */
    [data-testid="stMainBlockContainer"] div:has(.app-footer):not([data-testid="stElementContainer"]):not([data-testid="stElementContainer"] *) {{
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
    }}
    [data-testid="stElementContainer"]:has(.app-footer) {{
        margin-top: auto;
    }}
    @media (max-width: 1200px) {{
        [data-testid="stMainBlockContainer"], .block-container {{
            max-width: 85% !important;
        }}
    }}
    @media (max-width: 640px) {{
        [data-testid="stMainBlockContainer"], .block-container {{
            max-width: 100% !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }}
    }}

    /* Header */
    .app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 2rem;
        padding: 0.5rem 0 1.25rem 0;
        border-bottom: 3px solid {BRAND_PURPLE};
    }}
    .app-header img {{
        height: 64px;
        width: auto;
        max-width: 45%;
        object-fit: contain;
    }}
    .app-title {{
        margin: 1.75rem 0 0.25rem 0;
        color: {BRAND_PURPLE};
        font-weight: 700;
        font-size: 2.1rem;
        line-height: 1.2;
    }}
    .app-subtitle {{
        color: #555555;
        margin-bottom: 1.5rem;
    }}

    /* Accents */
    h2, h3 {{
        color: {BRAND_PURPLE} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {BRAND_PURPLE};
    }}
    [data-testid="stMetric"] {{
        border-left: 3px solid {BRAND_PURPLE};
        padding-left: 0.75rem;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        border: 1px dashed {BRAND_PURPLE};
        background-color: rgba(92, 6, 140, 0.04);
    }}
    [data-testid="stExpander"] details {{
        border-left: 3px solid rgba(92, 6, 140, 0.35);
    }}

    /* Buttons */
    div.stButton > button {{
        background-color: #FFFFFF !important;
        color: {BRAND_PURPLE} !important;
        border: 2px solid {BRAND_PURPLE} !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }}
    div.stButton > button:hover {{
        background-color: {BRAND_PURPLE} !important;
        color: #FFFFFF !important;
        border-color: {BRAND_PURPLE} !important;
    }}

    /* Footer */
    .app-footer {{
        margin-top: 4rem;
        padding: 1rem 0 0.75rem 0;
        border-top: 3px solid {BRAND_PURPLE};
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
    }}
    .app-footer img {{
        height: 50px;
        width: auto;
        object-fit: contain;
        flex-shrink: 0;
    }}
    .app-footer-text {{
        text-align: center;
        font-size: 0.8rem;
        color: #555555;
        line-height: 1.5;
    }}
    .app-footer-text strong {{
        color: {BRAND_PURPLE};
        letter-spacing: 0.03em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_analysis_state():
    st.session_state.analysis_results = None
    st.session_state.active_action = None


# Session state initialization
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "active_action" not in st.session_state:
    st.session_state.active_action = None


# Cache the ML Model into RAM memory per language
@st.cache_resource
def get_ml_detector(language: str):
    return MLBiasDetector(device="cpu", language=language)


st.markdown(
    f"""
    <div class="app-header">
        <img src="data:image/png;base64,{load_logo_base64('iuem-logo.jpg', trim=True)}" alt="IUEM logo">
        <img src="data:image/png;base64,{load_logo_base64('ull-logo.jpg', trim=True)}" alt="Universidad de La Laguna logo">
    </div>
    <div class="app-title">PDF Bias Analysis</div>
    <div class="app-subtitle">Upload a PDF document and select an analysis method. Language is automatically detected.</div>
    """,
    unsafe_allow_html=True,
)

# File Uploader
uploaded_file = st.file_uploader(
    "Upload PDF Document",
    type=["pdf"],
    on_change=clear_analysis_state,
)


def render_summary_metrics(summary: dict):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Detected Language", summary["detected_language"].capitalize())
    m2.metric("Total Pages", summary["total_pages"])
    m3.metric("Biased Pages", summary["biased_pages"])
    m4.metric("Neutral Pages", summary["neutral_pages"])
    m5.metric("Overall Bias Rate", f"{summary['bias_rate']:.1%}")


def render_keyword_results(results: dict):
    """Full-width view for Keyword-Based analysis."""
    st.success(f"Keyword Bias Analysis Complete! Language: **{results['summary']['detected_language'].upper()}**")
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
    """Simplified full-width view for ML-Based analysis."""
    st.success(f"ML Bias Analysis Complete! Language: **{results['summary']['detected_language'].upper()}**")
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


# Handle Upload & Actions
if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")

    # Left-aligned button columns (~25% width each)
    btn_col1, btn_col2, _, _ = st.columns([1, 1, 1, 1])

    with btn_col1:
        if st.button("🗝️ Keyword based analysis", use_container_width=True):
            with st.spinner("Detecting language and running Keyword Analysis..."):
                try:
                    st.session_state.analysis_results = run_keyword_bias_detection(uploaded_file)
                    st.session_state.active_action = "keyword"
                except Exception as e:
                    st.error(f"Error executing Keyword analysis: {str(e)}")

    with btn_col2:
        if st.button("🧠 ML Model based analysis", use_container_width=True):
            with st.spinner("Detecting language and executing ML Model Inferences..."):
                try:
                    st.session_state.analysis_results = run_ml_bias_detection(uploaded_file, get_ml_detector)
                    st.session_state.active_action = "ml"
                except Exception as e:
                    st.error(f"Error executing ML analysis: {str(e)}")

    # Full-Width Output View
    if st.session_state.analysis_results is not None:
        st.write("")
        if st.session_state.active_action == "keyword":
            render_keyword_results(st.session_state.analysis_results)
        elif st.session_state.active_action == "ml":
            render_ml_results(st.session_state.analysis_results)


st.markdown(
    f"""
    <div class="app-footer">
        <img src="data:image/png;base64,{load_logo_base64('iuem-logo-footer.png')}" alt="IUEM logo">
        <div class="app-footer-text">
            Built on the <strong>GENDERVISION-AI</strong> project, conducted at the<br>
            Instituto Universitario de Estudios de las Mujeres (IUEM), Universidad de La Laguna.
        </div>
        <img src="data:image/png;base64,{load_logo_base64('ull-logo-footer.png', trim=True)}" alt="Universidad de La Laguna logo">
    </div>
    """,
    unsafe_allow_html=True,
)
