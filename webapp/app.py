import base64
import html
import io
import json
import os
import sys
from datetime import datetime

# Ensure project root directory is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image, ImageChops
from analysis.llm_bias_detector import DEFAULT_MODEL as LLM_MODEL, LLMBiasDetector
from analysis.ml_bias_detector import MLBiasDetector
from utils.bias_analysis import (
    run_keyword_bias_detection,
    run_llm_bias_detection,
    run_ml_bias_detection,
)
from utils.report import build_report, finding_level, report_to_markdown

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


def get_llm_detector(language: str):
    return LLMBiasDetector(language=language)


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


SEVERITY_COLORS = {
    "high": "#f6c1c1", "medium": "#fbe0b5", "low": "#fff3b0",
}
FEEDBACK_PATH = os.path.join(PROJECT_ROOT, "data", "feedback", "feedback.jsonl")


def highlight_text(text: str, findings: list) -> str:
    """HTML of *text* with each finding wrapped in a coloured <mark> and a tooltip."""
    out, pos = [], 0
    for f in sorted(findings, key=lambda f: (f["start"], -f["end"])):
        if f["start"] < pos:
            continue  # overlaps a span already highlighted
        tip = f"{f['subtype']}: {f['explanation']}"
        if f.get("suggestion"):
            tip += f" | Suggestion: {f['suggestion']}"
        out.append(html.escape(text[pos:f["start"]]))
        out.append(
            f'<mark title="{html.escape(tip, quote=True)}" style="background:{SEVERITY_COLORS[finding_level(f["score"])]};'
            f'padding:0 2px;border-radius:3px;">{html.escape(text[f["start"]:f["end"]])}</mark>'
        )
        pos = f["end"]
    out.append(html.escape(text[pos:]))
    return '<div style="line-height:1.7;font-size:0.92rem;">' + "".join(out) + "</div>"


def save_feedback(page: dict, finding: dict, label: int):
    """Append a reviewer judgement to data/feedback in gold-set format (ROADMAP P4.3)."""
    sentence = finding.get("sentence") or page["full_text"][max(0, finding["start"] - 120):finding["end"] + 120]
    record = {
        "id": f"fb-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "language": page["language"],
        "split": "dev",
        "text": sentence,
        "label": label,
        "category": finding["subtype"] if label else "none",
        "hard_negative": not label,
        "source": "feedback-webapp",
        "flagged_by": finding["type"],
        "match": finding["match"],
    }
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_summary_metrics(summary: dict):
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Detected Language", summary["detected_language"].capitalize())
    m2.metric("Pages Analysed", summary["total_pages"])
    m3.metric("Biased Pages", summary["biased_pages"])
    m4.metric("Neutral Pages", summary["neutral_pages"])
    m5.metric("Overall Bias Rate", f"{summary['bias_rate']:.1%}")
    notes = []
    if len(summary.get("page_languages", {})) > 1:
        notes.append("Mixed languages: " + ", ".join(f"{n} {lang}" for lang, n in summary["page_languages"].items())
                     + " pages, each analysed in its own language.")
    if summary.get("references_from_page"):
        notes.append(f"References from page {summary['references_from_page']} onwards were not analysed.")
    if summary.get("ocr_pages"):
        notes.append(f"Text of pages {summary['ocr_pages']} was read with OCR and may contain recognition errors.")
    if summary.get("empty_pages"):
        notes.append(f"Pages without extractable text (possibly scanned): {summary['empty_pages']}.")
    for note in notes:
        st.caption(note)
    if summary.get("findings_by_subtype"):
        st.caption("Findings: " + ", ".join(f"{k.replace('_', ' ')} ({v})"
                                              for k, v in sorted(summary["findings_by_subtype"].items(),
                                                                 key=lambda kv: -kv[1])))


def render_method_details(page: dict, action: str):
    if action == "ml":
        ml_info = page.get("ml_detection", {})
        cat_info = page.get("ml_categorization", {})
        st.write(f"- Fine-tuned model: `{ml_info.get('label', 'N/A')}` ({ml_info.get('confidence', 0):.2f})")
        st.write(f"- Primary bias category (zero-shot): **{cat_info.get('top_category', 'N/A')}** "
                 f"({cat_info.get('top_score', 0):.2f})")
        if page.get("signal_scores"):
            st.write("- Signals: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in page["signal_scores"].items()))
    elif action == "llm":
        llm = page.get("llm_review", {})
        if not llm.get("available"):
            st.warning("LLM unavailable for this page; keyword signals only.")
        else:
            st.write(f"- Model: `{llm.get('model')}`, {len(llm.get('findings', []))} passages flagged")
    gb = page.get("gender_bias", {})
    st.write(f"- Gender reference direction: `{gb.get('bias_direction', 'N/A')}`")


def render_results(results: dict, action: str, filename: str):
    labels = {"keyword": "Keyword", "ml": "ML", "llm": "LLM review"}
    st.success(f"{labels[action]} analysis complete! Language: **{results['summary']['detected_language'].upper()}**")
    render_summary_metrics(results["summary"])
    report_md = report_to_markdown(build_report(results, action, filename))
    stem = os.path.splitext(filename)[0]
    d1, d2, _ = st.columns([1, 1, 2])
    with d1:
        st.download_button(
            "Download report (Markdown)",
            data=report_md,
            file_name=f"{stem} - {action}_bias_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download results (JSON)",
            data=json.dumps(results, indent=2, ensure_ascii=False),
            file_name=f"{stem} - {action}_bias_results.json",
            mime="application/json",
            use_container_width=True,
        )
    with st.expander("Recommendation report (all findings grouped by type)"):
        st.markdown(report_md)
    st.divider()

    st.subheader("Page Breakdown")
    show_neutral = st.toggle("Show neutral pages", value=False)
    for page in results["pages"]:
        if not page["is_biased"] and not page["findings"] and not show_neutral:
            continue
        status = "⚠️ BIAS DETECTED" if page["is_biased"] else "✅ Neutral"
        with st.expander(
            f"Page {page['page_number']} — {status} (Severity: {page['severity'].upper()}, "
            f"score {page['overall_bias_score']:.2f}, {len(page['findings'])} findings)"
        ):
            render_method_details(page, action)
            if page["findings"]:
                st.markdown("**Text with highlighted findings** (hover for explanation and suggestion)")
                st.markdown(highlight_text(page["full_text"], page["findings"]), unsafe_allow_html=True)
                st.markdown("**Findings and suggestions**")
                for i, f in enumerate(page["findings"]):
                    c1, c2, c3 = st.columns([6, 1, 1])
                    with c1:
                        st.markdown(
                            f"`{f['subtype'].replace('_', ' ')}` **{html.escape(f['match'])}**: {f['explanation']}"
                            + (f"  \n➜ *{html.escape(f['suggestion'])}*" if f.get("suggestion") else "")
                        )
                    key = f"{action}-{page['page_number']}-{i}"
                    with c2:
                        if st.button("👍", key=f"ok-{key}", help="Correct finding: save as a labelled example"):
                            save_feedback(page, f, 1)
                            st.toast("Saved as a correct finding.")
                    with c3:
                        if st.button("👎", key=f"fp-{key}", help="False positive: save as a labelled example"):
                            save_feedback(page, f, 0)
                            st.toast("Saved as a false positive.")
            else:
                st.caption(f"Text snippet: {page['text_snippet']}")


# Handle Upload & Actions
if uploaded_file is not None:
    st.info(f"**File:** {uploaded_file.name} | **Size:** {uploaded_file.size / 1024:.2f} KB")

    # Left-aligned button columns (~25% width each)
    btn_col1, btn_col2, btn_col3, _ = st.columns([1, 1, 1, 1])

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

    with btn_col3:
        llm_ready = LLMBiasDetector().is_available()
        if st.button("💬 LLM review (local, slow)", use_container_width=True, disabled=not llm_ready,
                     help=None if llm_ready else f"Start Ollama and run: ollama pull {LLM_MODEL}"):
            with st.spinner("Reviewing each page with the local LLM (about a minute per page on CPU)..."):
                try:
                    st.session_state.analysis_results = run_llm_bias_detection(uploaded_file, get_llm_detector)
                    st.session_state.active_action = "llm"
                except Exception as e:
                    st.error(f"Error executing LLM review: {str(e)}")

    # Full-Width Output View
    if st.session_state.analysis_results is not None:
        st.write("")
        render_results(st.session_state.analysis_results, st.session_state.active_action, uploaded_file.name)


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
