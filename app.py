import streamlit as st
import io
from datetime import datetime
from PIL import Image
from config import GROQ_DEFAULT_KEY, PHASE_IDLE, PHASE_RUNNING, PHASE_DONE, PHASE_ERROR
from ui.styles import CSS
from ui.image_input import render_image_input
from ui.results import render_results
from core.analysis import analyze_leaf
from core.classification import classify_leaf
from core.insights import get_rulebased_insights, get_groq_insights
from visualization.charts import generate_all_vis

# ═══════════════════════════════════════════════════════════════════════════════
# State machine helpers
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    for k, v in {
        "phase": PHASE_IDLE,
        "img_bytes": None,
        "img_source": "",
        "img_hash": None,
        "metrics": None,
        "classification": None,
        "insights": None,
        "vis_cache": None,
        "error_msg": "",
        "prev_input_method": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def main():
    st.set_page_config(
        page_title="Tulsi Leaf Health Analyzer",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    # Hero header
    st.markdown(
        """<div class="hero-header">
          <h1>&#127807; Tulsi Leaf Health Analyzer</h1>
          <p>AI-Powered Phytopathological Classification &amp; Medical Insights</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        patient_name = st.text_input("Patient / User Name", value="Anonymous")
        sample_id = st.text_input(
            "Sample ID", value=f"TUL-{datetime.now().strftime('%Y%m%d%H%M')}"
        )
        use_case = st.selectbox(
            "Medical Application",
            [
                "General Herbal Quality Control",
                "Ayurvedic Medicine Preparation",
                "Antimicrobial Research",
                "Adaptogenic / Stress Relief Formulation",
                "Anti-inflammatory Drug Screening",
                "Respiratory Therapeutics",
                "Immunomodulatory Research",
                "Phytochemical Extraction",
                "Cosmetic / Dermatological Use",
                "Veterinary Herbal Medicine",
            ],
        )
        # Offline mode toggle (always False by default, kept from original)
        use_offline = False
        groq_key = GROQ_DEFAULT_KEY  # could be made configurable

        st.markdown("---")
        st.markdown("### 📊 Thresholds (Leaf-Only)")
        from config import THRESHOLDS, CLASSIFICATION_WEIGHTS
        for key, info in THRESHOLDS.items():
            weight = CLASSIFICATION_WEIGHTS.get(key, 1)
            ref = (
                f"≤ {info['max']}"
                if key == "yellow_brown_ratio"
                else f"{info['min']} – {info['max']} {info['unit']}"
            )
            st.markdown(
                f"<div style='color:#6ab87a;font-size:0.77rem;margin:3px 0'>"
                f"<b>{info['label']}</b> <span style='color:#fbbf24;font-size:0.65rem'>(x{weight})</span><br>"
                f"<span style='color:#4a8a58'>{ref}</span></div>",
                unsafe_allow_html=True,
            )

    # Main layout: two columns
    col1, col2 = st.columns([1, 1.6], gap="large")

    with col1:
        pil_img = render_image_input(None)  # argument not used, we keep for consistency

    # Analysis pipeline (outside columns)
    if st.session_state["phase"] == PHASE_RUNNING and pil_img is not None:
        try:
            prog = st.progress(0, text="Step 1/4: Segmenting leaf from background...")
            raw_result = analyze_leaf(pil_img)
            metrics = raw_result["metrics"]

            prog.progress(25, text="Step 2/4: Running classification model...")
            classification = classify_leaf(metrics)

            prog.progress(50, text="Step 3/4: Generating visualizations...")
            vis_cache = generate_all_vis(raw_result, metrics, classification)

            prog.progress(75, text="Step 4/4: Generating AI insights...")
            insights = (
                get_rulebased_insights(metrics, classification, use_case)
                if use_offline
                else get_groq_insights(metrics, classification, use_case, groq_key)
            )

            prog.progress(100, text="✅ Analysis complete!")
            st.session_state["metrics"] = metrics
            st.session_state["classification"] = classification
            st.session_state["vis_cache"] = vis_cache
            st.session_state["insights"] = insights
            st.session_state["phase"] = PHASE_DONE

        except Exception as exc:
            st.session_state["error_msg"] = str(exc)
            st.session_state["phase"] = PHASE_ERROR

        prog.empty()
        st.rerun()

    # Results column
    with col2:
        phase = st.session_state["phase"]

        if phase == PHASE_IDLE or pil_img is None:
            st.markdown(
                """<div style="text-align:center;padding:5rem 2rem;color:#2d5a3a">
                  <div style="font-size:3rem;margin-bottom:1rem">📊</div>
                  <div style="font-size:1.2rem;color:#4a8a58;font-family:Georgia,serif">Results Will Appear Here</div>
                  <div style="font-size:0.85rem;margin-top:0.5rem;color:#2d5a3a">
                    Upload an image and click <b>Analyze Leaf</b></div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif phase == PHASE_ERROR:
            err = st.session_state.get("error_msg", "Unknown error")
            st.error(f"⚠️ Analysis Error: {err}")
            st.markdown(
                '<div class="insight-box" style="border-left-color:#f59e0b">'
                "Click <b>Analyze Leaf</b> to retry. Enable Offline Mode for persistent errors. "
                "Check that the image clearly shows a tulsi leaf.</div>",
                unsafe_allow_html=True,
            )
        elif phase == PHASE_RUNNING:
            st.info("🔄 Analysis in progress...")
        elif phase == PHASE_DONE:
            metrics = st.session_state["metrics"]
            classification = st.session_state["classification"]
            insights = st.session_state["insights"]
            vis_cache = st.session_state["vis_cache"] or {}

            if not (metrics and classification and insights):
                st.warning("Results incomplete. Click Analyze Leaf again.")
                return

            coverage = metrics.get("_leaf_coverage", 1.0)
            seg_method = metrics.get("_seg_method", "")
            render_results(
                metrics, classification, insights, vis_cache, pil_img,
                use_case, patient_name, sample_id,
                coverage, seg_method, use_offline,
            )

    # Debug expander
    with st.expander("🔧 Debug Info", expanded=False):
        st.json({
            "phase": st.session_state.get("phase"),
            "has_img": st.session_state.get("img_bytes") is not None,
            "has_metrics": st.session_state.get("metrics") is not None,
            "has_classification": st.session_state.get("classification") is not None,
            "has_vis": st.session_state.get("vis_cache") is not None,
            "error": st.session_state.get("error_msg", ""),
            "leaf_coverage": (
                st.session_state.get("metrics", {}).get("_leaf_coverage", "N/A")
                if st.session_state.get("metrics") else "N/A"
            ),
            "seg_method": (
                st.session_state.get("metrics", {}).get("_seg_method", "N/A")
                if st.session_state.get("metrics") else "N/A"
            ),
        })


if __name__ == "__main__":
    main()