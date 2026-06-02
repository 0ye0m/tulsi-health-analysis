import streamlit as st
import io
import requests
from PIL import Image
from datetime import datetime
from core.analysis import analyze_leaf  # not used here, but needed for state
from core.classification import classify_leaf
from core.insights import get_rulebased_insights, get_groq_insights
from visualization.charts import generate_all_vis


def render_image_input(state):
    """Render the left column (col1) – upload and buttons. Modifies st.session_state directly."""
    st.markdown('<div class="section-title">📥 Upload Leaf Image</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="warn-box" style="margin-bottom:0.8rem">'
        "💡 <b>Tips for best accuracy:</b><br>"
        "• Plain background (white/dark/solid colour)<br>"
        "• Good lighting — no harsh shadows on leaf<br>"
        "• Leaf fills most of the frame<br>"
        "• Single leaf, no overlapping</div>",
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Input:",
        ["📁 File Upload", "📷 Camera", "🔗 URL"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if st.session_state.get("prev_input_method") != input_method:
        if st.session_state.get("prev_input_method") is not None:
            # reset everything on method change
            for k in ("metrics", "classification", "insights", "vis_cache", "error_msg"):
                st.session_state[k] = None
            st.session_state["img_bytes"] = None
            st.session_state["img_hash"] = None
            st.session_state["phase"] = "idle"
        st.session_state["prev_input_method"] = input_method

    if "File" in input_method:
        upl = st.file_uploader(
            "Upload tulsi leaf image",
            type=["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
            label_visibility="collapsed",
        )
        if upl:
            _set_image(upl.read(), upl.name)
    elif "Camera" in input_method:
        cam = st.camera_input("Take a photo")
        if cam:
            _set_image(cam.read(), "camera_capture.jpg")
    else:
        url_v = st.text_input("Paste image URL:", placeholder="https://example.com/tulsi.jpg")
        if st.button("Load from URL", key="load_url"):
            if url_v.strip():
                with st.spinner("Downloading..."):
                    try:
                        r = requests.get(url_v.strip(), timeout=15,
                                         headers={"User-Agent": "Mozilla/5.0"})
                        r.raise_for_status()
                        raw = r.content
                        sigs = [b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"BM"]
                        if not any(raw[:len(s)] == s for s in sigs) or len(raw) < 100:
                            raise ValueError("URL did not return a valid image.")
                        _set_image(raw, url_v.strip().split("/")[-1][:50])
                    except Exception as e:
                        st.error(f"Failed to load image: {e}")
            else:
                st.warning("Please enter a URL first")

    # Display image if available
    pil_img = None
    if st.session_state.get("img_bytes"):
        try:
            pil_img = Image.open(io.BytesIO(st.session_state["img_bytes"]))
        except Exception:
            st.session_state["img_bytes"] = None
            st.error("Not a valid image.")

    if pil_img:
        st.image(pil_img, caption="Loaded Image", use_container_width=True)
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Image Info</div>'
            f'<div class="metric-value" style="font-size:1rem">{pil_img.size[0]} × {pil_img.size[1]} px</div>'
            f'<div class="metric-sub">Mode: {pil_img.mode} | Source: {st.session_state.get("img_source","?")}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True, key="analyze_btn"):
            # reset previous results
            for k in ("metrics", "classification", "insights", "vis_cache", "error_msg"):
                st.session_state[k] = None
            st.session_state["phase"] = "running"
            st.rerun()
        if st.button("🗑️ Clear & Reset", use_container_width=True, key="clear_btn"):
            st.session_state["img_bytes"] = None
            st.session_state["img_hash"] = None
            for k in ("metrics", "classification", "insights", "vis_cache", "error_msg"):
                st.session_state[k] = None
            st.session_state["phase"] = "idle"
            st.rerun()
    else:
        st.markdown(
            """<div style="text-align:center;padding:3.5rem 2rem;color:#2d5a3a">
              <div style="font-size:3.5rem;margin-bottom:1rem">🌿</div>
              <div style="font-size:1.2rem;color:#4a8a58;font-family:Georgia,serif">Upload a Tulsi Leaf Image</div>
              <div style="font-size:0.85rem;margin-top:0.5rem;color:#2d5a3a">Supported: JPG, PNG, TIFF, BMP, WebP</div>
            </div>""",
            unsafe_allow_html=True,
        )
    return pil_img


def _set_image(raw_bytes: bytes, source_name: str):
    h = hash(raw_bytes)
    if st.session_state.get("img_hash") != h:
        # reset all
        for k in ("metrics", "classification", "insights", "vis_cache", "error_msg"):
            st.session_state[k] = None
        st.session_state["phase"] = "idle"
        st.session_state["img_bytes"] = raw_bytes
        st.session_state["img_source"] = source_name
        st.session_state["img_hash"] = h