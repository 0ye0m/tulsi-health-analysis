
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import io
import json
import requests
from datetime import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Tulsi Leaf Health Analyzer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; }
.stApp { background: linear-gradient(135deg,#0a1a0d 0%,#0f2016 50%,#0a1a0d 100%); min-height:100vh; }
.hero-header { text-align:center; padding:2rem 1rem 1.5rem;
  background:linear-gradient(180deg,rgba(34,197,94,0.08) 0%,transparent 100%);
  border-bottom:1px solid rgba(34,197,94,0.15); margin-bottom:2rem; }
.hero-header h1 { font-family:Georgia,serif; font-size:2.6rem; font-weight:700;
  color:#e8f5e9; letter-spacing:-0.5px; margin:0; }
.hero-header p { color:#6ab87a; font-size:1rem; margin-top:0.4rem; font-weight:300; }
.vbadge { display:inline-block; background:rgba(34,197,94,0.15);
  border:1px solid rgba(34,197,94,0.3); color:#4ade80; font-size:0.72rem; font-weight:600;
  padding:0.15rem 0.6rem; border-radius:999px; letter-spacing:1px; text-transform:uppercase; margin-top:0.6rem; }
.metric-card { background:rgba(255,255,255,0.04); border:1px solid rgba(34,197,94,0.2);
  border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:0.8rem; }
.metric-label { color:#6ab87a; font-size:0.72rem; text-transform:uppercase;
  letter-spacing:1.5px; font-weight:600; margin-bottom:0.3rem; }
.metric-value { color:#e8f5e9; font-size:1.55rem; font-weight:600; }
.metric-sub { color:#4a8a58; font-size:0.78rem; margin-top:0.2rem; }
.result-healthy { background:linear-gradient(135deg,rgba(34,197,94,0.18),rgba(21,128,61,0.12));
  border:2px solid #22c55e; border-radius:16px; padding:2rem; text-align:center; }
.result-unhealthy { background:linear-gradient(135deg,rgba(239,68,68,0.18),rgba(153,27,27,0.12));
  border:2px solid #ef4444; border-radius:16px; padding:2rem; text-align:center; }
.result-title { font-family:Georgia,serif; font-size:2.2rem; font-weight:700; margin:0.5rem 0; }
.insight-box { background:rgba(255,255,255,0.03); border-left:3px solid #22c55e;
  border-radius:0 10px 10px 0; padding:1.1rem 1.4rem; margin:0.8rem 0;
  color:#c8e6c9; font-size:0.93rem; line-height:1.7; }
.section-title { font-family:Georgia,serif; color:#a5d6a7; font-size:1.25rem; font-weight:700;
  margin:1.4rem 0 0.7rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(34,197,94,0.2); }
.param-row { display:flex; justify-content:space-between; align-items:center;
  padding:0.5rem 0; border-bottom:1px solid rgba(255,255,255,0.05); color:#c8e6c9; font-size:0.88rem; }
.param-pass { color:#4ade80; font-weight:600; }
.param-fail { color:#f87171; font-weight:600; }
.badge { display:inline-block; padding:0.2rem 0.7rem; border-radius:999px;
  font-size:0.72rem; font-weight:600; text-transform:uppercase; letter-spacing:0.8px; }
div[data-testid="stSidebar"] { background:#07130a !important; border-right:1px solid rgba(34,197,94,0.15) !important; }
.stButton > button { background:linear-gradient(135deg,#16a34a,#15803d); color:white;
  border:none; border-radius:8px; font-weight:600; padding:0.6rem 2rem; width:100%;
  font-size:0.95rem; transition:all 0.2s; }
.stButton > button:hover { background:linear-gradient(135deg,#15803d,#166534);
  transform:translateY(-1px); box-shadow:0 4px 20px rgba(34,197,94,0.3); }
.stProgress > div > div { background:#22c55e !important; }
.warn-box { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.3);
  border-radius:10px; padding:1rem 1.2rem; color:#fde68a; font-size:0.88rem; margin:0.8rem 0; }
.pipeline-step { background:rgba(255,255,255,0.03); border:1px solid rgba(34,197,94,0.15);
  border-radius:12px; padding:1.4rem; margin:0.8rem 0; }
.pipeline-num { background:linear-gradient(135deg,#22c55e,#15803d); color:white;
  width:28px; height:28px; border-radius:50%; display:inline-flex; align-items:center;
  justify-content:center; font-weight:700; font-size:0.82rem; margin-right:0.7rem; }
.chip { display:inline-block; background:rgba(255,255,255,0.06);
  border:1px solid rgba(34,197,94,0.2); border-radius:7px; padding:0.35rem 0.7rem;
  font-size:0.78rem; color:#c8e6c9; margin:0.2rem; }
.chip b { color:#e8f5e9; }
</style>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — thresholds calibrated for LEAF-ONLY pixels
# ═══════════════════════════════════════════════════════════════════════════════
# These thresholds are stricter/more specific than v5/v6 because we now
# analyse only confirmed leaf tissue pixels (no background interference).
THRESHOLDS = {
    "green_dominance": {
        "min": 0.50,
        "max": 1.0,
        "unit": "",
        "label": "Green Dominance (leaf pixels)",
    },
    "hsv_hue_green_ratio": {
        "min": 0.40,
        "max": 1.0,
        "unit": "",
        "label": "Green Hue Coverage (leaf)",
    },
    "yellow_brown_ratio": {
        "min": 0.0,
        "max": 0.12,
        "unit": "",
        "label": "Yellow/Brown Spot Ratio (leaf)",
    },
    "green_channel_ratio": {
        "min": 0.36,
        "max": 1.0,
        "unit": "",
        "label": "Green Channel Ratio (leaf)",
    },
    "hsv_saturation": {
        "min": 50,
        "max": 255,
        "unit": "pts",
        "label": "HSV Saturation (leaf)",
    },
    "mean_intensity": {
        "min": 45,
        "max": 200,
        "unit": "pts",
        "label": "Mean Intensity (leaf)",
    },
    "std_dev_intensity": {
        "min": 8,
        "max": 90,
        "unit": "pts",
        "label": "Texture StdDev (leaf)",
    },
    "edge_density": {
        "min": 0.03,
        "max": 0.40,
        "unit": "",
        "label": "Edge / Vein Density (leaf)",
    },
}

CLASSIFICATION_WEIGHTS = {
    "green_dominance": 4,  # Most critical — chlorophyll presence
    "yellow_brown_ratio": 4,  # Most critical — disease/lesion indicator
    "hsv_hue_green_ratio": 3,  # High — spectral greenness
    "green_channel_ratio": 2,  # Medium — colour balance
    "hsv_saturation": 2,  # Medium — tissue vitality
    "mean_intensity": 1,
    "std_dev_intensity": 1,
    "edge_density": 1,
}

GROQ_DEFAULT_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

PHASE_IDLE = "idle"
PHASE_RUNNING = "running"
PHASE_DONE = "done"
PHASE_ERROR = "error"

INSIGHT_KEYS = [
    "clinical_summary",
    "pathological_indicators",
    "medical_relevance",
    "recommendations",
    "phytochemical_note",
    "quality_grade",
    "safety_flag",
    "detailed_pathology",
    "treatment_protocol",
    "environmental_factors",
    "pharmacopoeial_compliance",
]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def _save_fig(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f1a12", dpi=110)
    plt.close(fig)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# LEAF SEGMENTATION  ←  THE KEY ACCURACY FIX
# ═══════════════════════════════════════════════════════════════════════════════
def segment_leaf(img_np: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Robustly isolate the leaf from the background using a 3-stage pipeline:

    Stage 1 — Color thresholding in HSV space
        Tulsi leaves: hue 25-100 (yellowish-green to dark green, OpenCV 0-180),
        plus brown/necrotic regions hue 0-25.  Two masks are combined to catch
        healthy AND diseased leaf tissue.

    Stage 2 — Morphological cleanup
        Close small holes, open noise, dilate to recover leaf edges trimmed by
        conservative thresholding.

    Stage 3 — Keep largest connected component
        Removes stray background blobs that passed color thresholding.

    Falls back to full-image mask if <3% of pixels are segmented.

    Returns:
        leaf_mask  : uint8 (255 = leaf, 0 = background)
        coverage   : fraction of image that is leaf (0-1)
        method_used: description string for UI
    """
    h, w = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── Stage 1: Multi-band HSV thresholding ──────────────────────────────────
    # Band A: Green / yellow-green healthy tissue (OpenCV H: 25-100)
    mask_a = cv2.inRange(img_hsv, np.array([25, 30, 25]), np.array([100, 255, 240]))

    # Band B: Dark olive / very dark green (often under-saturated in shadow)
    mask_b = cv2.inRange(img_hsv, np.array([20, 15, 15]), np.array([100, 255, 100]))

    # Band C: Brown / yellow-brown DISEASED tissue — still leaf!
    # Hue 0-25, low-mid saturation, mid value
    mask_c = cv2.inRange(img_hsv, np.array([0, 35, 40]), np.array([25, 220, 220]))

    # Band D: Yellow (senescent tissue), hue 20-35, high saturation
    mask_d = cv2.inRange(img_hsv, np.array([18, 60, 60]), np.array([35, 255, 255]))

    combined = cv2.bitwise_or(
        cv2.bitwise_or(mask_a, mask_b), cv2.bitwise_or(mask_c, mask_d)
    )

    # ── Stage 2: Morphological cleanup ───────────────────────────────────────
    short = min(h, w)
    ksize = max(7, short // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Close to fill interior holes (veins, stems)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    # Open to remove isolated noise pixels
    ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ksmall, iterations=2)
    # Dilate to recover edges lost by conservative thresholding
    kdilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize // 2 + 3, ksize // 2 + 3)
    )
    combined = cv2.dilate(combined, kdilate, iterations=2)

    # ── Stage 3: Keep largest connected component ─────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    if num_labels <= 1:
        # Nothing found — fall back to full image
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
        coverage = 1.0
        method = "full-image fallback (no leaf detected by color)"
    else:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask_raw = (labels == largest).astype(np.uint8) * 255
        # Fill convex hull to recover internal holes (dark centre of leaf etc.)
        contours, _ = cv2.findContours(
            leaf_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        leaf_mask = np.zeros((h, w), dtype=np.uint8)
        if contours:
            hull = cv2.convexHull(max(contours, key=cv2.contourArea))
            cv2.fillPoly(leaf_mask, [hull], 255)
        else:
            leaf_mask = leaf_mask_raw
        coverage = float(np.sum(leaf_mask > 0)) / (h * w)
        method = "HSV multi-band + convex hull"

    # If coverage is suspiciously low, fall back
    if coverage < 0.03:
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
        coverage = 1.0
        method = "full-image fallback (coverage < 3%)"

    return leaf_mask, coverage, method


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE ANALYSIS  — metrics on leaf pixels ONLY
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_leaf(image: Image.Image) -> dict:
    """
    Full analysis pipeline:
    1. Segment leaf from background
    2. Compute all metrics ONLY on leaf pixels
    3. Return metrics + intermediates for visualisation
    """
    img_pil = image.convert("RGB")
    img_np = np.array(img_pil)
    if img_np.shape[0] < 8 or img_np.shape[1] < 8:
        raise ValueError(
            f"Image too small: {img_np.shape[1]}x{img_np.shape[0]}px — need at least 8x8."
        )

    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Segment leaf ──────────────────────────────────────────────────────────
    leaf_mask, coverage, seg_method = segment_leaf(img_np)
    leaf_bool = leaf_mask > 0  # boolean mask for indexing

    # ── Extract LEAF pixels only ──────────────────────────────────────────────
    R_full = img_np[:, :, 0].astype(float)
    G_full = img_np[:, :, 1].astype(float)
    B_full = img_np[:, :, 2].astype(float)

    # Apply mask — work on 1D arrays of leaf pixels
    R = R_full[leaf_bool]
    G = G_full[leaf_bool]
    B = B_full[leaf_bool]
    gray_leaf = gray[leaf_bool].astype(float)
    hue_leaf = img_hsv[:, :, 0][leaf_bool].astype(float)
    sat_leaf = img_hsv[:, :, 1][leaf_bool].astype(float)

    # ── Compute metrics ───────────────────────────────────────────────────────
    total = R + G + B + 1e-6

    # Greenness metrics
    green_dom_px = (G > R) & (G > B)
    green_dominance = (
        float(np.sum(green_dom_px) / len(green_dom_px)) if len(green_dom_px) > 0 else 0
    )
    green_ratio = float(np.mean(G / total))

    # HSV metrics on leaf
    hsv_saturation = float(np.mean(sat_leaf))
    green_hue_px = (hue_leaf >= 30) & (hue_leaf <= 90)  # OpenCV H: 30-90 = green/lime
    hsv_hue_green_ratio = (
        float(np.sum(green_hue_px) / len(hue_leaf)) if len(hue_leaf) > 0 else 0
    )

    # Disease detection — yellow/brown on leaf tissue only
    # Using 2D masks restricted to leaf area
    hue_2d = img_hsv[:, :, 0].astype(float)
    sat_2d = img_hsv[:, :, 1].astype(float)
    val_2d = img_hsv[:, :, 2].astype(float)

    # Brown necrotic spots: hue 0-18, sat 35-180, val 40-170
    brown_2d = (
        (hue_2d >= 0)
        & (hue_2d <= 18)
        & (sat_2d >= 35)
        & (sat_2d <= 200)
        & (val_2d >= 40)
        & (val_2d <= 180)
    )
    # Yellow spots (chlorosis, early disease): hue 15-30, sat > 50
    yellow_2d = (hue_2d >= 15) & (hue_2d <= 30) & (sat_2d >= 50)

    # Restrict to leaf pixels ONLY
    brown_leaf_mask = brown_2d & leaf_bool
    yellow_leaf_mask = yellow_2d & leaf_bool
    disease_mask_2d = brown_leaf_mask | yellow_leaf_mask

    leaf_pixel_count = float(np.sum(leaf_bool))
    yellow_brown_ratio = (
        float(np.sum(disease_mask_2d)) / leaf_pixel_count if leaf_pixel_count > 0 else 0
    )

    # Texture and structural
    mean_intensity = float(np.mean(gray_leaf))
    std_dev = float(np.std(gray_leaf))
    edges_full = cv2.Canny(gray, 50, 150)
    edges_leaf = edges_full.copy()
    edges_leaf[~leaf_bool] = 0
    edge_density = (
        float(np.sum(edges_leaf > 0)) / leaf_pixel_count if leaf_pixel_count > 0 else 0
    )

    metrics = {
        "mean_intensity": round(mean_intensity, 2),
        "green_channel_ratio": round(green_ratio, 4),
        "green_dominance": round(green_dominance, 4),
        "hsv_saturation": round(hsv_saturation, 2),
        "hsv_hue_green_ratio": round(hsv_hue_green_ratio, 4),
        "yellow_brown_ratio": round(yellow_brown_ratio, 4),
        "std_dev_intensity": round(std_dev, 2),
        "edge_density": round(edge_density, 4),
        "r_mean": round(float(np.mean(R)), 2),
        "g_mean": round(float(np.mean(G)), 2),
        "b_mean": round(float(np.mean(B)), 2),
        "r_std": round(float(np.std(R)), 2),
        "g_std": round(float(np.std(G)), 2),
        "b_std": round(float(np.std(B)), 2),
        "_leaf_coverage": round(coverage, 4),
        "_seg_method": seg_method,
        "_leaf_px_count": int(leaf_pixel_count),
    }

    intermediates = {
        "img_rgb": img_np,
        "img_hsv": img_hsv,
        "gray": gray,
        "R_full": R_full,
        "G_full": G_full,
        "B_full": B_full,
        "leaf_mask": leaf_mask,
        "leaf_bool": leaf_bool,
        "brown_mask": brown_leaf_mask,
        "yellow_mask": yellow_leaf_mask,
        "disease_mask": disease_mask_2d,
        "green_dom_mask": np.zeros_like(leaf_bool, dtype=bool),  # computed below
        "edges_leaf": edges_leaf,
        "edges_full": edges_full,
    }
    # Green-dom full 2D mask (for visualisation)
    gd_2d = (G_full > R_full) & (G_full > B_full) & leaf_bool
    intermediates["green_dom_mask"] = gd_2d

    return {"metrics": metrics, "intermediates": intermediates}


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def classify_leaf(metrics: dict) -> dict:
    T = THRESHOLDS
    W = CLASSIFICATION_WEIGHTS
    param_results = {}
    param_details = []

    for key in T:
        val = metrics.get(key)
        if val is None:
            continue
        lo, hi = T[key]["min"], T[key]["max"]
        passed = lo <= val <= hi
        param_results[key] = passed
        param_details.append(
            {
                "key": key,
                "label": T[key]["label"],
                "value": val,
                "min": lo,
                "max": hi,
                "unit": T[key]["unit"],
                "passed": passed,
                "weight": W.get(key, 1),
            }
        )

    passed_count = sum(param_results.values())
    total = len(param_results)
    weighted_score = sum(W.get(k, 1) * int(v) for k, v in param_results.items())
    max_weighted = sum(W.get(k, 1) for k in param_results)
    confidence_raw = weighted_score / max_weighted if max_weighted > 0 else 0

    # Hard penalty gates — if leaf is obviously diseased, cap confidence
    yb = metrics.get("yellow_brown_ratio", 0)
    gd = metrics.get("green_dominance", 1)
    gs = metrics.get("hsv_hue_green_ratio", 1)
    if yb > 0.30:
        confidence_raw = min(confidence_raw, 0.30)
    elif yb > 0.18:
        confidence_raw = min(confidence_raw, 0.50)
    if gd < 0.20:
        confidence_raw = min(confidence_raw, 0.25)
    elif gd < 0.35:
        confidence_raw = min(confidence_raw, 0.48)
    if gs < 0.25:
        confidence_raw = min(confidence_raw, 0.40)

    is_healthy = confidence_raw >= 0.55

    if is_healthy:
        confidence = round(min(confidence_raw * 100, 99.5), 1)
    else:
        confidence = round(max(min((1.0 - confidence_raw) * 100, 99.5), 15.0), 1)

    severity = "N/A"
    if not is_healthy:
        if confidence_raw < 0.28:
            severity = "Severe"
        elif confidence_raw < 0.42:
            severity = "Moderate"
        else:
            severity = "Mild"

    return {
        "status": "Healthy" if is_healthy else "Unhealthy",
        "is_healthy": is_healthy,
        "confidence": confidence,
        "severity": severity,
        "param_results": param_results,
        "param_details": param_details,
        "weighted_score": weighted_score,
        "max_weighted": max_weighted,
        "confidence_raw": round(confidence_raw, 4),
        "passed_count": passed_count,
        "total_params": total,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
def _darkbg_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs, facecolor="#0f1a12")
    return fig


def make_segmentation_vis(inter: dict, metrics: dict) -> bytes:
    """Show original + leaf mask + masked image side by side."""
    img_rgb = inter["img_rgb"]
    leaf_mask = inter["leaf_mask"]
    leaf_bool = inter["leaf_bool"]

    # Masked image — background replaced with dark colour
    masked = img_rgb.copy()
    masked[~leaf_bool] = [15, 30, 15]

    # Mask overlay
    overlay = img_rgb.copy()
    green_overlay = np.zeros_like(img_rgb)
    green_overlay[:, :, 1] = 100
    alpha = 0.35
    overlay[leaf_bool] = np.clip(
        overlay[leaf_bool].astype(float) * (1 - alpha)
        + green_overlay[leaf_bool].astype(float) * alpha,
        0,
        255,
    ).astype(np.uint8)
    overlay[~leaf_bool] = (overlay[~leaf_bool].astype(float) * 0.3).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        "Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Leaf Segmentation  (Coverage: {metrics['_leaf_coverage']:.1%})",
        color="#4ade80",
        fontsize=10,
        pad=8,
    )
    axes[2].imshow(masked)
    axes[2].set_title(
        "Leaf Pixels Only\n(Metrics computed here)", color="#fbbf24", fontsize=10, pad=8
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle(
        "Leaf Segmentation — Background Excluded from Analysis",
        color="#e8f5e9",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_rgb_bands(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"]
    R, G, B = inter["R_full"], inter["G_full"], inter["B_full"]
    leaf_bool = inter["leaf_bool"]
    # Gray out background
    R_vis = R.copy()
    R_vis[~leaf_bool] = 15
    G_vis = G.copy()
    G_vis[~leaf_bool] = 15
    B_vis = B.copy()
    B_vis[~leaf_bool] = 15
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), facecolor="#0f1a12")
    axes = axes.flatten()
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        "Original RGB", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].imshow(R_vis.astype(np.uint8), cmap="Reds", vmin=0, vmax=255)
    axes[1].set_title(
        f"Red Channel  (Leaf Mean:{metrics['r_mean']:.0f})",
        color="#f87171",
        fontsize=10,
        pad=8,
    )
    axes[2].imshow(G_vis.astype(np.uint8), cmap="Greens", vmin=0, vmax=255)
    axes[2].set_title(
        f"Green Channel (Leaf Mean:{metrics['g_mean']:.0f})",
        color="#4ade80",
        fontsize=10,
        pad=8,
    )
    axes[3].imshow(B_vis.astype(np.uint8), cmap="Blues", vmin=0, vmax=255)
    axes[3].set_title(
        f"Blue Channel  (Leaf Mean:{metrics['b_mean']:.0f})",
        color="#60a5fa",
        fontsize=10,
        pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle(
        "RGB Channel Decomposition (Leaf Region Only)",
        color="#e8f5e9",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig)


def make_disease_spot_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"].copy()
    disease_mask = inter["disease_mask"]
    leaf_bool = inter["leaf_bool"]
    overlay = img_rgb.copy()
    # Red tint disease spots
    idx = disease_mask & leaf_bool
    overlay[idx, 0] = np.minimum(overlay[idx, 0].astype(int) + 120, 255).astype(
        np.uint8
    )
    overlay[idx, 1] = np.maximum(overlay[idx, 1].astype(int) - 60, 0).astype(np.uint8)
    overlay[idx, 2] = np.maximum(overlay[idx, 2].astype(int) - 60, 0).astype(np.uint8)
    # Dim background
    overlay[~leaf_bool] = (overlay[~leaf_bool].astype(float) * 0.25).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        "Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].imshow(overlay)
    pct = metrics["yellow_brown_ratio"] * 100
    col = "#f87171" if pct > 12 else "#fbbf24" if pct > 6 else "#4ade80"
    axes[1].set_title(
        f"Disease Spots on Leaf  (Affected: {pct:.2f}%)\n[threshold: <12% of leaf]",
        color=col,
        fontsize=10,
        pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle(
        "Yellow / Brown Lesion Mapping (Leaf Pixels Only)",
        color="#e8f5e9",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_green_dominance_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"].copy()
    gd_mask = inter["green_dom_mask"]
    leaf_bool = inter["leaf_bool"]
    overlay = img_rgb.copy()
    overlay[gd_mask, 1] = np.minimum(overlay[gd_mask, 1].astype(int) + 55, 255).astype(
        np.uint8
    )
    non_gd_leaf = leaf_bool & ~gd_mask
    overlay[non_gd_leaf] = (overlay[non_gd_leaf].astype(float) * 0.55).astype(np.uint8)
    overlay[~leaf_bool] = (overlay[~leaf_bool].astype(float) * 0.15).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        "Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Green Dominance (G>R AND G>B)  {metrics['green_dominance']:.1%} of leaf",
        color="#4ade80",
        fontsize=10,
        pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle(
        "Chlorophyll Distribution — Green Dominant Pixels on Leaf",
        color="#e8f5e9",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_edge_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"]
    edges_leaf = inter["edges_leaf"]
    edge_col = np.zeros((*edges_leaf.shape, 3), dtype=np.uint8)
    edge_col[edges_leaf > 0] = [34, 197, 94]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title(
        "Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8
    )
    axes[1].imshow(edge_col)
    axes[1].set_title(
        f"Canny Edges on Leaf  (Density:{metrics['edge_density']:.2%})",
        color="#4ade80",
        fontsize=10,
        pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle(
        "Structural Edge Analysis (Leaf Region Only)",
        color="#e8f5e9",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_color_histogram(inter: dict, metrics: dict) -> bytes:
    leaf_bool = inter["leaf_bool"]
    R = inter["R_full"][leaf_bool].flatten().astype(np.uint8)
    G = inter["G_full"][leaf_bool].flatten().astype(np.uint8)
    B = inter["B_full"][leaf_bool].flatten().astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0f1a12")
    ax.set_facecolor("#0a1a0d")
    ax.hist(
        R,
        bins=256,
        range=(0, 256),
        color="#ef4444",
        alpha=0.5,
        label=f"Red  (Mean:{np.mean(R):.0f})",
        density=True,
    )
    ax.hist(
        G,
        bins=256,
        range=(0, 256),
        color="#22c55e",
        alpha=0.5,
        label=f"Green(Mean:{np.mean(G):.0f})",
        density=True,
    )
    ax.hist(
        B,
        bins=256,
        range=(0, 256),
        color="#3b82f6",
        alpha=0.5,
        label=f"Blue (Mean:{np.mean(B):.0f})",
        density=True,
    )
    ax.axvline(
        np.mean(G),
        color="#4ade80",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Green mean",
    )
    ax.set_xlabel("Pixel Intensity (0-255)", color="#6ab87a", fontsize=10)
    ax.set_ylabel("Normalised Frequency (Leaf Pixels)", color="#6ab87a", fontsize=10)
    ax.set_title(
        "RGB Histogram — Leaf Pixels Only",
        color="#e8f5e9",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.legend(
        loc="upper right",
        facecolor="#0f1a12",
        edgecolor="#1a3a22",
        labelcolor="#c8e6c9",
        fontsize=9,
    )
    ax.tick_params(colors="#4a8a58")
    ax.spines[:].set_edgecolor("#1a3a22")
    ax.set_xlim(0, 255)
    plt.tight_layout()
    return _save_fig(fig)


def make_radar_chart(metrics: dict) -> bytes:
    cats = [
        "Intensity",
        "Green\nRatio",
        "Green\nDom.",
        "Saturation",
        "Green\nHue",
        "Low\nDisease",
        "Texture",
        "Edges",
    ]
    T = THRESHOLDS
    keys = list(T.keys())

    def norm(key, val):
        lo, hi = T[key]["min"], T[key]["max"]
        if hi <= lo:
            return 0.5
        if key == "yellow_brown_ratio":
            return max(0.0, min(1.0, 1.0 - min(val / hi, 1.0)))
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return max(0.0, min(1.0, 1.0 - abs((val - mid) / half)))

    values = [norm(k, metrics.get(k, 0)) for k in keys] + [
        norm(keys[0], metrics.get(keys[0], 0))
    ]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0.0]
    fig, ax = plt.subplots(
        figsize=(5.5, 5.5), subplot_kw=dict(polar=True), facecolor="#0f1a12"
    )
    ax.set_facecolor("#0f1a12")
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color="#6ab87a", fontsize=8.5)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#2d5a3a", fontsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3a22")
    ax.plot(angles, values, color="#22c55e", linewidth=2.2)
    ax.fill(angles, values, color="#22c55e", alpha=0.22)
    ax.plot(
        angles,
        [0.5] * (N + 1),
        "--",
        color="#f59e0b",
        linewidth=1,
        alpha=0.6,
        label="Threshold",
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        labelcolor="#f59e0b",
        facecolor="#0f1a12",
        edgecolor="#1a3a22",
        fontsize=8,
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_channel_bar(metrics: dict) -> bytes:
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), facecolor="#0f1a12")
    labels = ["Green\nDominance", "Disease\nResistance", "Green\nHue Coverage"]
    norms = [
        metrics.get("green_dominance", 0),
        1.0 - min(metrics.get("yellow_brown_ratio", 0) / 0.12, 1.0),
        metrics.get("hsv_hue_green_ratio", 0),
    ]
    thr = [0.50, 0.58, 0.40]  # healthy threshold for each bar
    for ax, val, lbl, thr_v in zip(axes, norms, labels, thr):
        col = "#22c55e" if val >= thr_v else "#ef4444"
        ax.set_facecolor("#0f1a12")
        ax.barh([0], [1.0], color="#1a3a22", height=0.5, alpha=0.4, edgecolor="none")
        ax.barh([0], [val], color=col, height=0.5, alpha=0.85, edgecolor="none")
        ax.axvline(thr_v, color="#f59e0b", linewidth=2, linestyle="--", alpha=0.8)
        ax.set_xlim(0, 1.05)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"], color="#4a8a58", fontsize=7)
        ax.set_title(lbl, color="#6ab87a", fontsize=9, pad=6)
        ax.spines[:].set_visible(False)
        ax.text(
            min(val + 0.04, 0.97),
            0,
            f"{val*100:.0f}%",
            va="center",
            color=col,
            fontsize=9,
            fontweight="bold",
        )
    plt.suptitle(
        "Key Health Indicators (Leaf Pixels Only)", color="#a5d6a7", fontsize=10, y=1.05
    )
    plt.tight_layout()
    return _save_fig(fig)


def make_classification_breakdown(classification: dict) -> bytes:
    details = classification.get("param_details", [])
    if not details:
        return b""
    fig, ax = plt.subplots(
        figsize=(9, max(4, len(details) * 0.65)), facecolor="#0f1a12"
    )
    ax.set_facecolor("#0a1a0d")
    labels = [d["label"] for d in details]
    weights = [d["weight"] for d in details]
    passed = [d["passed"] for d in details]
    y_pos = range(len(details))
    cols = ["#22c55e" if p else "#ef4444" for p in passed]
    bar_vals = [w if p else 0 for w, p in zip(weights, passed)]
    ax.barh(y_pos, weights, color="#1a3a22", height=0.6, alpha=0.4, edgecolor="none")
    ax.barh(y_pos, bar_vals, color=cols, height=0.6, alpha=0.8, edgecolor="none")
    for i, (w, p) in enumerate(zip(weights, passed)):
        ax.text(
            max(weights) + 0.2,
            i,
            f"{'✓' if p else '✗'} (w={w})",
            va="center",
            color=cols[i],
            fontsize=9,
            fontweight="bold",
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#c8e6c9", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Weight", color="#6ab87a", fontsize=10)
    ax.set_title(
        f"Classification Breakdown  |  Score:{classification['weighted_score']}/{classification['max_weighted']}  "
        f"({classification['confidence_raw']*100:.1f}% raw  →  {'✓ HEALTHY' if classification['is_healthy'] else '✗ UNHEALTHY'})",
        color="#e8f5e9",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(colors="#4a8a58")
    ax.spines[:].set_edgecolor("#1a3a22")
    plt.tight_layout()
    return _save_fig(fig)


def generate_all_vis(raw: dict, metrics: dict, classification: dict) -> dict:
    inter = raw["intermediates"]
    return {
        "segmentation": make_segmentation_vis(inter, metrics),
        "rgb_bands": make_rgb_bands(inter, metrics),
        "disease": make_disease_spot_map(inter, metrics),
        "green_map": make_green_dominance_map(inter, metrics),
        "edge": make_edge_map(inter, metrics),
        "histogram": make_color_histogram(inter, metrics),
        "radar": make_radar_chart(metrics),
        "bars": make_channel_bar(metrics),
        "classification_breakdown": make_classification_breakdown(classification),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHT ENGINE — RULE-BASED
# ═══════════════════════════════════════════════════════════════════════════════
def get_rulebased_insights(metrics: dict, classification: dict, use_case: str) -> dict:
    healthy = classification["is_healthy"]
    severity = classification["severity"]
    passed = classification["param_results"]
    conf = classification["confidence"]
    passed_n = classification["passed_count"]
    total_n = classification["total_params"]
    coverage = metrics.get("_leaf_coverage", 1.0)
    seg_note = (
        f" Note: leaf segmentation coverage was {coverage:.1%}."
        if coverage < 0.15
        else ""
    )

    indicators = []
    if passed.get("green_dominance"):
        indicators.append(
            f"Strong green dominance ({metrics['green_dominance']:.1%} of leaf pixels have G>R AND G>B) confirms healthy chlorophyll distribution and active photosynthesis."
        )
    else:
        indicators.append(
            f"Reduced green dominance ({metrics['green_dominance']:.1%} of leaf pixels) — only on leaf tissue — indicates chlorophyll degradation, senescence, or pathogen-induced chlorosis."
        )
    if passed.get("yellow_brown_ratio"):
        indicators.append(
            f"Minimal yellow/brown lesions ({metrics['yellow_brown_ratio']:.2%} of leaf area) — well below the 12% threshold. No significant fungal or bacterial disease symptoms detected."
        )
    else:
        indicators.append(
            f"Elevated lesion ratio ({metrics['yellow_brown_ratio']:.2%} of leaf area exceeds 12% threshold) on confirmed leaf pixels — indicative of fungal leaf spot (Cercospora ocimicola), bacterial blight, or nutrient deficiency."
        )
    if passed.get("hsv_hue_green_ratio"):
        indicators.append(
            f"High green hue coverage ({metrics['hsv_hue_green_ratio']:.1%} of leaf in H:30-90 range) confirms dominant chlorophyll-a and chlorophyll-b spectral signature."
        )
    else:
        indicators.append(
            f"Low green hue coverage ({metrics['hsv_hue_green_ratio']:.1%}) on leaf tissue — pigment shift toward yellow/brown spectrum indicates carotenoid dominance during chlorophyll degradation."
        )
    if passed.get("hsv_saturation"):
        indicators.append(
            f"Adequate colour saturation (mean HSV-S: {metrics['hsv_saturation']:.0f}) on leaf pixels suggests well-hydrated, metabolically active tissue with intact cellular turgor."
        )
    else:
        indicators.append(
            f"Low saturation (HSV-S: {metrics['hsv_saturation']:.0f}) on leaf tissue may indicate dehydration, wilting, or early necrosis reducing tissue vitality."
        )
    if passed.get("edge_density"):
        indicators.append(
            f"Normal vein/edge density ({metrics['edge_density']:.2%}) on leaf region indicates intact vascular architecture and lamina structure."
        )
    else:
        indicators.append(
            f"Abnormal edge density ({metrics['edge_density']:.2%}) on leaf may reflect structural damage, tissue maceration, or advanced lesion coalescence."
        )
    if passed.get("mean_intensity"):
        indicators.append(
            f"Mean leaf pixel intensity ({metrics['mean_intensity']:.0f}) is within the expected range for healthy Ocimum sanctum tissue under standard imaging conditions."
        )
    else:
        indicators.append(
            f"Leaf mean intensity ({metrics['mean_intensity']:.0f}) is outside expected range — may reflect unusual colouration, shadowing, or image quality issues on the leaf surface."
        )

    if healthy:
        summary = (
            f"The tulsi (Ocimum sanctum) leaf presents as botanically healthy with {passed_n}/{total_n} parameters within normal ranges at {conf}% confidence. "
            f"Analysis performed on confirmed leaf tissue ({coverage:.1%} of image area). "
            f"Green dominance ({metrics['green_dominance']:.1%}), disease spot ratio ({metrics['yellow_brown_ratio']:.2%}), "
            f"and saturation ({metrics['hsv_saturation']:.0f}) all meet pharmacopoeial reference criteria for medicinal-grade Ocimum sanctum, suitable for {use_case.lower()}."
            + seg_note
        )
        med_rel = (
            f"For {use_case}, healthy leaf tissue with strong green dominance correlates with optimal concentrations of eugenol (0.5-1.1% dry weight), "
            f"rosmarinic acid (0.3-0.8%), and ursolic acid (0.05-0.3%). "
            f"Suitable for direct use in herbal preparations, hydro-distillation, tinctures, or standardised extraction without concern for degraded phytochemical profiles."
        )
        recs = [
            "Harvest during morning hours (6-10 AM) for maximum essential oil content",
            "Store at 15-25°C, dark, well-ventilated, <60% relative humidity",
            "For essential oil extraction, process within 2-4 hours of harvest using steam distillation at 100°C for 3-4 hours",
            "Air-dry at 35-40°C in single layer for 3-5 days; avoid direct sunlight to preserve eugenol",
            "Document harvest date, GPS coordinates, and growing conditions for GAP/pharmacopoeial traceability",
            f"For {use_case}: expected fresh-to-dry ratio ~4:1; target moisture content <10% for long-term storage",
        ]
        phyto = (
            "Healthy green dominance and saturation correlate with peak biosynthesis of eugenol (0.5-1.1% dry weight, anti-microbial/analgesic), "
            "rosmarinic acid (0.3-0.8%, potent antioxidant/anti-inflammatory), and ursolic acid (0.05-0.3%, hepatoprotective triterpenoid). "
            "Specimen appears at optimal maturation stage with intact glandular trichomes for maximum phytochemical yield."
        )
        pct_p = passed_n / total_n if total_n > 0 else 0
        grade = (
            "Grade A -- All critical parameters within healthy range; medicinal-grade specimen suitable for pharmacopoeial applications."
            if pct_p >= 0.87
            else "Grade B -- Minor deviations in non-critical parameters; suitable for most standard herbal applications."
        )
        safety = "SAFE -- Specimen passes all critical health checks. No contraindications detected based on visual phytopathological analysis of leaf tissue."
        detailed_path = (
            "No significant pathological findings on leaf tissue. Coloration, turgor, and structural integrity are consistent with healthy Ocimum sanctum morphology. "
            "Vascular bundles appear intact with no evidence of vascular wilt, necrotic lesions, chlorotic halos, or mildew growth on the leaf lamina."
        )
        treatment = (
            "No treatment required. Maintain current growing conditions: 6-8 hours direct sunlight daily, water when top 1 inch of soil is dry, "
            "ambient 20-30°C, balanced NPK fertiliser (10-10-10) every 4-6 weeks during growing season."
        )
    else:
        gd_val = round(metrics.get("green_dominance", 0) * 100, 1)
        yb_val = round(metrics.get("yellow_brown_ratio", 0) * 100, 2)
        gd_note = (
            f"green dominance at {gd_val}% (need >=50%)"
            if not passed.get("green_dominance")
            else ""
        )
        yb_note = (
            f"; disease ratio {yb_val}% (need <=12%)"
            if not passed.get("yellow_brown_ratio")
            else ""
        )
        summary = (
            f"The tulsi leaf -- analysed on confirmed leaf tissue ({coverage:.1%} of image) -- shows {severity.lower()} botanical degradation. "
            f"Only {passed_n}/{total_n} parameters meet healthy thresholds ({conf}% confidence). "
            f"Key deviations: {gd_note}{yb_note}. "
            f"Phytochemical potency may be compromised for {use_case.lower()}."
            + seg_note
        )
        med_rel = (
            f"Leaf tissue degradation (not background artefact) directly correlates with reduced eugenol, rosmarinic acid, and ursolic acid. "
            f"For {use_case.lower()}, this specimen may fail to deliver expected therapeutic potency and could introduce mycotoxins from fungal colonisation of necrotic tissue. "
            f"Expert pharmacognostic review is recommended before any clinical or commercial application."
        )
        if severity in ("Severe", "Moderate"):
            recs = [
                "REJECT this specimen for medicinal or research use — confirmed leaf tissue shows elevated disease markers and potential mycotoxin contamination risk",
                "Investigate growing conditions: check soil pH (6.5-7.5), nitrogen/potassium levels, irrigation frequency, and drainage",
                "Isolate source plant to prevent spread of fungal (Cercospora, Alternaria) or bacterial (Pseudomonas, Xanthomonas) pathogens",
                "Collect a fresh sample from an asymptomatic plant; prefer upper canopy leaves with 6+ hours direct sun",
                "Apply organic fungicide — neem oil 0.3% or copper oxychloride 0.2% spray, every 7 days for 3 applications",
                "Consult a plant pathologist for professional diagnosis before resuming harvest",
            ]
        else:
            recs = [
                "Use with caution — mild degradation on leaf tissue reduces but may not eliminate therapeutic potency; non-critical applications only",
                "Trim all visibly yellow/brown margins before processing; use only the healthy green leaf portions",
                "Consider for preliminary phytochemical screening or educational demonstration rather than clinical application",
                "Monitor source plant over 7-10 days; if degradation progresses, escalate to moderate treatment protocol",
                "Consult a botanist or pharmacognosist for professional assessment",
            ]
        phyto = (
            "Disease markers on leaf tissue indicate reduced glandular trichome biosynthetic activity. Eugenol may fall below the 0.5% WHO/AYUSH pharmacopoeial minimum. "
            "Oxidative stress may alter rosmarinic acid to caffeic acid ratio, reducing anti-inflammatory efficacy. "
            "Pathogen-derived polyphenol oxidases could further degrade remaining phytochemicals if specimen is not promptly disposed of."
        )
        pct_p = passed_n / total_n if total_n > 0 else 0
        grade = (
            "Grade C -- Significant deviations on confirmed leaf tissue; mandatory expert review before any use."
            if pct_p >= 0.50
            else "Grade D -- Specimen fails majority of health parameters; not recommended for medicinal or research use."
        )
        safety = (
            "CAUTION -- Minor deviations on leaf tissue; expert review recommended before clinical application."
            if severity == "Mild"
            else "REJECT -- Significant degradation on confirmed leaf pixels; NOT recommended for therapeutic, research, or commercial use due to phytochemical compromise and mycotoxin risk."
        )
        detailed_path = (
            f"Pathological assessment on confirmed leaf tissue reveals {severity.lower()} degradation. "
            + (
                f"Chlorosis and reduced green dominance ({metrics['green_dominance']:.1%}) suggest nutrient deficiency or pathogen-induced pigment breakdown. "
                if not passed.get("green_dominance")
                else ""
            )
            + (
                f"Lesions covering {metrics['yellow_brown_ratio']*100:.1f}% of leaf area consistent with fungal leaf spot or bacterial blight. "
                if not passed.get("yellow_brown_ratio")
                else ""
            )
            + (
                "Reduced tissue saturation indicating dehydration or cellular breakdown. "
                if not passed.get("hsv_saturation")
                else ""
            )
            + "Expert phytopathological examination is warranted."
        )
        if severity in ("Severe", "Moderate"):
            treatment = (
                "Immediate treatment: (1) Isolate plant; (2) Remove and destroy affected leaves with sterilised scissors; "
                "(3) Apply neem oil 0.3% or copper oxychloride 0.2%, every 7 days x3 applications; "
                "(4) Improve air circulation and reduce overhead watering; (5) Apply balanced NPK liquid fertiliser; "
                "(6) Monitor daily for 14 days; if no improvement, submit leaf sample to plant pathology laboratory."
            )
        else:
            treatment = (
                "Mild intervention: (1) Trim affected margins with sterilised scissors; (2) Ensure 6+ hours direct sunlight; "
                "(3) Check soil moisture — tulsi prefers well-drained, not waterlogged soil; "
                "(4) Apply diluted seaweed extract as foliar spray to boost plant immunity; "
                "(5) Monitor for 7 days; escalate to moderate protocol if degradation progresses."
            )

    env_factors = (
        "Environmental factors potentially influencing leaf health (based on detected leaf-tissue deviations): "
        + (
            "Low light conditions may cause chlorophyll degradation and reduced green pigment. "
            if not passed.get("green_dominance")
            else ""
        )
        + (
            "Excessive humidity (>80%) or overhead irrigation favours fungal pathogen development on leaf surfaces. "
            if not passed.get("yellow_brown_ratio")
            else ""
        )
        + (
            "Nitrogen or micronutrient deficiency can cause leaf chlorosis and reduced saturation. "
            if not passed.get("hsv_saturation")
            else ""
        )
        + "Optimal Ocimum sanctum conditions: well-drained loamy soil (pH 6.5-7.5), 6-8 hours direct sunlight, 20-35°C, relative humidity 40-65%."
    )
    if healthy:
        pharm = (
            "Specimen meets preliminary visual criteria for pharmacopoeial compliance per WHO Good Agricultural and Collection Practices (GACP) and AYUSH guidelines. "
            "Consistent with API/BIS standards: green to dark-green colour, characteristic aromatic odour, absence of significant disease markers on leaf tissue. "
            "Full compliance requires laboratory confirmation of essential oil content (≥0.5% dry weight), heavy metal limits, and microbial count testing."
        )
    else:
        pharm = (
            f"Specimen does NOT meet preliminary pharmacopoeial compliance criteria. Confirmed leaf tissue analysis shows {severity.lower()} deviations: "
            f"disease markers at {metrics['yellow_brown_ratio']:.2%} (threshold <12%) and green dominance at {metrics['green_dominance']:.1%} (threshold ≥50%). "
            f"Would likely fail organoleptic evaluation per API/BIS/WHO monographs for Ocimum sanctum. Exclude from any pharmacopoeial supply chain."
        )

    return {
        "clinical_summary": summary,
        "pathological_indicators": indicators,
        "medical_relevance": med_rel,
        "recommendations": recs,
        "phytochemical_note": phyto,
        "quality_grade": grade,
        "safety_flag": safety,
        "detailed_pathology": detailed_path,
        "treatment_protocol": treatment,
        "environmental_factors": env_factors,
        "pharmacopoeial_compliance": pharm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# INSIGHT ENGINE — GROQ AI
# ═══════════════════════════════════════════════════════════════════════════════
def get_groq_insights(
    metrics: dict, classification: dict, use_case: str, api_key: str
) -> dict:
    if not api_key or not api_key.strip():
        return get_rulebased_insights(metrics, classification, use_case)
    coverage = metrics.get("_leaf_coverage", 1.0)
    prompt = f"""You are an expert plant pathologist and pharmacognosist specialising in Tulsi (Ocimum sanctum) leaf health.

IMPORTANT: All metrics below were computed ONLY on confirmed leaf pixels (background excluded via HSV segmentation).
Leaf coverage: {coverage:.1%} of image.

ANALYSIS RESULTS:
- Classification: {classification['status']} ({classification['confidence']}% confidence)
- Severity: {classification['severity']}
- Parameters Passed: {classification['passed_count']}/{classification['total_params']}
- Use Case: {use_case}

LEAF-TISSUE BIOMETRIC PARAMETERS (leaf pixels only):
1. Mean Intensity: {metrics.get('mean_intensity','N/A')} pts (normal: 45-200)
2. Green Channel Ratio: {metrics.get('green_channel_ratio','N/A')} (normal: >=0.36)
3. Green Dominance (G>R AND G>B): {metrics.get('green_dominance','N/A')} (normal: >=0.50)
4. HSV Saturation: {metrics.get('hsv_saturation','N/A')} pts (normal: 50-255)
5. Green Hue Coverage (H:30-90): {metrics.get('hsv_hue_green_ratio','N/A')} (normal: >=0.40)
6. Yellow/Brown Spot Ratio: {metrics.get('yellow_brown_ratio','N/A')} (normal: <=0.12)
7. Texture StdDev: {metrics.get('std_dev_intensity','N/A')} pts (normal: 8-90)
8. Edge Density: {metrics.get('edge_density','N/A')} (normal: 0.03-0.40)

PASS/FAIL: {json.dumps(classification['param_results'], indent=2)}

Return a JSON response. Try to follow the structure but it's okay if slightly different. (no markdown, no preamble):
{{
  "clinical_summary": "3-4 sentences on leaf-tissue health status for {use_case}, referencing specific metric values and that analysis excluded background",
  "pathological_indicators": ["6-8 specific indicators referencing exact metric values from leaf-tissue analysis"],
  "medical_relevance": "3-4 sentences on therapeutic implications for {use_case}, referencing eugenol, rosmarinic acid, ursolic acid",
  "recommendations": ["5-7 specific actionable recommendations"],
  "phytochemical_note": "3-4 sentences on phytochemical impact for eugenol, rosmarinic acid, ursolic acid",
  "quality_grade": "Grade A/B/C/D with justification referencing specific leaf-tissue metric values",
  "safety_flag": "SAFE / CAUTION / REJECT with 2-sentence reasoning",
  "detailed_pathology": "3-4 sentences on pathological findings from leaf tissue analysis",
  "treatment_protocol": "Numbered treatment steps",
  "environmental_factors": "3-4 sentences on likely environmental causes based on leaf-tissue deviations",
  "pharmacopoeial_compliance": "2-3 sentences on WHO GACP, AYUSH, API/BIS compliance"
}}"""
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key.strip()}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",  
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400,  
            },
            timeout=45,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.split("\n")[1:] if not l.strip().startswith("```")
            )
        try:
            parsed = json.loads(raw)
        except Exception as e:
            print("❌ RAW GROQ RESPONSE:\n", raw)
            return get_rulebased_insights(metrics, classification, use_case)

        fallback = get_rulebased_insights(metrics, classification, use_case)
        for k in INSIGHT_KEYS:
            if k not in parsed or not parsed[k]:
                parsed[k] = fallback[k]
        return parsed
    except Exception as e:
        fallback = get_rulebased_insights(metrics, classification, use_case)
        fallback["clinical_summary"] = (
            f"[AI fallback ({str(e)[:50]})] " + fallback["clinical_summary"]
        )
        return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def generate_pdf(
    image,
    metrics,
    classification,
    insights,
    use_case,
    patient_name,
    sample_id,
    vis_cache,
):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    CD = colors.HexColor("#0a1a0d")
    CW = colors.white

    def S(n, **k):
        return ParagraphStyle(n, **k)

    sT = S(
        "sT",
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=CW,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    sH = S(
        "sH",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=CD,
        spaceBefore=10,
        spaceAfter=4,
    )
    sB = S(
        "sB",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#1f2937"),
        leading=15,
        spaceAfter=4,
        alignment=TA_JUSTIFY,
    )
    sBul = S(
        "sBl",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#1f2937"),
        leading=14,
        leftIndent=12,
        spaceAfter=3,
    )
    sSm = S(
        "sSm",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=colors.HexColor("#6b7280"),
        alignment=TA_CENTER,
    )
    story = []
    ts = datetime.now()

    def HR():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbf7d0"))

    # Header
    ht = Table(
        [[Paragraph("TULSI LEAF HEALTH ANALYSIS REPORT", sT)]], colWidths=[17 * cm]
    )
    ht.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), CD),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )
    story.extend([ht, Spacer(1, 0.3 * cm)])

    meta = [
        ["Sample ID", sample_id, "Date", ts.strftime("%d %b %Y")],
        ["Patient/User", patient_name, "Time", ts.strftime("%H:%M:%S")],
        ["Application", use_case, "Analyst", "AI/Rule-Based System"],
        [
            "Status",
            Paragraph(
                f'<font color="{"#166534" if classification["is_healthy"] else "#991b1b"}"><b>{classification["status"]}</b></font>',
                sB,
            ),
            "Confidence",
            f'{classification["confidence"]}%',
        ],
        [
            "Leaf Coverage",
            f'{metrics.get("_leaf_coverage",1)*100:.1f}%',
            "Seg. Method",
            metrics.get("_seg_method", "N/A"),
        ],
    ]
    mt = Table(meta, colWidths=[3.5 * cm, 5 * cm, 3 * cm, 5 * cm])
    mt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0fdf4")),
                ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0fdf4")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1fae5")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [CW, colors.HexColor("#f9fafb")]),
            ]
        )
    )
    story.extend([mt, Spacer(1, 0.4 * cm)])

    # Banner
    b_bg = (
        colors.HexColor("#dcfce7")
        if classification["is_healthy"]
        else colors.HexColor("#fee2e2")
    )
    b_tc = (
        colors.HexColor("#166534")
        if classification["is_healthy"]
        else colors.HexColor("#991b1b")
    )
    sev = (
        ""
        if classification["is_healthy"]
        else f"  |  Severity: {classification['severity']}"
    )
    gd = insights.get("quality_grade", "N/A").split("--")[0].strip().rstrip("-")
    bd = [
        [
            Paragraph(
                f'Status: <b>{classification["status"]}</b>{sev}  |  Confidence:{classification["confidence"]}%  |  Quality:{gd}',
                ParagraphStyle(
                    "bn",
                    fontName="Helvetica-Bold",
                    fontSize=11,
                    textColor=b_tc,
                    alignment=TA_CENTER,
                ),
            )
        ]
    ]
    bt = Table(bd, colWidths=[17 * cm])
    bt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), b_bg),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    story.extend([bt, Spacer(1, 0.4 * cm)])

    # Segmentation image
    seg_data = vis_cache.get("segmentation")
    if seg_data:
        story.append(Paragraph("Leaf Segmentation (Background Excluded)", sH))
        story.append(
            Paragraph(
                f"Metrics computed on leaf tissue only ({metrics.get('_leaf_coverage',1)*100:.1f}% of image). Background excluded from all calculations.",
                sB,
            )
        )
        story.append(RLImage(io.BytesIO(seg_data), width=15 * cm))
        story.append(Spacer(1, 0.3 * cm))

    # Leaf image + radar
    img_pil = image.copy()
    img_pil.thumbnail((200, 200))
    ib = io.BytesIO()
    img_pil.save(ib, format="PNG")
    ib.seek(0)
    ri = RLImage(ib, width=5 * cm)
    rb_data = vis_cache.get("radar")
    rb = (
        RLImage(io.BytesIO(rb_data), width=5.5 * cm, height=5.5 * cm) if rb_data else ri
    )
    it = Table([[ri, rb]], colWidths=[7 * cm, 10 * cm])
    it.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.extend([it, Spacer(1, 0.3 * cm), HR()])

    # Parameters
    story.extend(
        [
            Spacer(1, 0.3 * cm),
            Paragraph("Quantitative Parameters (Leaf Tissue Only)", sH),
        ]
    )
    story.append(
        Paragraph(
            "All values below are computed exclusively on segmented leaf pixels, not background.",
            sB,
        )
    )
    pr = [["Parameter", "Measured Value", "Normal Range (leaf)", "Status"]]
    for key in THRESHOLDS:
        val = metrics.get(key)
        info = THRESHOLDS[key]
        p = classification["param_results"].get(key, False)
        if val is None:
            continue
        ref = (
            f"<= {info['max']}"
            if key == "yellow_brown_ratio"
            else f"{info['min']} - {info['max']}"
        )
        pr.append(
            [
                info["label"],
                f"{val}",
                ref,
                Paragraph(
                    f'<font color="{"#166534" if p else "#991b1b"}"><b>{"✓ Pass" if p else "✗ Fail"}</b></font>',
                    sB,
                ),
            ]
        )
    pt = Table(pr, colWidths=[6 * cm, 2.8 * cm, 4.2 * cm, 3.5 * cm])
    pt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), CD),
                ("TEXTCOLOR", (0, 0), (-1, 0), CW),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1fae5")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CW, colors.HexColor("#f9fafb")]),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.extend([pt, Spacer(1, 0.4 * cm), HR()])

    # Classification breakdown
    story.extend(
        [Spacer(1, 0.3 * cm), Paragraph("Classification Scoring Breakdown", sH)]
    )
    sp = (
        classification["weighted_score"] / classification["max_weighted"]
        if classification["max_weighted"] > 0
        else 0
    )
    story.append(
        Paragraph(
            f"Weighted Score: <b>{classification['weighted_score']}/{classification['max_weighted']}</b> ({sp*100:.1f}%) | Params Passed: <b>{classification['passed_count']}/{classification['total_params']}</b> | Threshold: 55%",
            sB,
        )
    )
    cb = vis_cache.get("classification_breakdown")
    if cb:
        story.append(RLImage(io.BytesIO(cb), width=15 * cm))
    story.extend([Spacer(1, 0.3 * cm), HR()])

    # Visualizations
    for vk, vt in [
        ("disease", "Disease Spot Mapping"),
        ("green_map", "Green Dominance Map"),
        ("edge", "Edge Analysis"),
        ("histogram", "RGB Histogram"),
    ]:
        vd = vis_cache.get(vk)
        if vd:
            story.extend(
                [
                    Spacer(1, 0.2 * cm),
                    Paragraph(f"<b>{vt}</b>", sB),
                    RLImage(io.BytesIO(vd), width=14 * cm),
                    Spacer(1, 0.2 * cm),
                ]
            )
    story.append(HR())

    # Clinical insights
    story.extend([Spacer(1, 0.3 * cm), Paragraph("Clinical Insights", sH)])
    for ttl, key in [
        ("Clinical Summary", "clinical_summary"),
        ("Detailed Pathology", "detailed_pathology"),
        ("Medical Relevance", "medical_relevance"),
        ("Phytochemical Analysis", "phytochemical_note"),
        ("Treatment Protocol", "treatment_protocol"),
        ("Environmental Factors", "environmental_factors"),
        ("Pharmacopoeial Compliance", "pharmacopoeial_compliance"),
    ]:
        story.append(Paragraph(f"<b>{ttl}</b>", sB))
        story.append(Paragraph(str(insights.get(key, "N/A")), sB))
        story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph("<b>Pathological Indicators</b>", sB))
    for i in insights.get("pathological_indicators", []):
        story.append(Paragraph(f"- {i}", sBul))
    story.extend([Spacer(1, 0.2 * cm), Paragraph("<b>Recommendations</b>", sB)])
    for r in insights.get("recommendations", []):
        story.append(Paragraph(f">> {r}", sBul))
    story.extend([Spacer(1, 0.3 * cm), HR()])

    # Safety banner
    sf = insights.get("safety_flag", "CAUTION")
    sb_c = (
        colors.HexColor("#dcfce7")
        if "SAFE" in sf
        else (
            colors.HexColor("#fef3c7")
            if "CAUTION" in sf
            else colors.HexColor("#fee2e2")
        )
    )
    sc_c = (
        colors.HexColor("#166534")
        if "SAFE" in sf
        else (
            colors.HexColor("#92400e")
            if "CAUTION" in sf
            else colors.HexColor("#991b1b")
        )
    )
    gd2 = insights.get("quality_grade", "N/A").split("--")[0].strip().rstrip("-")
    sfd = [
        [
            Paragraph(
                f"Safety: <b>{sf}</b>  |  Quality: <b>{gd2}</b>",
                ParagraphStyle(
                    "sf",
                    fontName="Helvetica-Bold",
                    fontSize=10,
                    textColor=sc_c,
                    alignment=TA_CENTER,
                ),
            )
        ]
    ]
    sft = Table(sfd, colWidths=[17 * cm])
    sft.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), sb_c),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.extend([sft, Spacer(1, 0.3 * cm), HR(), Spacer(1, 0.2 * cm)])
    story.append(
        Paragraph(
            f"Tulsi Leaf Health Analyzer(Leaf-Segmented Analysis)  |  {ts.strftime('%d %b %Y, %H:%M')}  |  For research and medical advisory use only.",
            sSm,
        )
    )
    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE HELPERS
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


def _reset_results():
    for k in ("metrics", "classification", "insights", "vis_cache", "error_msg"):
        st.session_state[k] = None
    st.session_state["error_msg"] = ""
    st.session_state["phase"] = PHASE_IDLE


def _set_image(raw_bytes: bytes, source_name: str):
    h = hash(raw_bytes)
    if st.session_state["img_hash"] != h:
        _reset_results()
        st.session_state["img_bytes"] = raw_bytes
        st.session_state["img_source"] = source_name
        st.session_state["img_hash"] = h


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    _init_state()

    st.markdown(
        """
    <div class="hero-header">
      <h1>&#127807; Tulsi Leaf Health Analyzer</h1>
      <p>AI-Powered Phytopathological Classification &amp; Medical Insights</p>
      
    </div>""",
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────────
    groq_key = GROQ_DEFAULT_KEY
    use_offline = False
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
        


        st.markdown("---")
        st.markdown("### 📊 Thresholds (Leaf-Only)")
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

    col1, col2 = st.columns([1, 1.6], gap="large")

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN
    # ══════════════════════════════════════════════════════════════════════════
    with col1:
        st.markdown(
            '<div class="section-title">📥 Upload Leaf Image</div>',
            unsafe_allow_html=True,
        )
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
        if st.session_state["prev_input_method"] != input_method:
            if st.session_state["prev_input_method"] is not None:
                _reset_results()
                st.session_state["img_bytes"] = None
                st.session_state["img_hash"] = None
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
            url_v = st.text_input(
                "Paste image URL:", placeholder="https://example.com/tulsi.jpg"
            )
            if st.button("Load from URL", key="load_url"):
                if url_v.strip():
                    with st.spinner("Downloading..."):
                        try:
                            r = requests.get(
                                url_v.strip(),
                                timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"},
                            )
                            r.raise_for_status()
                            raw = r.content
                            sigs = [
                                b"\xff\xd8\xff",
                                b"\x89PNG\r\n\x1a\n",
                                b"GIF87a",
                                b"BM",
                            ]
                            if (
                                not any(raw[: len(s)] == s for s in sigs)
                                or len(raw) < 100
                            ):
                                raise ValueError("URL did not return a valid image.")
                            _set_image(raw, url_v.strip().split("/")[-1][:50])
                        except Exception as e:
                            st.error(f"Failed to load image: {e}")
                else:
                    st.warning("Please enter a URL first")

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
            if st.button(
                "🔬 Analyze Leaf",
                type="primary",
                use_container_width=True,
                key="analyze_btn",
            ):
                _reset_results()
                st.session_state["phase"] = PHASE_RUNNING
                st.rerun()
            if st.button("🗑️ Clear & Reset", use_container_width=True, key="clear_btn"):
                st.session_state["img_bytes"] = None
                st.session_state["img_hash"] = None
                _reset_results()
                st.rerun()
        else:
            st.markdown(
                """
            <div style="text-align:center;padding:3.5rem 2rem;color:#2d5a3a">
              <div style="font-size:3.5rem;margin-bottom:1rem">🌿</div>
              <div style="font-size:1.2rem;color:#4a8a58;font-family:Georgia,serif">Upload a Tulsi Leaf Image</div>
              <div style="font-size:0.85rem;margin-top:0.5rem;color:#2d5a3a">Supported: JPG, PNG, TIFF, BMP, WebP</div>
            </div>""",
                unsafe_allow_html=True,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSIS (outside columns)
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN: Results
    # ══════════════════════════════════════════════════════════════════════════
    with col2:
        phase = st.session_state["phase"]

        if phase == PHASE_IDLE or pil_img is None:
            st.markdown(
                """
            <div style="text-align:center;padding:5rem 2rem;color:#2d5a3a">
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

            # Segmentation quality warning
            if coverage < 0.05:
                st.warning(
                    f"⚠️ Low leaf coverage detected ({coverage:.1%}). The leaf may be hard to distinguish from the background. "
                    f"Try a plain background or ensure the leaf fills more of the frame. Results may be less accurate."
                )
            elif coverage > 0.90 and "fallback" in seg_method:
                st.warning(
                    "⚠️ Leaf could not be clearly separated from background — analysed full image. "
                    "For best accuracy use a plain white or dark background."
                )
            else:
                st.success(
                    f"✅ Leaf segmented successfully — {coverage:.1%} of image identified as leaf tissue ({seg_method})"
                )

            # Mode badge
            mode_lbl = "OFFLINE (Rule-Based)" if use_offline else "AI (Groq LLaMA3-70B)"
            st.markdown(
                f'<div style="text-align:center;margin:0.5rem 0">'
                f'<span class="badge" style="background:rgba(0,0,0,0.3);color:#fbbf24;font-size:0.82rem;padding:0.3rem 0.9rem">'
                
                f'<span class="badge" style="background:rgba(0,0,0,0.3);color:#60a5fa;font-size:0.82rem;padding:0.3rem 0.9rem">'
                f"🍃 Leaf: {coverage:.0%} of image</span></div>",
                unsafe_allow_html=True,
            )

            # Result card
            if classification["is_healthy"]:
                st.markdown(
                    f"""<div class="result-healthy">
                  <div style="font-size:2.8rem">🌿</div>
                  <div class="result-title" style="color:#22c55e">HEALTHY LEAF</div>
                  <div style="color:#86efac;font-size:0.95rem;margin-top:0.4rem">
                    Confidence: {classification['confidence']}% &nbsp;|&nbsp;
                    {classification['passed_count']}/{classification['total_params']} params passed &nbsp;|&nbsp;
                    Analysis: leaf tissue only
                  </div></div>""",
                    unsafe_allow_html=True,
                )
            else:
                sev_col = {
                    "Mild": "#f59e0b",
                    "Moderate": "#ef4444",
                    "Severe": "#dc2626",
                }.get(classification["severity"], "#ef4444")
                st.markdown(
                    f"""<div class="result-unhealthy">
                  <div style="font-size:2.8rem">⚠️</div>
                  <div class="result-title" style="color:#ef4444">UNHEALTHY LEAF</div>
                  <div style="color:#fca5a5;font-size:0.95rem;margin-top:0.4rem">
                    Confidence: {classification['confidence']}% &nbsp;|&nbsp;
                    Severity: <span style="color:{sev_col};font-weight:700">{classification['severity']}</span>
                    &nbsp;|&nbsp; {classification['passed_count']}/{classification['total_params']} params passed
                  </div></div>""",
                    unsafe_allow_html=True,
                )

            sf = insights.get("safety_flag", "CAUTION")
            sf_c = (
                "#4ade80"
                if "SAFE" in sf
                else "#fbbf24" if "CAUTION" in sf else "#f87171"
            )
            gd = insights.get("quality_grade", "N/A").split("--")[0].strip().rstrip("-")
            st.markdown(
                f'<div style="text-align:center;margin:0.6rem 0">'
                f'<span class="badge" style="background:rgba(0,0,0,0.3);color:{sf_c};font-size:0.82rem;padding:0.3rem 0.9rem">'
                f'🛡 {sf.split("--")[0].strip()}</span>&nbsp;&nbsp;'
                f'<span class="badge" style="background:rgba(0,0,0,0.3);color:#a78bfa;font-size:0.82rem;padding:0.3rem 0.9rem">'
                f"🏷 {gd}</span></div>",
                unsafe_allow_html=True,
            )

            tabs = st.tabs(
                [
                    "📊 Parameters",
                    "🧠 Insights",
                    "⚙️ Pipeline",
                    "🔬 Visuals",
                    "📈 Charts",
                    "📄 Report",
                ]
            )

            # ── TAB 0: Parameters ─────────────────────────────────────────────
            with tabs[0]:
                st.markdown(
                    '<div class="section-title">Quantitative Parameters (Leaf Tissue Only)</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="insight-box">All values computed on <b>{coverage:.1%}</b> of image confirmed as leaf tissue. '
                    f"Background pixels excluded. Segmentation: <i>{seg_method}</i></div>",
                    unsafe_allow_html=True,
                )
                for key in THRESHOLDS:
                    val = metrics.get(key)
                    if val is None:
                        continue
                    info = THRESHOLDS[key]
                    passed = classification["param_results"].get(key, False)
                    weight = CLASSIFICATION_WEIGHTS.get(key, 1)
                    icon = "✅" if passed else "❌"
                    badge = (
                        '<span class="param-pass">Pass</span>'
                        if passed
                        else '<span class="param-fail">Fail</span>'
                    )
                    ref = (
                        f"≤ {info['max']}"
                        if key == "yellow_brown_ratio"
                        else f"{info['min']}–{info['max']} {info['unit']}"
                    )
                    st.markdown(
                        f'<div class="param-row">'
                        f'<span>{icon} <b>{info["label"]}</b> <span style="color:#fbbf24;font-size:0.68rem">(x{weight})</span></span>'
                        f'<span style="color:#93c5fd">{val} {info["unit"]}</span>'
                        f'<span style="color:#4a8a58;font-size:0.78rem">{ref}</span>{badge}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="section-title">Leaf Channel Statistics</div>',
                    unsafe_allow_html=True,
                )
                for lbl, key, col in [
                    ("R Mean", "r_mean", "#f87171"),
                    ("G Mean", "g_mean", "#4ade80"),
                    ("B Mean", "b_mean", "#60a5fa"),
                    ("R StdDev", "r_std", "#f87171"),
                    ("G StdDev", "g_std", "#4ade80"),
                    ("B StdDev", "b_std", "#60a5fa"),
                ]:
                    st.markdown(
                        f'<div class="param-row"><span><span style="color:{col}">●</span> {lbl} (leaf)</span>'
                        f'<span style="color:#93c5fd">{metrics.get(key,"N/A")}</span></div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="section-title">Segmentation Info</div>',
                    unsafe_allow_html=True,
                )
                for lbl, key in [
                    ("Leaf Coverage", "_leaf_coverage"),
                    ("Segmentation Method", "_seg_method"),
                    ("Leaf Pixel Count", "_leaf_px_count"),
                ]:
                    val = metrics.get(key, "N/A")
                    if key == "_leaf_coverage":
                        val = f"{val:.1%}"
                    st.markdown(
                        f'<div class="param-row"><span>🍃 {lbl}</span><span style="color:#93c5fd">{val}</span></div>',
                        unsafe_allow_html=True,
                    )

            # ── TAB 1: Insights ───────────────────────────────────────────────
            with tabs[1]:
                for title, key, border in [
                    ("Clinical Summary", "clinical_summary", "#22c55e"),
                    ("Detailed Pathology", "detailed_pathology", "#f87171"),
                    ("Medical Relevance", "medical_relevance", "#a78bfa"),
                    ("Phytochemical Analysis", "phytochemical_note", "#f59e0b"),
                    ("Treatment Protocol", "treatment_protocol", "#c084fc"),
                    ("Environmental Factors", "environmental_factors", "#2dd4bf"),
                    (
                        "Pharmacopoeial Compliance",
                        "pharmacopoeial_compliance",
                        "#818cf8",
                    ),
                ]:
                    st.markdown(
                        f'<div class="section-title">{title}</div>',
                        unsafe_allow_html=True,
                    )
                    val = insights.get(key, "N/A")
                    if isinstance(val, list):
                        for item in val:
                            st.markdown(
                                f'<div class="insight-box" style="border-left-color:{border}">▸ {item}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            f'<div class="insight-box" style="border-left-color:{border}">{val}</div>',
                            unsafe_allow_html=True,
                        )
                st.markdown(
                    '<div class="section-title">Pathological Indicators</div>',
                    unsafe_allow_html=True,
                )
                for ind in insights.get("pathological_indicators", []):
                    st.markdown(
                        f'<div class="insight-box" style="border-left-color:#60a5fa">🔍 {ind}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    '<div class="section-title">Recommendations</div>',
                    unsafe_allow_html=True,
                )
                for r in insights.get("recommendations", []):
                    st.markdown(
                        f'<div class="insight-box" style="border-left-color:#34d399">▶ {r}</div>',
                        unsafe_allow_html=True,
                    )

            # ── TAB 2: Pipeline ───────────────────────────────────────────────
            with tabs[2]:
                st.markdown(
                    '<div class="section-title">⚙️ Analysis Pipeline</div>',
                    unsafe_allow_html=True,
                )

                steps = [
                    (
                        "Image Input & Validation",
                        "Decoded to PIL Image, converted to RGB, minimum size validated.",
                        [
                            f"Size: {pil_img.size[0]}x{pil_img.size[1]}px",
                            "Mode: RGB",
                            f"Total px: {pil_img.size[0]*pil_img.size[1]:,}",
                        ],
                    ),
                    (
                        "🆕 Leaf Segmentation",
                        "Multi-band HSV thresholding isolates leaf tissue from background. "
                        "4 HSV bands cover healthy green, dark green, brown/necrotic, and yellow/senescent tissue. "
                        "Largest connected component kept; convex hull filled to recover internal holes.",
                        [
                            f"Method: {seg_method}",
                            f"Leaf coverage: {coverage:.1%}",
                            f"Leaf pixels: {metrics.get('_leaf_px_count',0):,}",
                        ],
                    ),
                    (
                        "Metric Extraction (leaf pixels only)",
                        "All 8 health metrics computed exclusively on the segmented leaf region. "
                        "Background pixels are masked out before any calculation.",
                        [
                            f"Green dominance: {metrics.get('green_dominance',0):.2%}",
                            f"Disease ratio: {metrics.get('yellow_brown_ratio',0):.2%}",
                            f"Saturation: {metrics.get('hsv_saturation',0):.0f}",
                            f"Green hue: {metrics.get('hsv_hue_green_ratio',0):.2%}",
                        ],
                    ),
                    (
                        "Weighted Classification",
                        "8 parameters compared to leaf-tissue-specific thresholds. "
                        "Weighted score ≥55% = Healthy. Hard penalty gates applied for critical failures.",
                        [
                            f"Score: {classification['weighted_score']}/{classification['max_weighted']}",
                            f"Raw: {classification['confidence_raw']*100:.1f}%",
                            f"Result: {classification['status']}",
                        ],
                    ),
                    (
                        "AI Insight Generation",
                        f"{'Offline rule-based expert system' if use_offline else 'Groq LLaMA3-70B'} generates 11 clinical insight fields "
                        "referencing leaf-tissue metric values (not background-contaminated values).",
                        [
                            f"Mode: {'Offline' if use_offline else 'Groq AI'}",
                            f"Indicators: {len(insights.get('pathological_indicators',[]))}",
                            f"Recommendations: {len(insights.get('recommendations',[]))}",
                        ],
                    ),
                ]
                for i, (title, desc, chips) in enumerate(steps, 1):
                    chips_html = "".join(
                        f'<span class="chip">{c}</span>' for c in chips
                    )
                    st.markdown(
                        f"""<div class="pipeline-step">
                      <span class="pipeline-num">{i}</span>
                      <span style="color:#a5d6a7;font-family:Georgia,serif;font-size:1.05rem;font-weight:700">{title}</span>
                      <div class="pipeline-step-desc" style="margin-top:0.6rem">{desc}</div>
                      <div style="margin-top:0.5rem">{chips_html}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                    if i == 2 and vis_cache.get("segmentation"):
                        st.image(vis_cache["segmentation"], use_container_width=True)
                    if i == 4 and vis_cache.get("classification_breakdown"):
                        st.image(
                            vis_cache["classification_breakdown"],
                            use_container_width=True,
                        )

            # ── TAB 3: Visual Analysis ────────────────────────────────────────
            with tabs[3]:
                st.markdown(
                    '<div class="section-title">Visual Analysis Suite</div>',
                    unsafe_allow_html=True,
                )
                for vkey, vtitle, vdesc in [
                    (
                        "segmentation",
                        "Leaf Segmentation",
                        "Green overlay shows identified leaf tissue; background is dimmed.",
                    ),
                    (
                        "rgb_bands",
                        "RGB Channel Decomposition",
                        "R/G/B channels on leaf pixels only; background shown as dark.",
                    ),
                    (
                        "disease",
                        "Disease Spot Mapping",
                        "Yellow/brown lesions detected on leaf tissue (red overlay). Background excluded.",
                    ),
                    (
                        "green_map",
                        "Green Dominance Map",
                        "Leaf pixels where G>R AND G>B highlighted; non-green leaf areas dimmed.",
                    ),
                    (
                        "edge",
                        "Edge Detection (Leaf)",
                        "Canny edges on leaf region only. Vein structure and tissue integrity.",
                    ),
                    (
                        "histogram",
                        "RGB Histogram (Leaf)",
                        "Pixel intensity distribution from leaf pixels only — not whole image.",
                    ),
                ]:
                    vdata = vis_cache.get(vkey)
                    if vdata:
                        st.markdown(
                            f'<div class="section-title">{vtitle}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="insight-box" style="font-size:0.87rem">{vdesc}</div>',
                            unsafe_allow_html=True,
                        )
                        st.image(vdata, use_container_width=True)

            # ── TAB 4: Charts ─────────────────────────────────────────────────
            with tabs[4]:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        '<div style="text-align:center;color:#6ab87a;font-size:0.9rem;margin-bottom:0.3rem">Parameter Radar</div>',
                        unsafe_allow_html=True,
                    )
                    if vis_cache.get("radar"):
                        st.image(vis_cache["radar"], use_container_width=True)
                with c2:
                    st.markdown(
                        '<div style="text-align:center;color:#6ab87a;font-size:0.9rem;margin-bottom:0.3rem">Key Health Indicators</div>',
                        unsafe_allow_html=True,
                    )
                    if vis_cache.get("bars"):
                        st.image(vis_cache["bars"], use_container_width=True)

                sp = (
                    classification["weighted_score"] / classification["max_weighted"]
                    if classification["max_weighted"] > 0
                    else 0
                )
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Overall Health Score (Leaf Tissue)</div>'
                    f'<div class="metric-value">{sp*100:.0f} / 100</div>'
                    f'<div class="metric-sub">Weighted across {classification["total_params"]} parameters | Leaf coverage: {coverage:.1%}</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(sp)

                if vis_cache.get("classification_breakdown"):
                    st.markdown(
                        '<div class="section-title">Classification Scoring Breakdown</div>',
                        unsafe_allow_html=True,
                    )
                    st.image(
                        vis_cache["classification_breakdown"], use_container_width=True
                    )

            # ── TAB 5: Report ─────────────────────────────────────────────────
            with tabs[5]:
                st.markdown(
                    '<div class="section-title">Generate PDF Report</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="insight-box">Comprehensive medical-grade PDF including leaf segmentation, all visualisations, '
                    "clinical insights, and pharmacopoeial compliance assessment. "
                    "All metrics reported on leaf tissue only.</div>",
                    unsafe_allow_html=True,
                )
                try:
                    pdf_bytes = generate_pdf(
                        pil_img,
                        metrics,
                        classification,
                        insights,
                        use_case,
                        patient_name,
                        sample_id,
                        vis_cache,
                    )
                    fname = f"TulsiReport_{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                    report_json = {
                        "sample_id": sample_id,
                        "patient": patient_name,
                        "use_case": use_case,
                        "timestamp": datetime.now().isoformat(),
                        "leaf_coverage": coverage,
                        "segmentation_method": seg_method,
                        "metrics": {
                            k: v for k, v in metrics.items() if not k.startswith("_")
                        },
                        "classification": {
                            k: v
                            for k, v in classification.items()
                            if k not in ("param_results", "param_details")
                        },
                        "param_results": classification.get("param_results", {}),
                        "insights": insights,
                    }
                    st.download_button(
                        "⬇️ Download JSON Data",
                        data=json.dumps(report_json, indent=2, default=str),
                        file_name=f"TulsiData_{sample_id}.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                    st.success(
                        f"✅ Report ready! {len(vis_cache)} visualisations included."
                    )
                except Exception as e:
                    st.error(f"PDF error: {e}")
                    st.info("JSON download above still works.")

    # Debug expander
    with st.expander("🔧 Debug Info", expanded=False):
        st.json(
            {
                "phase": st.session_state.get("phase"),
                "has_img": st.session_state.get("img_bytes") is not None,
                "has_metrics": st.session_state.get("metrics") is not None,
                "has_classification": st.session_state.get("classification")
                is not None,
                "has_vis": st.session_state.get("vis_cache") is not None,
                "error": st.session_state.get("error_msg", ""),
                "leaf_coverage": (
                    st.session_state.get("metrics", {}).get("_leaf_coverage", "N/A")
                    if st.session_state.get("metrics")
                    else "N/A"
                ),
                "seg_method": (
                    st.session_state.get("metrics", {}).get("_seg_method", "N/A")
                    if st.session_state.get("metrics")
                    else "N/A"
                ),
            }
        )


if __name__ == "__main__":
    main()