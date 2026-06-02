import io
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2


def _save_fig(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#0f1a12", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _darkbg_fig(*args, **kwargs):
    fig = plt.figure(*args, **kwargs, facecolor="#0f1a12")
    return fig


def make_segmentation_vis(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"]
    leaf_mask = inter["leaf_mask"]
    leaf_bool = inter["leaf_bool"]

    masked = img_rgb.copy()
    masked[~leaf_bool] = [15, 30, 15]

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
    axes[0].set_title("Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8)
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Leaf Segmentation  (Coverage: {metrics['_leaf_coverage']:.1%})",
        color="#4ade80", fontsize=10, pad=8,
    )
    axes[2].imshow(masked)
    axes[2].set_title("Leaf Pixels Only\n(Metrics computed here)", color="#fbbf24", fontsize=10, pad=8)
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle("Leaf Segmentation — Background Excluded from Analysis",
                 color="#e8f5e9", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_fig(fig)


def make_rgb_bands(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"]
    R, G, B = inter["R_full"], inter["G_full"], inter["B_full"]
    leaf_bool = inter["leaf_bool"]
    R_vis = R.copy()
    R_vis[~leaf_bool] = 15
    G_vis = G.copy()
    G_vis[~leaf_bool] = 15
    B_vis = B.copy()
    B_vis[~leaf_bool] = 15
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), facecolor="#0f1a12")
    axes = axes.flatten()
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8)
    axes[1].imshow(R_vis.astype(np.uint8), cmap="Reds", vmin=0, vmax=255)
    axes[1].set_title(f"Red Channel  (Leaf Mean:{metrics['r_mean']:.0f})", color="#f87171", fontsize=10, pad=8)
    axes[2].imshow(G_vis.astype(np.uint8), cmap="Greens", vmin=0, vmax=255)
    axes[2].set_title(f"Green Channel (Leaf Mean:{metrics['g_mean']:.0f})", color="#4ade80", fontsize=10, pad=8)
    axes[3].imshow(B_vis.astype(np.uint8), cmap="Blues", vmin=0, vmax=255)
    axes[3].set_title(f"Blue Channel  (Leaf Mean:{metrics['b_mean']:.0f})", color="#60a5fa", fontsize=10, pad=8)
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle("RGB Channel Decomposition (Leaf Region Only)", color="#e8f5e9", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig)


def make_disease_spot_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"].copy()
    disease_mask = inter["disease_mask"]
    leaf_bool = inter["leaf_bool"]
    overlay = img_rgb.copy()
    idx = disease_mask & leaf_bool
    overlay[idx, 0] = np.minimum(overlay[idx, 0].astype(int) + 120, 255).astype(np.uint8)
    overlay[idx, 1] = np.maximum(overlay[idx, 1].astype(int) - 60, 0).astype(np.uint8)
    overlay[idx, 2] = np.maximum(overlay[idx, 2].astype(int) - 60, 0).astype(np.uint8)
    overlay[~leaf_bool] = (overlay[~leaf_bool].astype(float) * 0.25).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8)
    axes[1].imshow(overlay)
    pct = metrics["yellow_brown_ratio"] * 100
    col = "#f87171" if pct > 12 else "#fbbf24" if pct > 6 else "#4ade80"
    axes[1].set_title(f"Disease Spots on Leaf  (Affected: {pct:.2f}%)\n[threshold: <12% of leaf]", color=col, fontsize=10, pad=8)
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle("Yellow / Brown Lesion Mapping (Leaf Pixels Only)", color="#e8f5e9", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_fig(fig)


def make_green_dominance_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"].copy()
    gd_mask = inter["green_dom_mask"]
    leaf_bool = inter["leaf_bool"]
    overlay = img_rgb.copy()
    overlay[gd_mask, 1] = np.minimum(overlay[gd_mask, 1].astype(int) + 55, 255).astype(np.uint8)
    non_gd_leaf = leaf_bool & ~gd_mask
    overlay[non_gd_leaf] = (overlay[non_gd_leaf].astype(float) * 0.55).astype(np.uint8)
    overlay[~leaf_bool] = (overlay[~leaf_bool].astype(float) * 0.15).astype(np.uint8)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8)
    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Green Dominance (G>R AND G>B)  {metrics['green_dominance']:.1%} of leaf",
        color="#4ade80", fontsize=10, pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle("Chlorophyll Distribution — Green Dominant Pixels on Leaf",
                 color="#e8f5e9", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_fig(fig)


def make_edge_map(inter: dict, metrics: dict) -> bytes:
    img_rgb = inter["img_rgb"]
    edges_leaf = inter["edges_leaf"]
    edge_col = np.zeros((*edges_leaf.shape, 3), dtype=np.uint8)
    edge_col[edges_leaf > 0] = [34, 197, 94]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor="#0f1a12")
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", color="#a5d6a7", fontsize=10, fontweight="bold", pad=8)
    axes[1].imshow(edge_col)
    axes[1].set_title(
        f"Canny Edges on Leaf  (Density:{metrics['edge_density']:.2%})",
        color="#4ade80", fontsize=10, pad=8,
    )
    for ax in axes:
        ax.axis("off")
        ax.set_facecolor("#0a1a0d")
    fig.suptitle("Structural Edge Analysis (Leaf Region Only)", color="#e8f5e9", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save_fig(fig)


def make_color_histogram(inter: dict, metrics: dict) -> bytes:
    leaf_bool = inter["leaf_bool"]
    R = inter["R_full"][leaf_bool].flatten().astype(np.uint8)
    G = inter["G_full"][leaf_bool].flatten().astype(np.uint8)
    B = inter["B_full"][leaf_bool].flatten().astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="#0f1a12")
    ax.set_facecolor("#0a1a0d")
    ax.hist(R, bins=256, range=(0, 256), color="#ef4444", alpha=0.5,
            label=f"Red  (Mean:{np.mean(R):.0f})", density=True)
    ax.hist(G, bins=256, range=(0, 256), color="#22c55e", alpha=0.5,
            label=f"Green(Mean:{np.mean(G):.0f})", density=True)
    ax.hist(B, bins=256, range=(0, 256), color="#3b82f6", alpha=0.5,
            label=f"Blue (Mean:{np.mean(B):.0f})", density=True)
    ax.axvline(np.mean(G), color="#4ade80", linestyle="--", linewidth=1.5,
               alpha=0.8, label="Green mean")
    ax.set_xlabel("Pixel Intensity (0-255)", color="#6ab87a", fontsize=10)
    ax.set_ylabel("Normalised Frequency (Leaf Pixels)", color="#6ab87a", fontsize=10)
    ax.set_title("RGB Histogram — Leaf Pixels Only", color="#e8f5e9", fontsize=12,
                 fontweight="bold", pad=12)
    ax.legend(loc="upper right", facecolor="#0f1a12", edgecolor="#1a3a22",
              labelcolor="#c8e6c9", fontsize=9)
    ax.tick_params(colors="#4a8a58")
    ax.spines[:].set_edgecolor("#1a3a22")
    ax.set_xlim(0, 255)
    plt.tight_layout()
    return _save_fig(fig)


def make_radar_chart(metrics: dict) -> bytes:
    from config import THRESHOLDS
    cats = [
        "Intensity", "Green\nRatio", "Green\nDom.", "Saturation",
        "Green\nHue", "Low\nDisease", "Texture", "Edges",
    ]
    keys = list(THRESHOLDS.keys())

    def norm(key, val):
        lo, hi = THRESHOLDS[key]["min"], THRESHOLDS[key]["max"]
        if hi <= lo:
            return 0.5
        if key == "yellow_brown_ratio":
            return max(0.0, min(1.0, 1.0 - min(val / hi, 1.0)))
        mid = (lo + hi) / 2.0
        half = (hi - lo) / 2.0
        return max(0.0, min(1.0, 1.0 - abs((val - mid) / half)))

    values = [norm(k, metrics.get(k, 0)) for k in keys] + [norm(keys[0], metrics.get(keys[0], 0))]
    N = len(cats)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0.0]
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True), facecolor="#0f1a12")
    ax.set_facecolor("#0f1a12")
    ax.set_ylim(0, 1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, color="#6ab87a", fontsize=8.5)
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="#2d5a3a", fontsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a3a22")
    ax.plot(angles, values, color="#22c55e", linewidth=2.2)
    ax.fill(angles, values, color="#22c55e", alpha=0.22)
    ax.plot(angles, [0.5] * (N + 1), "--", color="#f59e0b", linewidth=1, alpha=0.6, label="Threshold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), labelcolor="#f59e0b",
              facecolor="#0f1a12", edgecolor="#1a3a22", fontsize=8)
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
    thr = [0.50, 0.58, 0.40]
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
        ax.text(min(val + 0.04, 0.97), 0, f"{val*100:.0f}%", va="center",
                color=col, fontsize=9, fontweight="bold")
    plt.suptitle("Key Health Indicators (Leaf Pixels Only)", color="#a5d6a7", fontsize=10, y=1.05)
    plt.tight_layout()
    return _save_fig(fig)


def make_classification_breakdown(classification: dict) -> bytes:
    details = classification.get("param_details", [])
    if not details:
        return b""
    fig, ax = plt.subplots(figsize=(9, max(4, len(details) * 0.65)), facecolor="#0f1a12")
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
        ax.text(max(weights) + 0.2, i, f"{'✓' if p else '✗'} (w={w})",
                va="center", color=cols[i], fontsize=9, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="#c8e6c9", fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("Weight", color="#6ab87a", fontsize=10)
    ax.set_title(
        f"Classification Breakdown  |  Score:{classification['weighted_score']}/{classification['max_weighted']}  "
        f"({classification['confidence_raw']*100:.1f}% raw  →  {'✓ HEALTHY' if classification['is_healthy'] else '✗ UNHEALTHY'})",
        color="#e8f5e9", fontsize=11, fontweight="bold", pad=12,
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