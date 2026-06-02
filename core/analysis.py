import numpy as np
from PIL import Image
import cv2
from .segmentation import segment_leaf


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

    leaf_mask, coverage, seg_method = segment_leaf(img_np)
    leaf_bool = leaf_mask > 0

    R_full = img_np[:, :, 0].astype(float)
    G_full = img_np[:, :, 1].astype(float)
    B_full = img_np[:, :, 2].astype(float)

    R = R_full[leaf_bool]
    G = G_full[leaf_bool]
    B = B_full[leaf_bool]
    gray_leaf = gray[leaf_bool].astype(float)
    hue_leaf = img_hsv[:, :, 0][leaf_bool].astype(float)
    sat_leaf = img_hsv[:, :, 1][leaf_bool].astype(float)

    total = R + G + B + 1e-6
    green_dom_px = (G > R) & (G > B)
    green_dominance = (
        float(np.sum(green_dom_px) / len(green_dom_px)) if len(green_dom_px) > 0 else 0
    )
    green_ratio = float(np.mean(G / total))

    hsv_saturation = float(np.mean(sat_leaf))
    green_hue_px = (hue_leaf >= 30) & (hue_leaf <= 90)
    hsv_hue_green_ratio = (
        float(np.sum(green_hue_px) / len(hue_leaf)) if len(hue_leaf) > 0 else 0
    )

    hue_2d = img_hsv[:, :, 0].astype(float)
    sat_2d = img_hsv[:, :, 1].astype(float)
    val_2d = img_hsv[:, :, 2].astype(float)

    brown_2d = (
        (hue_2d >= 0)
        & (hue_2d <= 18)
        & (sat_2d >= 35)
        & (sat_2d <= 200)
        & (val_2d >= 40)
        & (val_2d <= 180)
    )
    yellow_2d = (hue_2d >= 15) & (hue_2d <= 30) & (sat_2d >= 50)

    brown_leaf_mask = brown_2d & leaf_bool
    yellow_leaf_mask = yellow_2d & leaf_bool
    disease_mask_2d = brown_leaf_mask | yellow_leaf_mask

    leaf_pixel_count = float(np.sum(leaf_bool))
    yellow_brown_ratio = (
        float(np.sum(disease_mask_2d)) / leaf_pixel_count if leaf_pixel_count > 0 else 0
    )

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
        "green_dom_mask": np.zeros_like(leaf_bool, dtype=bool),
        "edges_leaf": edges_leaf,
        "edges_full": edges_full,
    }
    gd_2d = (G_full > R_full) & (G_full > B_full) & leaf_bool
    intermediates["green_dom_mask"] = gd_2d

    return {"metrics": metrics, "intermediates": intermediates}