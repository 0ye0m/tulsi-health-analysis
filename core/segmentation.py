import numpy as np
import cv2


def segment_leaf(img_np: np.ndarray) -> tuple[np.ndarray, float, str]:
    """
    Robustly isolate the leaf from the background using a 3‑stage pipeline:
    (1) Multi‑band HSV thresholding, (2) Morphological cleanup,
    (3) Keep largest connected component.
    Falls back to full‑image mask if <3% of pixels are segmented.
    """
    h, w = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # ── Stage 1: Multi‑band HSV thresholding ──────────────────────────────────
    mask_a = cv2.inRange(img_hsv, np.array([25, 30, 25]), np.array([100, 255, 240]))
    mask_b = cv2.inRange(img_hsv, np.array([20, 15, 15]), np.array([100, 255, 100]))
    mask_c = cv2.inRange(img_hsv, np.array([0, 35, 40]), np.array([25, 220, 220]))
    mask_d = cv2.inRange(img_hsv, np.array([18, 60, 60]), np.array([35, 255, 255]))
    combined = cv2.bitwise_or(
        cv2.bitwise_or(mask_a, mask_b), cv2.bitwise_or(mask_c, mask_d)
    )

    # ── Stage 2: Morphological cleanup ───────────────────────────────────────
    short = min(h, w)
    ksize = max(7, short // 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ksmall, iterations=2)
    kdilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ksize // 2 + 3, ksize // 2 + 3)
    )
    combined = cv2.dilate(combined, kdilate, iterations=2)

    # ── Stage 3: Keep largest connected component ─────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined, connectivity=8
    )
    if num_labels <= 1:
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
        coverage = 1.0
        method = "full-image fallback (no leaf detected by color)"
    else:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        leaf_mask_raw = (labels == largest).astype(np.uint8) * 255
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

    if coverage < 0.03:
        leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
        coverage = 1.0
        method = "full-image fallback (coverage < 3%)"

    return leaf_mask, coverage, method