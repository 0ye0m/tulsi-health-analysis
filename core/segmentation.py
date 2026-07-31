# import numpy as np
# import cv2


# def segment_leaf(img_np: np.ndarray) -> tuple[np.ndarray, float, str]:
#     """
#     Robustly isolate the leaf from the background using a 3‑stage pipeline:
#     (1) Multi‑band HSV thresholding, (2) Morphological cleanup,
#     (3) Keep largest connected component.
#     Falls back to full‑image mask if <3% of pixels are segmented.
#     """
#     h, w = img_np.shape[:2]
#     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

#     # ── Stage 1: Multi‑band HSV thresholding ──────────────────────────────────
#     mask_a = cv2.inRange(img_hsv, np.array([25, 30, 25]), np.array([100, 255, 240]))
#     mask_b = cv2.inRange(img_hsv, np.array([20, 15, 15]), np.array([100, 255, 100]))
#     mask_c = cv2.inRange(img_hsv, np.array([0, 35, 40]), np.array([25, 220, 220]))
#     mask_d = cv2.inRange(img_hsv, np.array([18, 60, 60]), np.array([35, 255, 255]))
#     combined = cv2.bitwise_or(
#         cv2.bitwise_or(mask_a, mask_b), cv2.bitwise_or(mask_c, mask_d)
#     )

#     # ── Stage 2: Morphological cleanup ───────────────────────────────────────
#     short = min(h, w)
#     ksize = max(7, short // 30)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
#     ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ksmall, iterations=2)
#     kdilate = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, (ksize // 2 + 3, ksize // 2 + 3)
#     )
#     combined = cv2.dilate(combined, kdilate, iterations=2)

#     # ── Stage 3: Keep largest connected component ─────────────────────────────
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#         combined, connectivity=8
#     )
#     if num_labels <= 1:
#         leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
#         coverage = 1.0
#         method = "full-image fallback (no leaf detected by color)"
#     else:
#         largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         leaf_mask_raw = (labels == largest).astype(np.uint8) * 255
#         contours, _ = cv2.findContours(
#             leaf_mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         leaf_mask = np.zeros((h, w), dtype=np.uint8)
#         if contours:
#             hull = cv2.convexHull(max(contours, key=cv2.contourArea))
#             cv2.fillPoly(leaf_mask, [hull], 255)
#         else:
#             leaf_mask = leaf_mask_raw
#         coverage = float(np.sum(leaf_mask > 0)) / (h * w)
#         method = "HSV multi-band + convex hull"

#     if coverage < 0.03:
#         leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
#         coverage = 1.0
#         method = "full-image fallback (coverage < 3%)"

#     return leaf_mask, coverage, method


# import numpy as np
# import cv2


# def _illumination_normalize(img_bgr: np.ndarray) -> np.ndarray:
#     lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
#     l = clahe.apply(l)
#     lab = cv2.merge((l, a, b))
#     return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# def _exg_mask(img_bgr: np.ndarray) -> np.ndarray:
#     img_f = img_bgr.astype(np.float32)
#     b, g, r = cv2.split(img_f)
#     exg = 2 * g - r - b
#     exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     _, mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return mask


# def _fill_holes(mask: np.ndarray) -> np.ndarray:
#     h, w = mask.shape
#     padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
#     flood = padded.copy()
#     ff_mask = np.zeros((h + 4, w + 4), np.uint8)
#     cv2.floodFill(flood, ff_mask, (0, 0), 255)
#     flood = flood[1:-1, 1:-1]
#     flood_inv = cv2.bitwise_not(flood)
#     return cv2.bitwise_or(mask, flood_inv)


# def _grabcut_refine(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     h, w = mask.shape
#     coverage_in = float(np.mean(mask > 0))

#     fg_k = max(3, min(h, w) // 80)
#     fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_k, fg_k))
#     sure_fg = cv2.erode(mask, fg_kernel, iterations=2)

#     grow = min(max(5, min(h, w) // 30), 30)
#     bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
#     probable_region = cv2.dilate(mask, bg_kernel, iterations=1)

#     gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
#     gc_mask[probable_region > 0] = cv2.GC_PR_FGD
#     gc_mask[mask > 0] = cv2.GC_PR_FGD
#     gc_mask[sure_fg > 0] = cv2.GC_FGD
#     gc_mask[probable_region == 0] = cv2.GC_BGD

#     if np.mean(gc_mask == cv2.GC_BGD) < 0.05:
#         return mask

#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)
#     try:
#         cv2.grabCut(img_bgr, gc_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
#     except cv2.error:
#         return mask

#     refined = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
#     refined_coverage = float(np.mean(refined > 0))

#     if refined_coverage > 0.92 or refined_coverage > coverage_in * 2.5 + 0.05:
#         return mask

#     return refined


# def segment_leaf(img_np: np.ndarray) -> tuple[np.ndarray, float, str]:
#     h, w = img_np.shape[:2]
#     img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
#     img_bgr_norm = _illumination_normalize(img_bgr)
#     img_hsv = cv2.cvtColor(img_bgr_norm, cv2.COLOR_BGR2HSV)

#     mask_a = cv2.inRange(img_hsv, np.array([25, 30, 20]), np.array([100, 255, 255]))
#     mask_b = cv2.inRange(img_hsv, np.array([20, 15, 10]), np.array([100, 255, 90]))
#     mask_c = cv2.inRange(img_hsv, np.array([0, 30, 30]), np.array([25, 220, 230]))
#     mask_d = cv2.inRange(img_hsv, np.array([18, 60, 50]), np.array([35, 255, 255]))
#     hsv_mask = cv2.bitwise_or(cv2.bitwise_or(mask_a, mask_b), cv2.bitwise_or(mask_c, mask_d))

#     exg_mask = _exg_mask(img_bgr_norm)
#     combined = cv2.bitwise_or(hsv_mask, exg_mask)

#     short = min(h, w)
#     ksize = max(7, short // 30)
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
#     ksmall = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#     combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, ksmall, iterations=1)
#     combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
#     combined = _fill_holes(combined)

#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
#     if num_labels <= 1:
#         leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
#         return leaf_mask, 1.0, "full-image fallback (no leaf detected by color)"

#     best_idx = None
#     best_score = -1.0
#     for i in range(1, num_labels):
#         area = stats[i, cv2.CC_STAT_AREA]
#         bw = stats[i, cv2.CC_STAT_WIDTH]
#         bh = stats[i, cv2.CC_STAT_HEIGHT]
#         density = area / float(bw * bh)
#         if density < 0.25:
#             continue
#         if area > best_score:
#             best_score = area
#             best_idx = i

#     if best_idx is None:
#         best_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

#     component_mask = (labels == best_idx).astype(np.uint8) * 255
#     component_mask = _fill_holes(component_mask)

#     refined_mask = _grabcut_refine(img_bgr, component_mask)
#     refined_mask = _fill_holes(refined_mask)

#     ksmall2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, ksmall2, iterations=1)
#     refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, ksmall2, iterations=1)

#     coverage = float(np.sum(refined_mask > 0)) / (h * w)
#     method = "CLAHE + HSV/ExG fusion + GrabCut refinement"

#     if coverage < 0.03:
#         leaf_mask = np.ones((h, w), dtype=np.uint8) * 255
#         return leaf_mask, 1.0, "full-image fallback (coverage < 3%)"

#     contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         biggest = max(contours, key=cv2.contourArea)
#         final_mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.drawContours(final_mask, [biggest], -1, 255, thickness=cv2.FILLED)
#         final_mask = _fill_holes(final_mask)
#     else:
#         final_mask = refined_mask

#     coverage = float(np.sum(final_mask > 0)) / (h * w)
#     return final_mask, coverage, method


import cv2
import numpy as np


# ============================================================
# 1. ILLUMINATION NORMALIZATION
# ============================================================

def _illumination_normalize(img_bgr: np.ndarray) -> np.ndarray:
    """
    Normalize uneven illumination using CLAHE on the L channel
    of LAB color space while preserving color information.
    """

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge(
        (l_channel, a_channel, b_channel)
    )

    enhanced_bgr = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced_bgr


# ============================================================
# 2. EXCESS GREEN (ExG)
# ============================================================

def _exg_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Generate vegetation mask using Excess Green:

        ExG = 2G - R - B

    Otsu thresholding automatically separates high-ExG pixels.
    """

    img_float = img_bgr.astype(np.float32)

    blue, green, red = cv2.split(img_float)

    exg = (
        2.0 * green
        - red
        - blue
    )

    exg = cv2.normalize(
        exg,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    exg_uint8 = exg.astype(np.uint8)

    _, mask = cv2.threshold(
        exg_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return mask


# ============================================================
# 3. KEEP LARGEST COMPONENT
# ============================================================

def _largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected foreground component.
    """

    h, w = mask.shape

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    if num_labels <= 1:
        return np.zeros(
            (h, w),
            dtype=np.uint8
        )

    areas = stats[1:, cv2.CC_STAT_AREA]

    largest_index = int(
        np.argmax(areas)
    ) + 1

    result = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    result[labels == largest_index] = 255

    return result


# ============================================================
# 4. GRABCUT REFINEMENT
# ============================================================

def _grabcut_refine(
    img_bgr: np.ndarray,
    candidate_mask: np.ndarray,
    roi_mask: np.ndarray
) -> np.ndarray:
    """
    Refine segmentation using GrabCut.

    Outside ROI:
        definite background

    Inside ROI:
        probable background

    Candidate pixels:
        probable foreground

    Eroded candidate:
        definite foreground
    """

    h, w = candidate_mask.shape

    # --------------------------------------------------------
    # Validate candidate
    # --------------------------------------------------------

    if np.count_nonzero(candidate_mask) == 0:
        return candidate_mask.copy()

    # --------------------------------------------------------
    # Create sure foreground
    # --------------------------------------------------------

    fg_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    sure_fg = cv2.erode(
        candidate_mask,
        fg_kernel,
        iterations=2
    )

    # If erosion destroyed everything, use original candidate
    if np.count_nonzero(sure_fg) == 0:
        sure_fg = candidate_mask.copy()

    # --------------------------------------------------------
    # GrabCut mask
    # --------------------------------------------------------

    gc_mask = np.full(
        (h, w),
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Search area
    gc_mask[roi_mask > 0] = cv2.GC_PR_BGD

    # Potential leaf
    gc_mask[candidate_mask > 0] = cv2.GC_PR_FGD

    # Strong leaf interior
    gc_mask[sure_fg > 0] = cv2.GC_FGD

    # --------------------------------------------------------
    # Models required by OpenCV
    # --------------------------------------------------------

    bgd_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        dtype=np.float64
    )

    try:

        cv2.grabCut(
            img_bgr,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        # GrabCut can fail when foreground/background variation
        # is insufficient. Candidate mask is safer fallback.
        return candidate_mask.copy()

    # --------------------------------------------------------
    # Extract foreground
    # --------------------------------------------------------

    foreground = (
        (gc_mask == cv2.GC_FGD)
        | (gc_mask == cv2.GC_PR_FGD)
    )

    refined = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    refined[foreground] = 255

    # Never allow segmentation outside ROI
    refined = cv2.bitwise_and(
        refined,
        roi_mask
    )

    return refined


# ============================================================
# 5. MAIN LEAF SEGMENTATION
# ============================================================

def segment_leaf(
    img_np: np.ndarray
) -> tuple[np.ndarray, float, str]:

    """
    Medicinal leaf segmentation pipeline.

    Pipeline:

    RGB
      ↓
    Bilateral filtering
      ↓
    CLAHE illumination normalization
      ↓
    HSV + Excess Green detection
      ↓
    Strong green leaf core
      ↓
    Largest component
      ↓
    Convex-hull ROI
      ↓
    Yellow/brown tissue recovery
      ↓
    GrabCut boundary refinement
      ↓
    Largest actual contour
      ↓
    Final leaf mask

    Convex hull is used ONLY for localization.
    It is NOT used as the final segmentation boundary.
    """

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    if img_np is None:
        raise ValueError("Input image cannot be None.")

    if not isinstance(img_np, np.ndarray):
        raise TypeError(
            "Input image must be a NumPy array."
        )

    if img_np.ndim != 3:
        raise ValueError(
            "Input image must have 3 dimensions: H x W x 3."
        )

    if img_np.shape[2] != 3:
        raise ValueError(
            "Input image must contain exactly 3 RGB channels."
        )

    # Ensure uint8 for OpenCV
    if img_np.dtype != np.uint8:

        img_np = np.clip(
            img_np,
            0,
            255
        ).astype(np.uint8)

    h, w = img_np.shape[:2]

    image_area = float(h * w)

    # ========================================================
    # STAGE 1 — PREPROCESSING
    # ========================================================

    img_bgr = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    # Edge-preserving noise reduction
    img_smooth = cv2.bilateralFilter(
        img_bgr,
        7,
        40,
        40
    )

    # Lighting normalization
    img_norm = _illumination_normalize(
        img_smooth
    )

    # ========================================================
    # STAGE 2 — COLOR SPACES
    # ========================================================

    hsv = cv2.cvtColor(
        img_norm,
        cv2.COLOR_BGR2HSV
    )

    _, saturation, _ = cv2.split(hsv)

    # ========================================================
    # STAGE 3 — STRONG GREEN LEAF DETECTION
    # ========================================================

    # Normal healthy green
    green_mask = cv2.inRange(
        hsv,
        np.array(
            [28, 45, 25],
            dtype=np.uint8
        ),
        np.array(
            [95, 255, 255],
            dtype=np.uint8
        )
    )

    # Dark green regions
    dark_green_mask = cv2.inRange(
        hsv,
        np.array(
            [25, 25, 10],
            dtype=np.uint8
        ),
        np.array(
            [100, 255, 120],
            dtype=np.uint8
        )
    )

    hsv_green = cv2.bitwise_or(
        green_mask,
        dark_green_mask
    )

    # ========================================================
    # STAGE 4 — EXCESS GREEN
    # ========================================================

    exg = _exg_mask(
        img_norm
    )

    # ExG may react to low-color backgrounds.
    # Require minimum saturation.
    saturation_mask = cv2.inRange(
        saturation,
        30,
        255
    )

    exg = cv2.bitwise_and(
        exg,
        saturation_mask
    )

    # Require vegetation evidence from HSV OR ExG
    vegetation = cv2.bitwise_or(
        hsv_green,
        exg
    )

    # ========================================================
    # STAGE 5 — CLEAN VEGETATION MASK
    # ========================================================

    kernel3 = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    vegetation = cv2.morphologyEx(
        vegetation,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

    vegetation = cv2.morphologyEx(
        vegetation,
        cv2.MORPH_CLOSE,
        kernel3,
        iterations=2
    )

    # ========================================================
    # STAGE 6 — FIND MAIN LEAF CORE
    # ========================================================

    leaf_core = _largest_component(
        vegetation
    )

    core_pixels = int(
        np.count_nonzero(leaf_core)
    )

    core_coverage = (
        core_pixels / image_area
    )

    # No reliable leaf detected
    if core_coverage < 0.01:

        empty_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        return (
            empty_mask,
            0.0,
            "segmentation failed — no reliable leaf core detected"
        )

    # ========================================================
    # STAGE 7 — INITIAL CONTOUR
    # ========================================================

    contours, _ = cv2.findContours(
        leaf_core,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return (
            leaf_core,
            float(core_coverage),
            "green-core segmentation"
        )

    main_contour = max(
        contours,
        key=cv2.contourArea
    )

    # ========================================================
    # STAGE 8 — CONVEX HULL
    #
    # ONLY FOR LOCALIZATION.
    # NOT FINAL SEGMENTATION.
    # ========================================================

    hull = cv2.convexHull(
        main_contour
    )

    hull_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    cv2.fillConvexPoly(
        hull_mask,
        hull,
        255
    )

    # ========================================================
    # STAGE 9 — EXPAND ROI
    # ========================================================

    short_side = min(h, w)

    roi_kernel_size = max(
        5,
        short_side // 60
    )

    # Keep odd
    if roi_kernel_size % 2 == 0:
        roi_kernel_size += 1

    # Avoid excessively large kernel
    roi_kernel_size = min(
        roi_kernel_size,
        31
    )

    roi_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            roi_kernel_size,
            roi_kernel_size
        )
    )

    roi_mask = cv2.dilate(
        hull_mask,
        roi_kernel,
        iterations=2
    )

    # ========================================================
    # STAGE 10 — RECOVER YELLOW LEAF TISSUE
    # ========================================================

    yellow_mask = cv2.inRange(
        hsv,
        np.array(
            [18, 45, 40],
            dtype=np.uint8
        ),
        np.array(
            [35, 255, 255],
            dtype=np.uint8
        )
    )

    # ========================================================
    # STAGE 11 — RECOVER BROWN / DAMAGED TISSUE
    # ========================================================

    brown_mask = cv2.inRange(
        hsv,
        np.array(
            [3, 55, 25],
            dtype=np.uint8
        ),
        np.array(
            [25, 255, 220],
            dtype=np.uint8
        )
    )

    damaged_mask = cv2.bitwise_or(
        yellow_mask,
        brown_mask
    )

    # VERY IMPORTANT:
    # Brown/yellow pixels are accepted only around the leaf.
    damaged_mask = cv2.bitwise_and(
        damaged_mask,
        roi_mask
    )

    # ========================================================
    # STAGE 12 — CANDIDATE LEAF
    # ========================================================

    candidate = cv2.bitwise_or(
        leaf_core,
        damaged_mask
    )

    candidate = cv2.bitwise_and(
        candidate,
        roi_mask
    )

    # IMPORTANT:
    #
    # DO NOT do:
    #
    # candidate = cv2.bitwise_or(candidate, hull_mask)
    #
    # because that would force background inside the convex
    # hull to become leaf tissue.

    # ========================================================
    # STAGE 13 — CONTROLLED MORPHOLOGY
    # ========================================================

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel3,
        iterations=1
    )

    # ========================================================
    # STAGE 14 — GRABCUT
    # ========================================================

    refined = _grabcut_refine(
        img_bgr,
        candidate,
        roi_mask
    )

    # ========================================================
    # STAGE 15 — CLEAN GRABCUT RESULT
    # ========================================================

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_OPEN,
        kernel3,
        iterations=1
    )

    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        kernel3,
        iterations=1
    )

    # Keep only main object
    refined = _largest_component(
        refined
    )

    refined_pixels = int(
        np.count_nonzero(refined)
    )

    # GrabCut failed
    if refined_pixels == 0:

        return (
            leaf_core,
            float(core_coverage),
            "green-core fallback — GrabCut failed"
        )

    # ========================================================
    # STAGE 16 — FINAL ACTUAL CONTOUR
    # ========================================================

    final_contours, _ = cv2.findContours(
        refined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not final_contours:

        return (
            refined,
            float(refined_pixels / image_area),
            "HSV/ExG + GrabCut segmentation"
        )

    biggest_contour = max(
        final_contours,
        key=cv2.contourArea
    )

    final_mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    # Fill actual contour.
    # NO convex hull here.
    cv2.drawContours(
        final_mask,
        [biggest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # ========================================================
    # STAGE 17 — FINAL EDGE CLEANUP
    # ========================================================

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel3,
        iterations=1
    )

    final_pixels = int(
        np.count_nonzero(final_mask)
    )

    coverage = (
        final_pixels / image_area
    )

    # ========================================================
    # STAGE 18 — SANITY CHECKS
    # ========================================================

    # Result unexpectedly small
    if coverage < 0.03:

        return (
            leaf_core,
            float(core_coverage),
            "green-core fallback — final mask too small"
        )

    # Result unexpectedly huge
    if coverage > 0.80:

        return (
            leaf_core,
            float(core_coverage),
            "green-core fallback — final mask too large"
        )

    # Compare against ROI.
    roi_coverage = (
        np.count_nonzero(roi_mask)
        / image_area
    )

    if coverage > roi_coverage:

        return (
            leaf_core,
            float(core_coverage),
            "green-core fallback — final mask exceeded ROI"
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    method = (
        "Bilateral + CLAHE + HSV/ExG core + "
        "convex-hull ROI + damaged-tissue recovery + "
        "GrabCut contour refinement"
    )

    return (
        final_mask,
        float(coverage),
        method
    )