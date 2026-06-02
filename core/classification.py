from config import THRESHOLDS, CLASSIFICATION_WEIGHTS


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