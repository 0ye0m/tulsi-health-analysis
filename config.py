# ═══════════════════════════════════════════════════════════════════════════════
# Constants & Thresholds – calibrated for LEAF‑ONLY pixels
# ═══════════════════════════════════════════════════════════════════════════════

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
    "green_dominance": 4,
    "yellow_brown_ratio": 4,
    "hsv_hue_green_ratio": 3,
    "green_channel_ratio": 2,
    "hsv_saturation": 2,
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