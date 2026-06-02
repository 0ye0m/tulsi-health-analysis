import json
import requests
from config import INSIGHT_KEYS


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
        except Exception:
            return get_rulebased_insights(metrics, classification, use_case)

        fallback = get_rulebased_insights(metrics, classification, use_case)
        for k in INSIGHT_KEYS:
            if k not in parsed or not parsed[k]:
                parsed[k] = fallback[k]
        return parsed
    except Exception:
        fallback = get_rulebased_insights(metrics, classification, use_case)
        fallback["clinical_summary"] = (
            f"[AI fallback] " + fallback["clinical_summary"]
        )
        return fallback