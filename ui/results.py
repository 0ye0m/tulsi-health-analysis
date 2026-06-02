import streamlit as st
import json
from datetime import datetime
from report.pdf_generator import generate_pdf


def render_results(
    metrics,
    classification,
    insights,
    vis_cache,
    pil_img,
    use_case,
    patient_name,
    sample_id,
    coverage,
    seg_method,
    use_offline,
):
    """Render the right column (col2) – all result tabs."""
    # Segmentation warning / success
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
        f"🧠 {mode_lbl}</span>&nbsp;&nbsp;"
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
    sf_c = "#4ade80" if "SAFE" in sf else "#fbbf24" if "CAUTION" in sf else "#f87171"
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

    # Tab 0: Parameters
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
        from config import THRESHOLDS, CLASSIFICATION_WEIGHTS

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
            '<div class="section-title">Segmentation Info</div>', unsafe_allow_html=True
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

    # Tab 1: Insights
    with tabs[1]:
        for title, key, border in [
            ("Clinical Summary", "clinical_summary", "#22c55e"),
            ("Detailed Pathology", "detailed_pathology", "#f87171"),
            ("Medical Relevance", "medical_relevance", "#a78bfa"),
            ("Phytochemical Analysis", "phytochemical_note", "#f59e0b"),
            ("Treatment Protocol", "treatment_protocol", "#c084fc"),
            ("Environmental Factors", "environmental_factors", "#2dd4bf"),
            ("Pharmacopoeial Compliance", "pharmacopoeial_compliance", "#818cf8"),
        ]:
            st.markdown(
                f'<div class="section-title">{title}</div>', unsafe_allow_html=True
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
            '<div class="section-title">Recommendations</div>', unsafe_allow_html=True
        )
        for r in insights.get("recommendations", []):
            st.markdown(
                f'<div class="insight-box" style="border-left-color:#34d399">▶ {r}</div>',
                unsafe_allow_html=True,
            )

    # Tab 2: Pipeline
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
            chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
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
                    vis_cache["classification_breakdown"], use_container_width=True
                )

    # Tab 3: Visual Analysis
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
                    f'<div class="section-title">{vtitle}</div>', unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="insight-box" style="font-size:0.87rem">{vdesc}</div>',
                    unsafe_allow_html=True,
                )
                st.image(vdata, use_container_width=True)

    # Tab 4: Charts
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
            st.image(vis_cache["classification_breakdown"], use_container_width=True)

    # Tab 5: Report
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
                "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
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
            st.success(f"✅ Report ready! {len(vis_cache)} visualisations included.")
        except Exception as e:
            st.error(f"PDF error: {e}")
