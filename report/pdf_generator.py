import io
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY


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
        buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    CD = colors.HexColor("#0a1a0d")
    CW = colors.white

    def S(n, **k):
        return ParagraphStyle(n, **k)

    sT = S("sT", fontName="Helvetica-Bold", fontSize=18, textColor=CW,
           alignment=TA_CENTER, spaceAfter=4)
    sH = S("sH", fontName="Helvetica-Bold", fontSize=12, textColor=CD,
           spaceBefore=10, spaceAfter=4)
    sB = S("sB", fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#1f2937"),
           leading=15, spaceAfter=4, alignment=TA_JUSTIFY)
    sBul = S("sBl", fontName="Helvetica", fontSize=9.5, textColor=colors.HexColor("#1f2937"),
             leading=14, leftIndent=12, spaceAfter=3)
    sSm = S("sSm", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER)
    story = []
    ts = datetime.now()

    def HR():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bbf7d0"))

    # Header
    ht = Table([[Paragraph("TULSI LEAF HEALTH ANALYSIS REPORT", sT)]], colWidths=[17 * cm])
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CD),
        ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.extend([ht, Spacer(1, 0.3 * cm)])

    meta = [
        ["Sample ID", sample_id, "Date", ts.strftime("%d %b %Y")],
        ["Patient/User", patient_name, "Time", ts.strftime("%H:%M:%S")],
        ["Application", use_case, "Analyst", "AI/Rule-Based System"],
        ["Status", Paragraph(f'<font color="{"#166534" if classification["is_healthy"] else "#991b1b"}"><b>{classification["status"]}</b></font>', sB),
         "Confidence", f'{classification["confidence"]}%'],
        ["Leaf Coverage", f'{metrics.get("_leaf_coverage",1)*100:.1f}%',
         "Seg. Method", metrics.get("_seg_method", "N/A")],
    ]
    mt = Table(meta, colWidths=[3.5 * cm, 5 * cm, 3 * cm, 5 * cm])
    mt.setStyle(TableStyle([
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
    ]))
    story.extend([mt, Spacer(1, 0.4 * cm)])

    # Banner
    b_bg = colors.HexColor("#dcfce7") if classification["is_healthy"] else colors.HexColor("#fee2e2")
    b_tc = colors.HexColor("#166534") if classification["is_healthy"] else colors.HexColor("#991b1b")
    sev = "" if classification["is_healthy"] else f"  |  Severity: {classification['severity']}"
    gd = insights.get("quality_grade", "N/A").split("--")[0].strip().rstrip("-")
    bd = [[Paragraph(
        f'Status: <b>{classification["status"]}</b>{sev}  |  Confidence:{classification["confidence"]}%  |  Quality:{gd}',
        ParagraphStyle("bn", fontName="Helvetica-Bold", fontSize=11, textColor=b_tc, alignment=TA_CENTER),
    )]]
    bt = Table(bd, colWidths=[17 * cm])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), b_bg),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.extend([bt, Spacer(1, 0.4 * cm)])

    # Segmentation image
    seg_data = vis_cache.get("segmentation")
    if seg_data:
        story.append(Paragraph("Leaf Segmentation (Background Excluded)", sH))
        story.append(Paragraph(
            f"Metrics computed on leaf tissue only ({metrics.get('_leaf_coverage',1)*100:.1f}% of image). Background excluded from all calculations.",
            sB,
        ))
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
    rb = RLImage(io.BytesIO(rb_data), width=5.5 * cm, height=5.5 * cm) if rb_data else ri
    it = Table([[ri, rb]], colWidths=[7 * cm, 10 * cm])
    it.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.extend([it, Spacer(1, 0.3 * cm), HR()])

    # Parameters table
    story.extend([Spacer(1, 0.3 * cm), Paragraph("Quantitative Parameters (Leaf Tissue Only)", sH)])
    story.append(Paragraph("All values below are computed exclusively on segmented leaf pixels, not background.", sB))
    pr = [["Parameter", "Measured Value", "Normal Range (leaf)", "Status"]]
    from config import THRESHOLDS
    for key in THRESHOLDS:
        val = metrics.get(key)
        info = THRESHOLDS[key]
        p = classification["param_results"].get(key, False)
        if val is None:
            continue
        ref = f"<= {info['max']}" if key == "yellow_brown_ratio" else f"{info['min']} - {info['max']}"
        pr.append([
            info["label"], f"{val}", ref,
            Paragraph(f'<font color="{"#166534" if p else "#991b1b"}"><b>{"✓ Pass" if p else "✗ Fail"}</b></font>', sB),
        ])
    pt = Table(pr, colWidths=[6 * cm, 2.8 * cm, 4.2 * cm, 3.5 * cm])
    pt.setStyle(TableStyle([
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
    ]))
    story.extend([pt, Spacer(1, 0.4 * cm), HR()])

    # Classification breakdown
    story.extend([Spacer(1, 0.3 * cm), Paragraph("Classification Scoring Breakdown", sH)])
    sp = classification["weighted_score"] / classification["max_weighted"] if classification["max_weighted"] > 0 else 0
    story.append(Paragraph(
        f"Weighted Score: <b>{classification['weighted_score']}/{classification['max_weighted']}</b> ({sp*100:.1f}%) | "
        f"Params Passed: <b>{classification['passed_count']}/{classification['total_params']}</b> | Threshold: 55%",
        sB,
    ))
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
            story.extend([
                Spacer(1, 0.2 * cm),
                Paragraph(f"<b>{vt}</b>", sB),
                RLImage(io.BytesIO(vd), width=14 * cm),
                Spacer(1, 0.2 * cm),
            ])
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
    sb_c = (colors.HexColor("#dcfce7") if "SAFE" in sf else
            colors.HexColor("#fef3c7") if "CAUTION" in sf else
            colors.HexColor("#fee2e2"))
    sc_c = (colors.HexColor("#166534") if "SAFE" in sf else
            colors.HexColor("#92400e") if "CAUTION" in sf else
            colors.HexColor("#991b1b"))
    gd2 = insights.get("quality_grade", "N/A").split("--")[0].strip().rstrip("-")
    sfd = [[Paragraph(
        f"Safety: <b>{sf}</b>  |  Quality: <b>{gd2}</b>",
        ParagraphStyle("sf", fontName="Helvetica-Bold", fontSize=10, textColor=sc_c, alignment=TA_CENTER),
    )]]
    sft = Table(sfd, colWidths=[17 * cm])
    sft.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), sb_c),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.extend([sft, Spacer(1, 0.3 * cm), HR(), Spacer(1, 0.2 * cm)])
    story.append(Paragraph(
        f"Tulsi Leaf Health Analyzer(Leaf-Segmented Analysis)  |  {ts.strftime('%d %b %Y, %H:%M')}  |  For research and medical advisory use only.",
        sSm,
    ))
    doc.build(story)
    return buf.getvalue()