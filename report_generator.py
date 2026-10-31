import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_clinical_report_pdf(patient_data: dict, analysis_result: dict) -> bytes:
    """
    Generates an EDUCATIONAL DEMO PDF (not a clinical report; not for clinical use)
    combining Complete Blood Count (CBC), AI Support Vector Machine (RBF) Classification,
    SHAP Feature Attribution Explainability, and Classical Discrimination Indices Concordance.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#0f172a')     # Slate 900
    c_secondary = colors.HexColor('#1e40af')   # Blue 800
    c_accent = colors.HexColor('#0284c7')      # Sky 600
    c_text = colors.HexColor('#334155')        # Slate 700
    c_border = colors.HexColor('#cbd5e1')      # Slate 300
    c_bg_light = colors.HexColor('#f8fafc')    # Slate 50
    c_thal = colors.HexColor('#b91c1c')        # Red 700
    c_ida = colors.HexColor('#047857')         # Emerald 700
    
    # Typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=15,
        leading=18,
        textColor=c_primary,
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=11,
        textColor=c_accent,
        fontName='Helvetica-Bold'
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=10.5,
        leading=13,
        textColor=c_secondary,
        fontName='Helvetica-Bold',
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=8,
        leading=10.5,
        textColor=c_text,
        fontName='Helvetica'
    )
    body_bold = ParagraphStyle(
        'DocBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=c_primary,
        fontName='Helvetica'
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell,
        fontName='Helvetica-Bold'
    )
    badge_thal = ParagraphStyle(
        'BadgeThal',
        parent=table_cell_bold,
        textColor=c_thal
    )
    badge_ida = ParagraphStyle(
        'BadgeIDA',
        parent=table_cell_bold,
        textColor=c_ida
    )

    story = []

    # --- HEADER / CLINICAL LOGO ---
    header_data = [
        [
            Paragraph("<b>THALA-AI PRO - EDUCATIONAL DEMO OUTPUT</b><br/>"
                      "<font size='7.5' color='#64748b'>Educational research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation.</font>", title_style),
            Paragraph(f"<b>DEMO ID:</b> {patient_data.get('patient_id', 'DEMO-INPUT')}<br/>"
                      f"<b>DATE:</b> {datetime.now().strftime('%d-%b-%Y %H:%M')}<br/>"
                      f"<b>STATUS:</b> <font color='#b91c1c'><b>NOT FOR CLINICAL USE</b></font>", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_secondary, spaceBefore=4, spaceAfter=8))

    # --- PATIENT DEMOGRAPHICS ---
    patient_info = [
        [
            Paragraph("<b>Input Label:</b>", body_bold),
            Paragraph(str(patient_data.get('patient_name', 'Demo input (not a patient)')), body_style),
            Paragraph("<b>Demo ID:</b>", body_bold),
            Paragraph(str(patient_data.get('patient_id', 'DEMO-INPUT')), body_style),
            Paragraph("<b>Input Source:</b>", body_bold),
            Paragraph("Values entered manually (demo)", body_style),
        ],
        [
            Paragraph("<b>Age / Gender:</b>", body_bold),
            Paragraph(f"{patient_data.get('age', 'N/A')} / {patient_data.get('gender', 'N/A')}", body_style),
            Paragraph("<b>Training Data:</b>", body_bold),
            Paragraph("Synthetic (Google Gemini)", body_style),
            Paragraph("<b>Generated:</b>", body_bold),
            Paragraph(datetime.now().strftime('%d-%b-%Y'), body_style),
        ]
    ]
    p_table = Table(patient_info, colWidths=[75, 105, 70, 110, 75, 105])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg_light),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 8))

    # --- AI DIAGNOSTIC EXECUTIVE SUMMARY ---
    pred = analysis_result.get('prediction', {})
    is_thal = pred.get('predicted_class') == 1
    prob_thal = pred.get('prob_thalassemia', 0.5) * 100
    prob_ida = pred.get('prob_ida', 0.5) * 100
    confidence = pred.get('confidence_tier', 'High')
    verdict = "BETA THALASSEMIA TRAIT / CARRIER (HIGH RISK)" if is_thal else "IRON DEFICIENCY ANEMIA / NON-THALASSEMIA PROFILE"
    verdict_color = c_thal if is_thal else c_ida

    summary_box_data = [
        [
            Paragraph(f"<b>AI CLASSIFIER EVALUATION:</b> <font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", ParagraphStyle('SumH', parent=body_bold, fontSize=9.5, textColor=c_primary)),
            Paragraph(f"<b>Thalassemia Probability:</b> {prob_thal:.1f}% &nbsp;|&nbsp; <b>IDA Probability:</b> {prob_ida:.1f}% &nbsp;|&nbsp; <b>Confidence:</b> {confidence}", body_style)
        ]
    ]
    s_table = Table(summary_box_data, colWidths=[270, 270])
    s_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fef2f2') if is_thal else colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 1, verdict_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(s_table)
    story.append(Spacer(1, 6))

    # Optional: Combined IDA & Beta Thalassemia Trait Alert Box
    comb_risk = analysis_result.get('combined_risk', {})
    if comb_risk.get('detected', False):
        comb_box_data = [
            [
                Paragraph("<b><font color='#b45309'>⚠️ CLINICAL RISK ALERT: SUSPECTED COMBINED IDA & BETA THALASSEMIA TRAIT (DUAL PATHOLOGY)</font></b><br/>"
                          f"<font size='7.5' color='#78350f'>{comb_risk.get('detail', '')}<br/>"
                          "<b>Diagnostic Pitfall:</b> Co-existing iron deficiency suppresses HbA₂ synthesis, potentially masking Beta Thalassemia Trait on HPLC (giving false normal HbA₂ &lt; 3.5%). Measure Serum Ferritin; re-evaluate HPLC after therapeutic iron repletion.</font>",
                          body_style)
            ]
        ]
        c_table = Table(comb_box_data, colWidths=[540])
        c_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffbeb')),
            ('BOX', (0, 0), (-1, -1), 1.2, colors.HexColor('#d97706')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(c_table)
        story.append(Spacer(1, 6))

    # --- COMPLETE BLOOD COUNT (CBC) & RBC INDICES ---
    story.append(Paragraph("1. AUTOMATED HEMATOLOGY PARAMETERS (COMPLETE BLOOD COUNT)", section_heading))
    cbc_rows = [
        ["Parameter", "Result", "Biological Ref Range", "Units", "Clinical Significance", "Flag"]
    ]
    cbc_list = analysis_result.get('cbc_evaluated', [])
    for row in cbc_list:
        flag = row.get('flag', 'Normal')
        flag_p = Paragraph(f"<b><font color='{'#b91c1c' if 'L' in flag or 'H' in flag else '#059669'}'>{flag}</font></b>", table_cell_bold)
        cbc_rows.append([
            Paragraph(f"<b>{row.get('name')}</b>", table_cell),
            Paragraph(f"<b>{row.get('value')}</b>", table_cell),
            Paragraph(row.get('ref_range', ''), table_cell),
            Paragraph(row.get('unit', ''), table_cell),
            Paragraph(row.get('comment', ''), table_cell),
            flag_p
        ])

    cbc_table = Table(cbc_rows, colWidths=[100, 55, 95, 55, 185, 50])
    cbc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_secondary),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_light]),
    ]))
    story.append(cbc_table)
    story.append(Spacer(1, 8))

    # --- CLASSICAL DISCRIMINATION INDICES CONCORDANCE ---
    story.append(Paragraph("2. CONCORDANCE WITH CLASSICAL DISCRIMINANT FORMULAS", section_heading))
    idx_summary = analysis_result.get('concordance_summary', {})
    agreement_count = idx_summary.get('agreement_count', 0)
    total_indices = idx_summary.get('total_indices', 8)
    agreement_pct = idx_summary.get('agreement_pct', 0)
    
    story.append(Paragraph(
        f"Consensus Analysis: <b>{agreement_count} of {total_indices} Classical Indices ({agreement_pct:.1f}%)</b> "
        f"concur with the AI Classifier prediction. Classical single-ratio formulas frequently suffer from cutoff vulnerability in borderline or combined cases.",
        body_style
    ))
    story.append(Spacer(1, 4))

    idx_rows = [
        ["Formula / Index", "Calculation Equation", "Observed Value", "Cutoff Rule", "Index Verdict", "AI Agreement"]
    ]
    for idx_item in analysis_result.get('indices', []):
        agree = idx_item.get('agrees_with_ai', True)
        agree_p = Paragraph(f"<b><font color='{'#059669' if agree else '#d97706'}'>{'Concordant' if agree else 'Discordant'}</font></b>", table_cell_bold)
        verdict_p = Paragraph(f"<b>{idx_item.get('interpretation')}</b>", badge_thal if 'Thal' in idx_item.get('interpretation') else badge_ida)
        idx_rows.append([
            Paragraph(f"<b>{idx_item.get('name')}</b>", table_cell),
            Paragraph(idx_item.get('formula', ''), table_cell),
            Paragraph(f"<b>{idx_item.get('value'):.2f}</b>", table_cell),
            Paragraph(idx_item.get('cutoff_rule', ''), table_cell),
            verdict_p,
            agree_p
        ])

    idx_table = Table(idx_rows, colWidths=[95, 125, 65, 110, 85, 60])
    idx_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_light]),
    ]))
    story.append(idx_table)
    story.append(Spacer(1, 8))

    # --- SHAP EXPLAINABLE AI ATTRIBUTIONS ---
    story.append(Paragraph("3. EXPLAINABLE AI: PATIENT-SPECIFIC SHAP FEATURE CONTRIBUTIONS", section_heading))
    base_val = analysis_result.get('shap_analysis', {}).get('base_value', 0.461)
    story.append(Paragraph(
        f"Population Expected Base Risk: <b>{base_val*100:.1f}%</b>. The Support Vector Machine (RBF) model leverages calibrated non-linear dual decision kernels to quantify the exact individual contribution (&phi;) of each blood parameter towards the final diagnostic probability:",
        body_style
    ))
    story.append(Spacer(1, 4))

    shap_rows = [
        ["Rank", "Feature Parameter", "Observed Value", "SHAP Impact (&phi;)", "Directional Effect", "Clinical Rationale"]
    ]
    shap_list = analysis_result.get('shap_analysis', {}).get('features', [])
    for s in shap_list:
        phi = s.get('shap_value', 0.0)
        direction = "Favors Thalassemia" if phi > 0 else "Favors IDA / Non-Thal"
        dir_color = '#b91c1c' if phi > 0 else '#047857'
        dir_p = Paragraph(f"<b><font color='{dir_color}'>{direction}</font></b>", table_cell_bold)
        shap_rows.append([
            Paragraph(f"#{s.get('rank')}", table_cell),
            Paragraph(f"<b>{s.get('name')}</b>", table_cell),
            Paragraph(f"{s.get('actual_value')}", table_cell),
            Paragraph(f"<b>{'+' if phi > 0 else ''}{phi:.4f}</b>", table_cell),
            dir_p,
            Paragraph(s.get('explanation', ''), table_cell)
        ])

    shap_table = Table(shap_rows, colWidths=[30, 95, 60, 70, 95, 190])
    shap_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_bg_light]),
    ]))
    story.append(shap_table)
    story.append(Spacer(1, 8))

    # --- CLINICAL INTERPRETATION & RECOMMENDATIONS ---
    story.append(Paragraph("4. CLINICAL RECOMMENDATIONS & PROTOCOL TRIAGE", section_heading))
    recs = analysis_result.get('recommendations', [])
    rec_texts = "<br/>".join([f"&bull; <b>{r.get('title')}:</b> {r.get('desc')}" for r in recs])
    rec_box = [
        [
            Paragraph(
                f"<font size='7.5' color='#1e293b'>{rec_texts}</font><br/><br/>"
                f"<font size='7' color='#64748b'><b>Disclaimer:</b> Educational research prototype trained on synthetic data generated with Google Gemini. Not a medical device; not for clinical use; no laboratory accreditation. Model performance (97.7% accuracy, 96.0% BTT sensitivity, 99.2% IDA specificity, ROC-AUC 0.994) was measured on a held-out synthetic test set of 220 records only. Definitive diagnosis of hemoglobinopathies requires quantitative High-Performance Liquid Chromatography (HPLC) / Capillary Electrophoresis and molecular genetic testing.</font>",
                body_style
            )
        ]
    ]
    r_table = Table(rec_box, colWidths=[540])
    r_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(r_table)
    story.append(Spacer(1, 10))

    # --- SIGNATURE BLOCK ---
    sig_data = [
        [
            Paragraph("<b>Generated By:</b><br/>THALA-AI PRO educational demo<br/>Model: SVM (RBF kernel), 7 inputs, trained on synthetic data", table_cell),
            Paragraph("<b>Status:</b><br/>Not reviewed or validated for clinical use.<br/>Not a medical device; no laboratory accreditation.", table_cell)
        ]
    ]
    sig_table = Table(sig_data, colWidths=[270, 270])
    sig_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(KeepTogether(sig_table))

    doc.build(story)
    return buffer.getvalue()
