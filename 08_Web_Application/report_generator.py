"""
Diagnostic Pathology Report Generator
Generates clinical agronomic reports in both PDF (via ReportLab with HTML/text fallback) and Markdown formats.
Contains timestamp, patient ID, crop taxonomy, prediction metrics, and chemical/organic prescriptions.
"""

import os
import io
import datetime

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

def generate_pdf_report(crop_name: str, result_dict: dict, knowledge_dict: dict):
    """
    Generates a pathology PDF report binary buffer.
    If reportlab is not installed, generates a formatted text/HTML report bytes fallback.
    """
    if not HAS_REPORTLAB:
        # Fallback to rich UTF-8 encoded text / markdown report buffer
        md_content = generate_markdown_report(crop_name, result_dict, knowledge_dict)
        return md_content.encode("utf-8")

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
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontSize=20, leading=24,
        textColor=colors.HexColor('#0F172A'), fontName='Helvetica-Bold', alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'], fontSize=10, leading=14,
        textColor=colors.HexColor('#64748B'), alignment=1
    )
    h2_style = ParagraphStyle(
        'H2Style', parent=styles['Heading2'], fontSize=13, leading=17,
        textColor=colors.HexColor('#1E293B'), fontName='Helvetica-Bold',
        spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'], fontSize=9.5, leading=13.5,
        textColor=colors.HexColor('#334155')
    )
    bullet_style = ParagraphStyle(
        'BulletText', parent=styles['Normal'], fontSize=9, leading=13,
        textColor=colors.HexColor('#1E293B'), leftIndent=12, firstLineIndent=-8
    )

    story = []
    story.append(Paragraph("AGROVISION DIAGNOSTICS & PATHOLOGY LABORATORY", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Autonomous Computer Vision Foliar Assessment & Agronomic Advisory Report", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width='100%', thickness=2, color=colors.HexColor('#10B981'), spaceAfter=14))

    diag_id = f"AGV-{datetime.datetime.now().strftime('%Y%m%d')}-{abs(hash(result_dict['predicted_class'])) % 10000:04d}"
    curr_time = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    pred_class = result_dict['predicted_class']
    conf = result_dict['confidence']
    severity = knowledge_dict.get('severity_level', 'Unknown')
    pathogen = knowledge_dict.get('pathogen_type', 'N/A')
    causal = knowledge_dict.get('causal_agent', 'N/A')
    sc_name = knowledge_dict.get('scientific_name', 'N/A')

    meta_data = [
        [Paragraph("<b>Diagnosis ID:</b> " + diag_id, body_style), Paragraph("<b>Timestamp:</b> " + curr_time, body_style)],
        [Paragraph(f"<b>Host Crop:</b> {crop_name}", body_style), Paragraph(f"<b>Host Scientific Name:</b> <i>{sc_name}</i>", body_style)],
        [Paragraph(f"<b>Predicted Condition:</b> <font color='#0F172A'><b>{pred_class.replace('_', ' ')}</b></font>", body_style), Paragraph(f"<b>Diagnostic Confidence:</b> <b>{conf:.2f}%</b>", body_style)],
        [Paragraph(f"<b>Severity Rating:</b> <b>{severity}</b>", body_style), Paragraph(f"<b>Pathogen Type:</b> {pathogen}", body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[270, 270])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Diagnostic Probability Spectrum", h2_style))
    prob_data = [["Class Name", "Confidence Score", "Status"]]
    for c_name, c_prob in result_dict.get("class_probs", {}).items():
        tag = "Identified Match" if c_name == pred_class else "Alternative"
        prob_data.append([c_name.replace('_', ' '), f"{c_prob:.2f}%", tag])
    prob_table = Table(prob_data, colWidths=[200, 170, 170])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Clinical Foliar Symptoms", h2_style))
    for symptom in knowledge_dict.get("visual_symptoms", []):
        story.append(Paragraph(f"• {symptom}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Ecological & Organic Interventions", h2_style))
    for org in knowledge_dict.get("organic_management", []):
        story.append(Paragraph(f"• {org}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Targeted Chemical & Fungicide Schedule", h2_style))
    for chem in knowledge_dict.get("chemical_management", []):
        story.append(Paragraph(f"• {chem}", bullet_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Agronomic Prevention & Field Hygiene", h2_style))
    for prev in knowledge_dict.get("prevention_guidelines", []):
        story.append(Paragraph(f"• {prev}", bullet_style))
    story.append(Spacer(1, 20))

    footer_text = (
        "<i>Disclaimer: This automated diagnostic report is generated using deep learning computer vision " 
        "trained on multi-source datasets (PlantVillage & Mendeley GVLiD). For large-scale industrial treatment, " 
        "always cross-verify with your local certified agricultural extension officer.</i>"
    )
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#94A3B8'), spaceAfter=8))
    story.append(Paragraph(footer_text, subtitle_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def generate_markdown_report(crop_name: str, result_dict: dict, knowledge_dict: dict):
    curr_time = datetime.datetime.now().strftime("%B %d, %Y - %I:%M %p")
    pred_class = result_dict['predicted_class']
    conf = result_dict['confidence']
    
    lines = [
        "# 🌿 AGROVISION DIAGNOSTIC PATHOLOGY REPORT",
        f"**Generated:** {curr_time}  ",
        f"**Target Crop:** {crop_name}  ",
        f"**Diagnosis:** {pred_class.replace('_', ' ')} ({conf:.2f}% Confidence)  ",
        f"**Severity:** {knowledge_dict.get('severity_level', 'N/A')}  ",
        f"**Pathogen:** {knowledge_dict.get('causal_agent', 'N/A')} ({knowledge_dict.get('pathogen_type', 'N/A')})  ",
        "",
        "---",
        "",
        "### 📊 Probability Breakdown",
    ]
    for c_name, c_prob in result_dict.get("class_probs", {}).items():
        lines.append(f"- **{c_name.replace('_', ' ')}:** {c_prob:.2f}%")
    
    lines.append("")
    lines.append("### 🔍 Foliar Symptoms")
    for s in knowledge_dict.get("visual_symptoms", []):
        lines.append(f"- {s}")
    
    lines.append("")
    lines.append("### 🌱 Organic & Biocontrol Actions")
    for o in knowledge_dict.get("organic_management", []):
        lines.append(f"- {o}")
        
    lines.append("")
    lines.append("### 🧪 Recommended Fungicides & Dosages")
    for c in knowledge_dict.get("chemical_management", []):
        lines.append(f"- {c}")
        
    lines.append("")
    lines.append("### 🛡️ Preventive Field Management")
    for p in knowledge_dict.get("prevention_guidelines", []):
        lines.append(f"- {p}")
        
    lines.append("")
    lines.append("---")
    lines.append("*AgroVision CV Multi-Plant Pathology Suite*")
    return "\n".join(lines)