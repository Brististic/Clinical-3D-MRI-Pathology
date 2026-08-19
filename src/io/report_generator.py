import io
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_clinical_pdf(patient_id, slice_idx, dice_score, iou_score, pred_vol, gt_vol, fig_matplotlib):
    """
    Generates a structured clinical PDF report in-memory using ReportLab.
    Returns bytes buffer ready for browser download.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold'
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0284c7'),
        fontName='Helvetica-Bold',
        spaceBefore=10,
        spaceAfter=6
    )

    normal_text = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # 1. Header & Branding
    story.append(Paragraph("NEURODELINEATE — CLINICAL DIAGNOSTIC REPORT", title_style))
    story.append(Paragraph("Automated Volumetric Brain Pathology Segmentation Suite", normal_text))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=12))

    # 2. Patient & Exam Metadata Table
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        [Paragraph("<b>Patient Identifier:</b>", normal_text), Paragraph(patient_id, normal_text),
         Paragraph("<b>Generated Date:</b>", normal_text), Paragraph(timestamp, normal_text)],
        [Paragraph("<b>Primary Modality:</b>", normal_text), Paragraph("MRI (FLAIR / Multi-contrast)", normal_text),
         Paragraph("<b>Key Slice Index:</b>", normal_text), Paragraph(f"Slice #{slice_idx}", normal_text)]
    ]
    
    meta_table = Table(meta_data, colWidths=[120, 145, 110, 155])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # 3. Quantitative Measurements Table
    story.append(Paragraph("Quantitative Volumetric Findings", section_heading))
    
    vol_err = abs(pred_vol - gt_vol) if gt_vol > 0 else 0.0
    err_pct = (vol_err / gt_vol * 100) if gt_vol > 0 else 0.0

    metrics_data = [
        ["Diagnostic Parameter", "Measured Value", "Reference (Ground Truth)", "Clinical Status"],
        ["Predicted Tumor Volume", f"{pred_vol:.2f} cm³", f"{gt_vol:.2f} cm³", f"Error: {vol_err:.2f} cm³ ({err_pct:.1f}%)"],
        ["Dice Similarity (DSC)", f"{dice_score:.4f}", "1.0000", "High Boundary Overlap" if dice_score > 0.7 else "Moderate Overlap"],
        ["Jaccard Index (IoU)", f"{iou_score:.4f}", "1.0000", "Concordant" if iou_score > 0.6 else "Review Required"],
    ]

    metrics_table = Table(metrics_data, colWidths=[160, 110, 130, 130])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#ffffff'), colors.HexColor('#f1f5f9')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 12))

    # 4. Multi-Panel Visual Findings (Render Matplotlib figure directly into PDF)
    story.append(Paragraph("Visual Pathology Delineation", section_heading))
    
    img_buf = io.BytesIO()
    fig_matplotlib.savefig(img_buf, format='png', dpi=200, bbox_inches='tight')
    img_buf.seek(0)
    
    report_img = RLImage(img_buf, width=530, height=170)
    story.append(report_img)
    story.append(Spacer(1, 15))

    # 5. Diagnostic Impression & Disclaimer
    story.append(Paragraph("Algorithmic Impression & Notes", section_heading))
    notes = (
        "Segmentation completed using adaptive statistical homogeneity region growing and morphological boundary smoothing. "
        "Calculated voxel dimensions derived from primary NIfTI coordinate affine matrices. "
        "<b>Disclaimer:</b> For experimental DIP research validation only. Not certified as a primary standalone diagnostic device."
    )
    story.append(Paragraph(notes, normal_text))

    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer