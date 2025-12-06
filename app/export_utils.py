"""
Export utilities for generating CSV and PDF reports.
Provides batch export, individual report export, and analytics summaries.
"""
import io
import csv
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfgen import canvas


def export_predictions_csv(predictions):
    """
    Export predictions to CSV format.
    
    Args:
        predictions: List of Prediction objects or dictionaries
    
    Returns:
        CSV data as string
    """
    output = io.StringIO()
    
    # Define CSV columns
    fieldnames = [
        'PDL ID', 'Name', 'Age', 'Gender', 'Assessment Date',
        'Risk Level', 'Probability', 'Prior Convictions', 'Offense Type',
        'Substance Abuse', 'Mental Health', 'Family Support', 'Gang Affiliation'
    ]
    
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for pred in predictions:
        # Handle both dict and object types
        if hasattr(pred, 'to_dict'):
            pred = pred.to_dict()
        
        writer.writerow({
            'PDL ID': pred.get('pdl_id', 'N/A'),
            'Name': pred.get('name', 'Unknown'),
            'Age': pred.get('age', 'N/A'),
            'Gender': pred.get('gender', 'N/A'),
            'Assessment Date': pred.get('timestamp', 'N/A'),
            'Risk Level': pred.get('risk_level', 'N/A'),
            'Probability': f"{pred.get('probability', 0) * 100:.1f}%" if pred.get('probability') else 'N/A',
            'Prior Convictions': pred.get('prior_convictions', 'N/A'),
            'Offense Type': pred.get('offense_type', 'N/A'),
            'Substance Abuse': pred.get('substance_abuse', 'N/A'),
            'Mental Health': pred.get('mental_health', 'N/A'),
            'Family Support': pred.get('family_support', 'N/A'),
            'Gang Affiliation': pred.get('gang_affiliation', 'N/A')
        })
    
    return output.getvalue()


def generate_pdf_report(prediction_data, rehabilitation_plan=None):
    """
    Generate a PDF report for a single assessment.
    
    Args:
        prediction_data: Dictionary containing prediction information
        rehabilitation_plan: List of rehabilitation plan items
    
    Returns:
        PDF bytes buffer
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Container for PDF elements
    elements = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#10069F'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#10069F'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    title = Paragraph("BJMP Recidivism Risk Assessment Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.3*inch))
    
    # Assessment Info
    info_data = [
        ['Report Date:', datetime.now().strftime('%B %d, %Y')],
        ['PDL ID:', prediction_data.get('pdl_id', 'N/A')],
        ['Name:', prediction_data.get('name', 'Unknown')],
        ['Assessment Date:', prediction_data.get('timestamp', 'N/A')],
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Risk Assessment Result
    elements.append(Paragraph("Risk Assessment Result", heading_style))
    
    risk_level = prediction_data.get('risk_level', 'Unknown').upper()
    probability = prediction_data.get('probability', 0)
    
# Risk color mapping
    risk_colors = {
        'HIGH': colors.HexColor('#E4002B'),
        'MEDIUM': colors.HexColor('#FFCD00'),
        'LOW': colors.HexColor('#059669')
    }
    
    risk_color = risk_colors.get(risk_level, colors.grey)
    
    risk_data = [
        ['Risk Level:', risk_level],
        ['Probability:', f'{int(probability * 100)}%'],
        ['Model Version:', prediction_data.get('model_version', '1.0')]
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 4*inch])
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TEXTCOLOR', (1, 0), (1, 0), risk_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#DDDDDD')),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Demographic Information
    elements.append(Paragraph("Demographic Information", heading_style))
    demo_data = [
        ['Age:', str(prediction_data.get('age', 'N/A'))],
        ['Gender:', prediction_data.get('gender', 'N/A')],
        ['Civil Status:', prediction_data.get('civil_status', 'N/A')],
        ['Education:', prediction_data.get('education', 'N/A')],
        ['Employment:', prediction_data.get('employment', 'N/A')],
    ]
    demo_table = Table(demo_data, colWidths=[2*inch, 4*inch])
    demo_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(demo_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Criminal History
    elements.append(Paragraph("Criminal History", heading_style))
    criminal_data = [
        ['Prior Convictions:', str(prediction_data.get('prior_convictions', 'N/A'))],
        ['Offense Type:', prediction_data.get('offense_type', 'N/A')],
        ['Sentence Length:', f"{prediction_data.get('sentence_length', 0)} years"],
        ['Time Served:', f"{prediction_data.get('time_served', 0)} years"],
    ]
    criminal_table = Table(criminal_data, colWidths=[2*inch, 4*inch])
    criminal_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(criminal_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Risk Factors
    elements.append(Paragraph("Risk Factors", heading_style))
    factors_data = [
        ['Substance Abuse:', prediction_data.get('substance_abuse', 'N/A')],
        ['Mental Health Issues:', prediction_data.get('mental_health', 'N/A')],
        ['Family Support:', prediction_data.get('family_support', 'N/A')],
        ['Gang Affiliation:', prediction_data.get('gang_affiliation', 'N/A')],
        ['Program Participation:', prediction_data.get('program_participation', 'N/A')],
    ]
    factors_table = Table(factors_data, colWidths=[2*inch, 4*inch])
    factors_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#666666')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    elements.append(factors_table)
    
    # Rehabilitation Plan
    if rehabilitation_plan and len(rehabilitation_plan) > 0:
        elements.append(Spacer(1, 0.3*inch))
        elements.append(Paragraph("Recommended Rehabilitation Plan", heading_style))
        
        for i, item in enumerate(rehabilitation_plan, 1):
            plan_text = f"{i}. {item}"
            elements.append(Paragraph(plan_text, styles['Normal']))
            elements.append(Spacer(1, 0.1*inch))
    
    # Footer
    elements.append(Spacer(1, 0.5*inch))
    footer_text = f"<i>This report was generated by the BJMP Recidivism Prediction System on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}. " \
                  "This assessment is for guidance purposes only and should be used in conjunction with professional judgment.</i>"
    elements.append(Paragraph(footer_text, styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    # Get PDF bytes
    pdf_bytes = buffer.getvalue()
    buffer.close()
    
    return pdf_bytes


def export_analytics_summary(statistics):
    """
    Export analytics summary as CSV.
    
    Args:
        statistics: Dictionary containing analytics data
    
    Returns:
        CSV data as string
    """
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['BJMP Recidivism Prediction System - Analytics Summary'])
    writer.writerow(['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')])
    writer.writerow([])
    
    # Overall Statistics
    writer.writerow(['Overall Statistics'])
    writer.writerow(['Total Assessments', statistics.get('total', 0)])
    writer.writerow(['Low Risk', statistics.get('low', {}).get('count', 0)])
    writer.writerow(['Medium Risk', statistics.get('medium', {}).get('count', 0)])
    writer.writerow(['High Risk', statistics.get('high', {}).get('count', 0)])
    writer.writerow([])
    
    # Risk Distribution
    writer.writerow(['Risk Distribution'])
    total = statistics.get(' total', 1)
    writer.writerow(['Low Risk %', f"{(statistics.get('low', {}).get('count', 0) / total * 100):.1f}%"])
    writer.writerow(['Medium Risk %', f"{(statistics.get('medium', {}).get('count', 0) / total * 100):.1f}%"])
    writer.writerow(['High Risk %', f"{(statistics.get('high', {}).get('count', 0) / total * 100):.1f}%"])
    
    return output.getvalue()
