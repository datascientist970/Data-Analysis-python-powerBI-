from __future__ import annotations
import os
from typing import Dict, List

from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from .config import AFFORDANCE_CLASSES


def _format_class_summary(labels_present: List[int], mean_conf: float) -> str:
    names = [AFFORDANCE_CLASSES[i] for i in labels_present]
    return f"Affordances: {', '.join(names)} | Mean confidence: {mean_conf:.2f}"


def generate_report(
    image_basename: str,
    out_dir: str,
    composite_png_path: str,
    sup_labels: 'np.ndarray',
    sup_conf: 'np.ndarray',
    fused_labels: 'np.ndarray',
    fused_conf: 'np.ndarray',
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, f"{image_basename}_report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # Title
    c.setFont('Helvetica-Bold', 16)
    c.drawString(40, height - 40, 'Hybrid Affordance Detection Report')
    c.setFont('Helvetica', 11)
    c.drawString(40, height - 60, f"Image: {image_basename}")

    # Composite image
    img = ImageReader(composite_png_path)
    # Fit composite within the page
    target_w = width - 80
    target_h = height - 200
    c.drawImage(img, 40, 120, width=target_w, height=target_h, preserveAspectRatio=True, mask='auto')

    # Summaries
    import numpy as np
    sup_present = sorted(list(set(np.unique(sup_labels)) - {0}))
    fused_present = sorted(list(set(np.unique(fused_labels)) - {0}))
    sup_summary = _format_class_summary(sup_present, float(np.mean(sup_conf)))
    fused_summary = _format_class_summary(fused_present, float(np.mean(fused_conf)))

    c.setFont('Helvetica', 11)
    c.drawString(40, 90, f"Supervised: {sup_summary}")
    c.drawString(40, 70, f"Hybrid: {fused_summary}")

    c.showPage()
    c.save()
    return pdf_path
