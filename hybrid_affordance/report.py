from __future__ import annotations
from typing import List, Tuple
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_pdf import PdfPages

from . import INDEX_TO_CLASS, AFFORDANCE_CLASSES
from .colors import index_to_color_mask
from .utils import overlay_color_on_image, top_class_confidence_summary


def _safe_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def visualize_results(
    rgb_image: np.ndarray,
    supervised_mask: np.ndarray,
    unsupervised_mask: np.ndarray,
    final_mask: np.ndarray,
    final_conf: np.ndarray,
    probs: np.ndarray,
    class_names: List[str] = AFFORDANCE_CLASSES,
) -> Image.Image:
    rgb = Image.fromarray(rgb_image)
    sup_color = Image.fromarray(index_to_color_mask(supervised_mask))
    unsup_color = Image.fromarray(index_to_color_mask(unsupervised_mask.clip(min=0)))
    final_color = Image.fromarray(index_to_color_mask(final_mask))

    sup_overlay = overlay_color_on_image(rgb, sup_color, alpha=0.5)
    unsup_overlay = overlay_color_on_image(rgb, unsup_color, alpha=0.5)
    final_overlay = overlay_color_on_image(rgb, final_color, alpha=0.5)

    # Make a horizontal panel: [original | supervised | unsupervised | final]
    w, h = rgb.size
    panel = Image.new('RGB', (w * 4, h), color=(0, 0, 0))
    panel.paste(rgb, (0, 0))
    panel.paste(sup_overlay, (w, 0))
    panel.paste(unsup_overlay, (2 * w, 0))
    panel.paste(final_overlay, (3 * w, 0))

    draw = ImageDraw.Draw(panel)
    font = _safe_font(18)
    labels = ['Original', 'Supervised', 'Unsupervised', 'Hybrid Final']
    for i, text in enumerate(labels):
        draw.rectangle([(i * w, 0), (i * w + 120, 24)], fill=(0, 0, 0))
        draw.text((i * w + 5, 3), text, fill=(255, 255, 255), font=font)

    # Add legend on the right of final panel
    legend_x = 3 * w + 5
    legend_y = 28
    for idx, name in enumerate(class_names):
        color = index_to_color_mask(np.array([[idx]], dtype=np.int64)).reshape(3).tolist()
        draw.rectangle([(legend_x, legend_y + idx * 22), (legend_x + 18, legend_y + 18 + idx * 22)], fill=tuple(color))
        draw.text((legend_x + 24, legend_y + idx * 22), name, fill=(255, 255, 255), font=font)

    # Add top-k detected affordance summary with avg confidence on final panel
    summary = top_class_confidence_summary(probs, final_mask, class_names)
    summary = sorted(summary, key=lambda t: t[1], reverse=True)[:4]
    box_y0 = h - 100
    draw.rectangle([(3 * w, box_y0), (4 * w, h)], fill=(0, 0, 0, 180))
    y = box_y0 + 6
    for name, ratio, conf_val in summary:
        line = f"{name}: {ratio*100:.1f}% | conf {conf_val:.2f}"
        draw.text((3 * w + 8, y), line, fill=(255, 255, 255), font=font)
        y += 22

    return panel


def save_pdf_report(
    save_path: str,
    image_name: str,
    vis_image: Image.Image,
    probs: np.ndarray,
    final_mask: np.ndarray,
    class_names: List[str] = AFFORDANCE_CLASSES,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # pdf page 1: the composed visualization
    # page 2: table of class coverage and avg confidence
    summary = top_class_confidence_summary(probs, final_mask, class_names)

    with PdfPages(save_path) as pdf:
        # Page 1
        pdf.savefig(_pil_to_fig(vis_image))
        # Page 2: summary
        fig = _summary_table_figure(image_name, summary)
        pdf.savefig(fig)


def _pil_to_fig(img: Image.Image):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(img)
    ax.axis('off')
    fig.tight_layout()
    return fig


def _summary_table_figure(image_name: str, summary: List[Tuple[str, float, float]]):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    ax.set_title(f"Affordance Summary: {image_name}")
    if len(summary) == 0:
        ax.text(0.1, 0.5, 'No affordances detected', fontsize=12)
        fig.tight_layout()
        return fig
    # Build table data
    cols = ['Affordance', 'Pixel Ratio', 'Avg Confidence']
    table_data = []
    for name, ratio, conf in summary:
        table_data.append([name, f"{ratio:.3f}", f"{conf:.3f}"])

    table = ax.table(cellText=table_data, colLabels=cols, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    fig.tight_layout()
    return fig
