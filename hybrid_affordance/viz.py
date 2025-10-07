from __future__ import annotations
from typing import Dict, List, Tuple
import os
import numpy as np
import matplotlib.pyplot as plt

from .config import AFFORDANCE_CLASSES, AFFORDANCE_COLORS


def get_affordance_names() -> List[str]:
    return AFFORDANCE_CLASSES


def get_affordance_palette() -> Dict[int, Tuple[int, int, int]]:
    return AFFORDANCE_COLORS


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    palette = get_affordance_palette()
    for k, color in palette.items():
        rgb[labels == k] = np.array(color, dtype=np.uint8)
    return rgb


def overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    image = image.astype(np.float32)
    mask = mask_rgb.astype(np.float32)
    comp = (1 - alpha) * image + alpha * mask
    comp = np.clip(comp, 0, 255).astype(np.uint8)
    return comp


def render_grid(
    image: np.ndarray,
    sup_labels: np.ndarray,
    unsup_labels: np.ndarray,
    fused_labels: np.ndarray,
    sup_conf: np.ndarray,
    unsup_conf: np.ndarray,
    save_path: str,
) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sup_rgb = colorize_labels(sup_labels)
    unsup_rgb = colorize_labels(unsup_labels)
    fused_rgb = colorize_labels(fused_labels)

    sup_overlay = overlay(image, sup_rgb, alpha=0.45)
    unsup_overlay = overlay(image, unsup_rgb, alpha=0.45)
    fused_overlay = overlay(image, fused_rgb, alpha=0.45)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axs = axs.ravel()
    axs[0].imshow(image)
    axs[0].set_title('Original')
    axs[0].axis('off')

    axs[1].imshow(sup_overlay)
    axs[1].set_title(f'Supervised (mean conf={float(sup_conf.mean()):.2f})')
    axs[1].axis('off')

    axs[2].imshow(unsup_overlay)
    axs[2].set_title(f'Unsupervised (mean conf={float(unsup_conf.mean()):.2f})')
    axs[2].axis('off')

    axs[3].imshow(fused_overlay)
    axs[3].set_title('Hybrid Fused')
    axs[3].axis('off')

    # Add legend
    names = get_affordance_names()
    palette = get_affordance_palette()
    # Build a small legend grid outside
    patches = []
    labels = []
    import matplotlib.patches as mpatches
    for idx, name in enumerate(names):
        color = np.array(palette[idx]) / 255.0
        patches.append(mpatches.Patch(color=color, label=name))
        labels.append(name)
    fig.legend(handles=patches, loc='lower center', ncol=min(8, len(names)))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
