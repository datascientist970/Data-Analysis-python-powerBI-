from __future__ import annotations
from typing import Dict, List
import numpy as np

# RGB colors for each affordance class index
# background, grasp, cut, scoop, contain, pound, support, wrap-grasp
_CLASS_COLORS: List[List[int]] = [
    [0, 0, 0],         # background - black
    [255, 215, 0],     # grasp - gold
    [220, 20, 60],     # cut - crimson
    [0, 191, 255],     # scoop - deep sky blue
    [123, 104, 238],   # contain - medium slate blue
    [255, 140, 0],     # pound - dark orange
    [34, 139, 34],     # support - forest green
    [255, 105, 180],   # wrap-grasp - hot pink
]


def get_palette() -> np.ndarray:
    """Return color palette as (num_classes, 3) uint8 array in RGB order."""
    return np.asarray(_CLASS_COLORS, dtype=np.uint8)


def index_to_color_mask(index_mask: np.ndarray) -> np.ndarray:
    """Map an integer index mask (H, W) to an RGB color image (H, W, 3)."""
    palette = get_palette()
    h, w = index_mask.shape
    color = palette[index_mask.clip(min=0, max=len(palette)-1)]
    return color.reshape(h, w, 3)
