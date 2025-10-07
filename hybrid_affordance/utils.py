from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .colors import get_palette


def load_image_rgb(path: str) -> Image.Image:
    img = Image.open(path).convert('RGB')
    return img


def resize_image(img: Image.Image, size: int) -> Image.Image:
    # Simple square resize for segmentation; keeps behavior predictable
    return img.resize((size, size), resample=Image.BILINEAR)


def get_preprocess_fn(encoder_name: str = 'efficientnet-b3'):
    params = smp.encoders.get_preprocessing_params(encoder_name, pretrained='imagenet')
    mean = torch.tensor(params['mean']).view(1, 3, 1, 1)
    std = torch.tensor(params['std']).view(1, 3, 1, 1)

    def _fn(x: torch.Tensor) -> torch.Tensor:
        return (x - mean.to(x.device)) / std.to(x.device)

    return _fn


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    # HWC -> CHW
    arr = np.transpose(arr, (0, 1, 2))
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor


def tensor_to_pil_img(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)
    arr = (t.numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def colorize_mask(index_mask: np.ndarray) -> Image.Image:
    palette = get_palette()
    colored = palette[index_mask]
    return Image.fromarray(colored.astype(np.uint8))


def overlay_color_on_image(img: Image.Image, color_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    img_np = np.asarray(img).astype(np.float32)
    mask_np = np.asarray(color_mask).astype(np.float32)
    blended = (1 - alpha) * img_np + alpha * mask_np
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def compute_class_prototypes(
    embeddings_flat: np.ndarray,
    probs: np.ndarray,
    confidence_threshold: float = 0.8,
) -> np.ndarray:
    """
    Compute per-class prototype vectors by averaging embeddings where
    supervised confidence for the class is above a threshold.

    embeddings_flat: (N, C)
    probs: (N, K)
    returns: (K, C)
    """
    num_points, emb_dim = embeddings_flat.shape
    num_classes = probs.shape[1]
    prototypes = np.zeros((num_classes, emb_dim), dtype=np.float32)
    for k in range(num_classes):
        mask = probs[:, k] >= confidence_threshold
        if mask.any():
            prototypes[k] = embeddings_flat[mask].mean(axis=0)
        else:
            # Fallback: global mean if class not present confidently
            prototypes[k] = embeddings_flat.mean(axis=0)
    return prototypes


def top_class_confidence_summary(
    probs: np.ndarray,
    final_mask: np.ndarray,
    class_names: List[str],
) -> List[Tuple[str, float, float]]:
    """
    For each class present in final_mask, compute:
    - pixel ratio (fraction of pixels assigned to this class)
    - average supervised confidence (avg max prob where class is selected)

    Returns list of tuples (class_name, pixel_ratio, avg_confidence).
    """
    h, w = final_mask.shape
    n = h * w
    result = []
    for idx, name in enumerate(class_names):
        pix = (final_mask == idx)
        count = int(pix.sum())
        if count == 0:
            continue
        ratio = count / float(n)
        avg_conf = float(probs[pix].max(axis=1).mean())
        result.append((name, ratio, avg_conf))
    return result
