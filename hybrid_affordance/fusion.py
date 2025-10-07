from __future__ import annotations
from typing import Literal, Tuple
import numpy as np

from . import INDEX_TO_CLASS, NUM_CLASSES


def fuse_supervised_unsupervised(
    supervised_probs: np.ndarray,  # (H, W, K)
    supervised_conf: np.ndarray,   # (H, W)
    unsupervised_labels: np.ndarray,  # (H, W) ints starting at 0 or -1 for noise
    unsupervised_to_class: np.ndarray,  # (num_clusters,) mapping to [0..K-1]
    confidence_threshold: float = 0.5,
    blend: Literal['hard', 'soft'] = 'hard',
    soft_alpha: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - final_mask: (H, W) int labels in [0..K-1]
    - final_conf: (H, W) float confidence estimate

    If supervised_conf > threshold: take supervised argmax
    else: use unsupervised mapping; if noise (-1), fallback to supervised argmax

    If blend == 'soft', blend supervised prob vector with a 1-hot from unsupervised class.
    """
    h, w, k = supervised_probs.shape
    argmax_sup = supervised_probs.argmax(axis=2)

    final_mask = argmax_sup.copy()
    final_conf = supervised_conf.copy()

    low_conf = supervised_conf <= confidence_threshold

    # Map unsupervised labels to class ids
    unsup_flat = unsupervised_labels.reshape(-1)
    mapped = np.full_like(unsup_flat, fill_value=-1)
    valid = unsup_flat >= 0
    mapped[valid] = unsupervised_to_class[unsup_flat[valid]]
    mapped = mapped.reshape(h, w)

    if blend == 'hard':
        # Replace low-conf pixels with unsupervised if available
        use_unsup = low_conf & (mapped >= 0)
        final_mask[use_unsup] = mapped[use_unsup]
        # Confidence: where we used unsup, set to 0.5 baseline; elsewhere keep supervised
        final_conf[use_unsup] = np.maximum(final_conf[use_unsup], 0.5)
        # If noise in unsup, keep supervised
    else:  # soft blending
        final_probs = supervised_probs.copy()
        one_hot = np.zeros_like(supervised_probs)
        for cls in range(k):
            one_hot[:, :, cls] = (mapped == cls).astype(np.float32)
        blended = (1 - soft_alpha) * final_probs + soft_alpha * one_hot
        final_mask = blended.argmax(axis=2)
        final_conf = blended.max(axis=2)
        # Ensure high-conf supervised stays dominant
        final_mask[~low_conf] = argmax_sup[~low_conf]
        final_conf[~low_conf] = supervised_conf[~low_conf]

    return final_mask, final_conf
