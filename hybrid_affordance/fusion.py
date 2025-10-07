from __future__ import annotations
from typing import Dict
import numpy as np


def fuse_affordances(
    supervised_labels: np.ndarray,
    supervised_confidence: np.ndarray,
    unsup_labels: np.ndarray,
    unsup_confidence: np.ndarray,
    confidence_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    assert supervised_labels.shape == unsup_labels.shape
    assert supervised_confidence.shape == unsup_confidence.shape
    use_supervised = supervised_confidence >= confidence_threshold
    final_labels = np.where(use_supervised, supervised_labels, unsup_labels).astype(np.int32)
    final_confidence = np.where(use_supervised, supervised_confidence, unsup_confidence).astype(np.float32)
    return {
        'labels': final_labels,
        'confidence': final_confidence,
    }
