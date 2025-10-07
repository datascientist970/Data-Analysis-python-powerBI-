from __future__ import annotations
import os
from typing import Tuple
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    return np.asarray(image)


def save_image(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray(image.astype(np.uint8))
    img.save(path)


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(image.astype(np.uint8))
    return np.asarray(img.resize(size[::-1], resample=Image.BILINEAR))
