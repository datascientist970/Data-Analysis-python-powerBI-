from __future__ import annotations
from typing import Dict, Tuple, Optional
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from . import NUM_CLASSES, DEFAULT_ENCODER
from .utils import load_image_rgb, resize_image, pil_to_tensor, get_preprocess_fn


class SupervisedAffordanceModel:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        encoder_name: str = DEFAULT_ENCODER,
        image_size: int = 512,
    ) -> None:
        self.model_path = model_path
        self.encoder_name = encoder_name
        self.image_size = image_size
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=NUM_CLASSES,
            activation=None,
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at: {model_path}")
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys in state_dict: {missing[:10]}{'...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[warn] Unexpected keys in state_dict: {unexpected[:10]}{'...' if len(unexpected)>10 else ''}")
        self.model.eval()

        self.preprocess = get_preprocess_fn(self.encoder_name)

    @torch.no_grad()
    def predict_image(
        self,
        image_path: str,
    ) -> Dict[str, np.ndarray]:
        """
        Returns a dict with keys:
        - probs: (H, W, K) supervised softmax probabilities
        - pred: (H, W) int argmax
        - conf: (H, W) float in [0,1] max softmax
        - embedding: (H, W, C) encoder-derived per-pixel embedding
        - input_image: original resized PIL image (as numpy RGB)
        """
        img = load_image_rgb(image_path)
        img_resized = resize_image(img, self.image_size)
        x = pil_to_tensor(img_resized).unsqueeze(0).to(self.device)  # 1x3xHxW, [0,1]
        x = self.preprocess(x)

        # Encoder features
        features = self.model.encoder(x)  # list of feature maps at different scales
        # Choose multiple scales, upsample to input size, and concatenate
        target_hw = x.shape[-2:]
        upsampled = []
        for f in features[1:]:  # skip very early stem to reduce noise
            f_up = F.interpolate(f, size=target_hw, mode='bilinear', align_corners=False)
            upsampled.append(f_up)
        embedding_tensor = torch.cat(upsampled, dim=1)  # 1 x C_concat x H x W
        # L2 normalize per-pixel for metric stability
        embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)

        # Segmentation logits
        logits = self.model(x)  # 1xKxHxW
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)  # 1xHxW each

        # to numpy
        probs_np = probs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        pred_np = pred.squeeze(0).detach().cpu().numpy().astype(np.int64)
        conf_np = conf.squeeze(0).detach().cpu().numpy().astype(np.float32)

        emb_np = embedding_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        img_np = np.asarray(img_resized).astype(np.uint8)

        return {
            'probs': probs_np,
            'pred': pred_np,
            'conf': conf_np,
            'embedding': emb_np,
            'input_image': img_np,
        }
