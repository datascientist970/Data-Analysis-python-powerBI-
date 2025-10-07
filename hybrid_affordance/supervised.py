from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
from .config import AFFORDANCE_CLASSES


class SupervisedAffordanceModel:
    def __init__(
        self,
        model_path: str,
        encoder_name: str = 'efficientnet-b3',
        num_classes: int = 8,
        device: str = 'cpu',
    ) -> None:
        self.device = torch.device(device)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            classes=num_classes,
            activation=None,
            encoder_weights=None,
        )
        state = torch.load(model_path, map_location=self.device)
        if 'state_dict' in state:
            state = state['state_dict']
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device)
        self.model.eval()

        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.mean = torch.tensor(params['mean']).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(params['std']).view(1, 3, 1, 1).to(self.device)
        self.num_classes = num_classes

    @torch.no_grad()
    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = (tensor - self.mean) / self.std
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = image.shape[:2]
        x = self._to_tensor(image)
        logits = self.model(x)  # [1, C, H, W]
        probs = F.softmax(logits, dim=1)
        probs_np = probs.squeeze(0).permute(1, 2, 0).cpu().numpy()  # H W C
        labels = np.argmax(probs_np, axis=-1).astype(np.int32)
        confidence = np.max(probs_np, axis=-1).astype(np.float32)
        return {
            'probs': probs_np,
            'labels': labels,
            'confidence': confidence,
        }

    @torch.no_grad()
    def extract_encoder_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = self._to_tensor(image)
        # Use encoder multi-scale features and upsample to input resolution
        features = self.model.encoder(x)
        upsampled = []
        for feat in features:
            up = F.interpolate(feat, size=x.shape[-2:], mode='bilinear', align_corners=False)
            upsampled.append(up)
        feats = torch.cat(upsampled, dim=1)  # [1, Csum, H, W]
        feats_np = feats.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()  # H W C
        flat = feats_np.reshape(-1, feats_np.shape[-1])  # N C
        return feats_np, flat
