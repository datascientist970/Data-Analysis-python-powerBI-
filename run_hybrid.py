from __future__ import annotations
import argparse
import glob
import os
from typing import List

import numpy as np
import torch

from hybrid_affordance.config import DEFAULT_MODEL_PATH, DEFAULT_RESULTS_DIR
from hybrid_affordance.supervised import SupervisedAffordanceModel
from hybrid_affordance.unsupervised import UnsupervisedDiscoverer, UnsupervisedConfig
from hybrid_affordance.fusion import fuse_affordances
from hybrid_affordance.utils.image_io import load_image
from hybrid_affordance.viz import render_grid
from hybrid_affordance.report import generate_report


def list_images(path: str) -> List[str]:
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
    paths = []
    if os.path.isdir(path):
        for e in exts:
            paths.extend(glob.glob(os.path.join(path, e)))
    elif os.path.isfile(path):
        paths = [path]
    else:
        raise FileNotFoundError(path)
    paths.sort()
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Hybrid Affordance Detection')
    parser.add_argument('--images', type=str, required=True, help='Image file or directory')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH, help='Path to best_model.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--spm', type=str, default='pca', choices=['pca', 'ae', 'vae'])
    parser.add_argument('--spm-dim', type=int, default=32)
    parser.add_argument('--spm-epochs', type=int, default=10)
    parser.add_argument('--mcm', type=str, default='umap', choices=['umap', 'isomap', 'spectral'])
    parser.add_argument('--mcm-dim', type=int, default=8)
    parser.add_argument('--cluster', type=str, default='hdbscan', choices=['dbscan', 'hdbscan'])
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'])
    parser.add_argument('--dbscan-eps', type=float, default=0.5)
    parser.add_argument('--dbscan-min-samples', type=int, default=50)
    parser.add_argument('--hdbscan-min-cluster-size', type=int, default=300)
    parser.add_argument('--confidence-threshold', type=float, default=0.5)
    parser.add_argument('--out', type=str, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Supervised model
    sup = SupervisedAffordanceModel(model_path=args.model, device=args.device)

    # Unsupervised config
    cfg = UnsupervisedConfig(
        spm=args.spm,
        spm_dim=args.spm_dim,
        spm_epochs=args.spm_epochs,
        mcm=args.mcm,
        mcm_dim=args.mcm_dim,
        cluster=args.cluster,
        metric=args.metric,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        device=args.device,
    )
    unsup = UnsupervisedDiscoverer(cfg)

    images = list_images(args.images)
    if not images:
        print('No images found.')
        return

    for img_path in images:
        img = load_image(img_path)
        h, w = img.shape[:2]
        # Supervised
        sup_out = sup.predict(img)
        sup_labels = sup_out['labels']
        sup_conf = sup_out['confidence']
        # Features
        feats_hw, feats_flat = sup.extract_encoder_features(img)
        # Unsupervised discovery
        disc = unsup.discover(feats_flat, sup_out['probs'].reshape(-1, sup.num_classes), h, w)
        unsup_labels = disc['unsup_labels']
        unsup_conf = disc['unsup_confidence']
        # Fusion
        fused = fuse_affordances(sup_labels, sup_conf, unsup_labels, unsup_conf, confidence_threshold=args.confidence_threshold)
        fused_labels = fused['labels']
        fused_conf = fused['confidence']

        base = os.path.splitext(os.path.basename(img_path))[0]
        composite_path = os.path.join(args.out, f"{base}_grid.png")
        render_grid(
            image=img,
            sup_labels=sup_labels,
            unsup_labels=unsup_labels,
            fused_labels=fused_labels,
            sup_conf=sup_conf,
            unsup_conf=unsup_conf,
            save_path=composite_path,
        )

        # PDF report
        generate_report(
            image_basename=base,
            out_dir=args.out,
            composite_png_path=composite_path,
            sup_labels=sup_labels,
            sup_conf=sup_conf,
            fused_labels=fused_labels,
            fused_conf=fused_conf,
        )
        print(f"Saved results for {img_path} -> {composite_path}")


if __name__ == '__main__':
    main()
