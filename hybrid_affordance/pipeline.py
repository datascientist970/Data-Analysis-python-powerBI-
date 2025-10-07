from __future__ import annotations
from typing import Literal, Optional, Tuple
import os
import numpy as np
from PIL import Image

from . import AFFORDANCE_CLASSES, NUM_CLASSES
from .supervised import SupervisedAffordanceModel
from .unsupervised.spm import PCAProjector, AEProjector, VAEProjector
from .unsupervised.mcm import UMAPReducer, IsomapReducer, SpectralReducer
from .unsupervised.cluster import DBSCANClusterer, HDBSCANClusterer
from .fusion import fuse_supervised_unsupervised
from .colors import index_to_color_mask
from .report import visualize_results, save_pdf_report


SPMType = Literal['pca', 'ae', 'vae']
MCMType = Literal['umap', 'isomap', 'spectral']
ClusterType = Literal['dbscan', 'hdbscan']
MetricType = Literal['euclidean', 'cosine']


def _select_spm(spm: SPMType, n_components: int):
    if spm == 'pca':
        return PCAProjector(n_components=n_components)
    if spm == 'ae':
        return AEProjector(n_components=n_components)
    if spm == 'vae':
        return VAEProjector(n_components=n_components)
    raise ValueError(f"Unknown SPM: {spm}")


def _select_mcm(mcm: MCMType, n_components: int, metric: MetricType):
    if mcm == 'umap':
        return UMAPReducer(n_components=n_components, metric=metric)
    if mcm == 'isomap':
        return IsomapReducer(n_components=n_components, metric=metric)
    if mcm == 'spectral':
        return SpectralReducer(n_components=n_components)
    raise ValueError(f"Unknown MCM: {mcm}")


def _select_clusterer(cluster: ClusterType, metric: MetricType):
    if cluster == 'dbscan':
        return DBSCANClusterer(metric=metric)
    if cluster == 'hdbscan':
        return HDBSCANClusterer(metric=metric)
    raise ValueError(f"Unknown clusterer: {cluster}")


def _flatten_hw(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr.reshape(h * w, -1)


def _unflatten_hw(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    return arr.reshape(h, w)


def run_pipeline(
    image_path: str,
    model_path: str,
    out_dir: str,
    image_size: int = 512,
    spm: SPMType = 'pca',
    spm_components: int = 32,
    mcm: MCMType = 'umap',
    mcm_components: int = 2,
    cluster: ClusterType = 'hdbscan',
    metric: MetricType = 'euclidean',
    fusion_mode: Literal['hard', 'soft'] = 'hard',
    confidence_threshold: float = 0.5,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Supervised inference + embedding extraction
    sup = SupervisedAffordanceModel(model_path=model_path, image_size=image_size)
    out = sup.predict_image(image_path)
    probs = out['probs']  # (H, W, K)
    pred = out['pred']
    conf = out['conf']
    emb = out['embedding']  # (H, W, C)
    rgb = out['input_image']
    h, w, k = probs.shape

    # 2) SPM
    emb_flat = _flatten_hw(emb)
    projector = _select_spm(spm, spm_components)
    Z = projector.fit_transform(emb_flat)  # (N, D1)

    # 3) MCM
    reducer = _select_mcm(mcm, mcm_components, metric)
    Y = reducer.fit_transform(Z)  # (N, D2)

    # 4) Clustering
    clusterer = _select_clusterer(cluster, metric)
    cluster_labels = clusterer.fit_predict(Y)  # (N,)
    cluster_labels_2d = _unflatten_hw(cluster_labels, h, w)

    # Map clusters to affordance classes via supervised probabilities
    # For each cluster, compute average supervised distribution and pick argmax
    valid = cluster_labels >= 0
    num_clusters = cluster_labels[valid].max() + 1 if valid.any() else 0
    mapping = np.zeros((max(num_clusters, 1),), dtype=np.int64)
    if num_clusters > 0:
        probs_flat = probs.reshape(h * w, k)
        for c in range(num_clusters):
            mask = cluster_labels == c
            if mask.any():
                mean_dist = probs_flat[mask].mean(axis=0)
                mapping[c] = int(mean_dist.argmax())
            else:
                mapping[c] = 0

    # 5) Fusion
    final_mask, final_conf = fuse_supervised_unsupervised(
        supervised_probs=probs,
        supervised_conf=conf,
        unsupervised_labels=cluster_labels_2d,
        unsupervised_to_class=mapping,
        confidence_threshold=confidence_threshold,
        blend=fusion_mode,
        soft_alpha=0.5,
    )

    # 6) Visualization and reporting
    vis = visualize_results(
        rgb_image=rgb,
        supervised_mask=pred,
        unsupervised_mask=np.where(cluster_labels_2d >= 0, mapping[cluster_labels_2d.clip(min=0)], 0),
        final_mask=final_mask,
        final_conf=final_conf,
        probs=probs,
    )

    base = os.path.splitext(os.path.basename(image_path))[0]
    png_path = os.path.join(out_dir, f"{base}_hybrid.png")
    pdf_path = os.path.join(out_dir, f"{base}_report.pdf")
    vis.save(png_path)
    save_pdf_report(pdf_path, base, vis, probs, final_mask)

    # Also save raw masks
    np.save(os.path.join(out_dir, f"{base}_supervised.npy"), pred)
    np.save(os.path.join(out_dir, f"{base}_unsupervised.npy"), cluster_labels_2d)
    np.save(os.path.join(out_dir, f"{base}_final.npy"), final_mask)
