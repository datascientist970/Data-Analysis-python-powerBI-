from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap, SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import umap
import hdbscan


class AutoencoderSPM(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        hidden = max(128, min(512, in_dim // 2))
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

    def fit_transform(
        self,
        X: np.ndarray,
        epochs: int = 10,
        batch_size: int = 8192,
        lr: float = 1e-3,
        device: str = 'cpu',
        verbose: bool = False,
    ) -> np.ndarray:
        device_t = torch.device(device)
        self.to(device_t)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(device_t)
                z, xh = self(xb)
                loss = F.mse_loss(xh, xb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.eval()
        with torch.no_grad():
            Z = []
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            for (xb,) in loader:
                xb = xb.to(device_t)
                z, _ = self(xb)
                Z.append(z.cpu())
        return torch.cat(Z, dim=0).numpy()

    @torch.no_grad()
    def transform(self, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
        device_t = torch.device(device)
        self.to(device_t)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=False, drop_last=False)
        Z = []
        for (xb,) in loader:
            xb = xb.to(device_t)
            z, _ = self(xb)
            Z.append(z.cpu())
        return torch.cat(Z, dim=0).numpy()


class VAESPM(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        hidden = max(128, min(512, in_dim // 2))
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.fc_dec1 = nn.Linear(latent_dim, hidden)
        self.fc_out = nn.Linear(hidden, in_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_dec1(z))
        return self.fc_out(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

    def loss_fn(self, x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        recon = F.mse_loss(x_hat, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + 1e-3 * kld

    def fit_transform(
        self,
        X: np.ndarray,
        epochs: int = 10,
        batch_size: int = 8192,
        lr: float = 1e-3,
        device: str = 'cpu',
        verbose: bool = False,
    ) -> np.ndarray:
        device_t = torch.device(device)
        self.to(device_t)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for _ in range(epochs):
            for (xb,) in loader:
                xb = xb.to(device_t)
                xh, mu, logvar = self(xb)
                loss = self.loss_fn(xh, xb, mu, logvar)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        self.eval()
        with torch.no_grad():
            Z = []
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            for (xb,) in loader:
                xb = xb.to(device_t)
                _, mu, _ = self(xb)
                Z.append(mu.cpu())
        return torch.cat(Z, dim=0).numpy()

    @torch.no_grad()
    def transform(self, X: np.ndarray, device: str = 'cpu') -> np.ndarray:
        device_t = torch.device(device)
        self.to(device_t)
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X).float())
        loader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=False, drop_last=False)
        Z = []
        for (xb,) in loader:
            xb = xb.to(device_t)
            _, mu, _ = self(xb)
            Z.append(mu.cpu())
        return torch.cat(Z, dim=0).numpy()


@dataclass
class UnsupervisedConfig:
    spm: Literal['pca', 'ae', 'vae'] = 'pca'
    spm_dim: int = 32
    spm_epochs: int = 10
    mcm: Literal['umap', 'isomap', 'spectral'] = 'umap'
    mcm_dim: int = 8
    cluster: Literal['dbscan', 'hdbscan'] = 'hdbscan'
    metric: Literal['euclidean', 'cosine'] = 'euclidean'
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 50
    hdbscan_min_cluster_size: int = 300
    device: str = 'cpu'


class UnsupervisedDiscoverer:
    def __init__(self, cfg: Optional[UnsupervisedConfig] = None) -> None:
        self.cfg = cfg or UnsupervisedConfig()

    def _run_spm(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.spm == 'pca':
            pca = PCA(n_components=cfg.spm_dim, whiten=True, random_state=42)
            return pca.fit_transform(X)
        elif cfg.spm == 'ae':
            ae = AutoencoderSPM(in_dim=X.shape[1], latent_dim=cfg.spm_dim)
            return ae.fit_transform(X, epochs=cfg.spm_epochs, device=cfg.device)
        elif cfg.spm == 'vae':
            vae = VAESPM(in_dim=X.shape[1], latent_dim=cfg.spm_dim)
            return vae.fit_transform(X, epochs=cfg.spm_epochs, device=cfg.device)
        else:
            raise ValueError(f"Unknown SPM: {cfg.spm}")

    def _run_mcm(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.mcm == 'umap':
            reducer = umap.UMAP(n_components=cfg.mcm_dim, metric=cfg.metric, random_state=42)
            return reducer.fit_transform(X)
        elif cfg.mcm == 'isomap':
            reducer = Isomap(n_components=cfg.mcm_dim, metric=cfg.metric)
            return reducer.fit_transform(X)
        elif cfg.mcm == 'spectral':
            reducer = SpectralEmbedding(n_components=cfg.mcm_dim, affinity='nearest_neighbors', random_state=42)
            return reducer.fit_transform(X)
        else:
            raise ValueError(f"Unknown MCM: {cfg.mcm}")

    def _cluster(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        if cfg.cluster == 'dbscan':
            model = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples, metric=cfg.metric, n_jobs=-1)
            labels = model.fit_predict(X)
        elif cfg.cluster == 'hdbscan':
            model = hdbscan.HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size, metric=cfg.metric, core_dist_n_jobs=0)
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unknown cluster: {cfg.cluster}")
        return labels

    def _reassign_outliers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        # Assign label -1 to nearest cluster center
        valid = labels >= 0
        if not np.any(valid):
            return np.zeros_like(labels)
        unique = np.unique(labels[valid])
        centers = []
        for k in unique:
            centers.append(X[labels == k].mean(axis=0))
        centers = np.stack(centers, axis=0)
        out_idx = np.where(labels < 0)[0]
        if out_idx.size > 0:
            D = pairwise_distances(X[out_idx], centers, metric='cosine' if self.cfg.metric == 'cosine' else 'euclidean')
            nn = np.argmin(D, axis=1)
            reassigned = unique[nn]
            labels[out_idx] = reassigned
        # Reindex to 0..K-1 for compactness
        mapping = {k: i for i, k in enumerate(sorted(np.unique(labels)))}
        labels = np.vectorize(mapping.get)(labels)
        return labels

    def discover(
        self,
        encoder_flat_features: np.ndarray,  # [N, C]
        supervised_probs_flat: np.ndarray,  # [N, 8]
        height: int,
        width: int,
    ) -> Dict[str, np.ndarray]:
        # 1) Normalize features for stability
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(encoder_flat_features)
        # 2) SPM
        Z = self._run_spm(X)
        # 3) MCM
        M = self._run_mcm(Z)
        # 4) Clustering on manifold
        cluster_labels = self._cluster(M)
        cluster_labels = self._reassign_outliers(M, cluster_labels)
        # 5) Map clusters to affordance classes via supervised probabilities
        K = int(cluster_labels.max()) + 1
        cluster_to_class = np.zeros(K, dtype=np.int32)
        cluster_conf = np.zeros(K, dtype=np.float32)
        for k in range(K):
            idx = (cluster_labels == k)
            if np.any(idx):
                mean_probs = supervised_probs_flat[idx].mean(axis=0)
                mapped_class = int(np.argmax(mean_probs))
                confidence = float(np.max(mean_probs))
                cluster_to_class[k] = mapped_class
                cluster_conf[k] = confidence
            else:
                cluster_to_class[k] = 0
                cluster_conf[k] = 0.0
        unsup_labels_flat = cluster_to_class[cluster_labels]
        unsup_conf_flat = cluster_conf[cluster_labels]
        # Reshape to image
        cluster_labels_2d = cluster_labels.reshape(height, width)
        unsup_labels_2d = unsup_labels_flat.reshape(height, width)
        unsup_conf_2d = unsup_conf_flat.reshape(height, width)
        return {
            'cluster_labels': cluster_labels_2d,
            'unsup_labels': unsup_labels_2d,
            'unsup_confidence': unsup_conf_2d,
        }
