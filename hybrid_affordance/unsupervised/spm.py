from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class SubspaceProjector:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:  # (N,C) -> (N,D)
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PCAProjector(SubspaceProjector):
    def __init__(self, n_components: int = 32, whiten: bool = False) -> None:
        self.pca = PCA(n_components=n_components, whiten=whiten, random_state=42)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        hidden = max(128, bottleneck_dim * 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class AEProjector(SubspaceProjector):
    def __init__(self, n_components: int = 32, epochs: int = 10, batch_size: int = 2048, lr: float = 1e-3, device: Optional[str] = None) -> None:
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model: Optional[AutoEncoder] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        n, c = X.shape
        model = AutoEncoder(c, self.n_components).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        X_tensor = torch.from_numpy(X).to(self.device)
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                batch = X_tensor[idx]
                opt.zero_grad()
                x_hat, _ = model(batch)
                loss = loss_fn(x_hat, batch)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            _, z = model(X_tensor)
        self.model = model
        return z.detach().cpu().numpy().astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "AEProjector must be fit before transform"
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, z = self.model(X_tensor)
        return z.detach().cpu().numpy().astype(np.float32)


class VAE(nn.Module):
    def __init__(self, input_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        hidden = max(128, bottleneck_dim * 4)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc_mu = nn.Linear(hidden, bottleneck_dim)
        self.fc_logvar = nn.Linear(hidden, bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, hidden)
        self.fc4 = nn.Linear(hidden, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h3 = torch.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    recon = F.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kld


class VAEProjector(SubspaceProjector):
    def __init__(self, n_components: int = 32, epochs: int = 10, batch_size: int = 2048, lr: float = 1e-3, device: Optional[str] = None) -> None:
        self.n_components = n_components
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model: Optional[VAE] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        n, c = X.shape
        model = VAE(c, self.n_components).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        X_tensor = torch.from_numpy(X).to(self.device)
        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i+self.batch_size]
                batch = X_tensor[idx]
                opt.zero_grad()
                x_hat, mu, logvar = model(batch)
                loss = vae_loss(x_hat, batch, mu, logvar)
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode(X_tensor)
        self.model = model
        return mu.detach().cpu().numpy().astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.model is not None, "VAEProjector must be fit before transform"
        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(X_tensor)
        return mu.detach().cpu().numpy().astype(np.float32)
