from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.manifold import Isomap, SpectralEmbedding
try:
    import umap
except Exception:
    umap = None


class ManifoldReducer:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class UMAPReducer(ManifoldReducer):
    def __init__(self, n_components: int = 2, metric: str = 'euclidean', n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> None:
        if umap is None:
            raise ImportError("umap-learn is not installed")
        self.reducer = umap.UMAP(n_components=n_components, metric=metric, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.transform(X)


class IsomapReducer(ManifoldReducer):
    def __init__(self, n_components: int = 2, n_neighbors: int = 5, metric: str = 'euclidean') -> None:
        self.reducer = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric=metric)
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._fitted = True
        return self.reducer.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Isomap has no out-of-sample transform; refit on full X
        return self.reducer.fit_transform(X)


class SpectralReducer(ManifoldReducer):
    def __init__(self, n_components: int = 2) -> None:
        self.reducer = SpectralEmbedding(n_components=n_components, random_state=42)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        # SpectralEmbedding has no transform, refit
        return self.reducer.fit_transform(X)
