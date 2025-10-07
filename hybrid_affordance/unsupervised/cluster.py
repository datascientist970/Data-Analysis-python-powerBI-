from __future__ import annotations
from typing import Optional
import numpy as np
from sklearn.cluster import DBSCAN
try:
    import hdbscan
except Exception:
    hdbscan = None


class Clusterer:
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DBSCANClusterer(Clusterer):
    def __init__(self, eps: float = 0.5, min_samples: int = 20, metric: str = 'euclidean') -> None:
        self.db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.db.fit_predict(X)


class HDBSCANClusterer(Clusterer):
    def __init__(self, min_cluster_size: int = 50, metric: str = 'euclidean') -> None:
        if hdbscan is None:
            raise ImportError("hdbscan is not installed")
        self.db = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric, core_dist_n_jobs=-1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.db.fit_predict(X)
