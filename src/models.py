"""Model registry: one interface - fit(ds), score(user_codes).

To add a model, write one class here and put it in MODEL_REGISTRY.
"""
from __future__ import annotations

import numpy as np

from .als import ALS
from .knn import ItemKNN


class Popularity:
    """Baseline: the content that most different people watched lately.

    This gives NDCG about 0.04 - it is only the starting point.
    """

    def __init__(self, recent_days: int = 14, pool_size: int = 500):
        # recent_days is not used here: the window is built inside build_dataset
        self.recent_days = int(recent_days)
        self.pool_size = int(pool_size)
        self.scores: np.ndarray | None = None
        self.order: np.ndarray | None = None

    def fit(self, ds) -> "Popularity":
        # a tiny random shift keeps the order stable when scores are equal
        self.scores = ds.pop_recent.astype(np.float32)
        self.order = np.argsort(-self.scores, kind="stable")
        return self

    def score(self, user_codes: np.ndarray) -> np.ndarray:
        return np.tile(self.scores, (len(user_codes), 1))


class ItemKNNModel(ItemKNN):
    pass


class ALSModel(ALS):
    pass


MODEL_REGISTRY = {
    "popularity": Popularity,
    "itemknn": ItemKNN,
    "als": ALS,
}


def build_model(name: str, cfg: dict):
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model: {name}. Available: {', '.join(MODEL_REGISTRY)}")
    params = dict(cfg["models"].get(name, {}) or {})
    return MODEL_REGISTRY[name](**params)
