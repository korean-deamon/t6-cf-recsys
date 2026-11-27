"""ALS - Alternating Least Squares for implicit feedback.

Idea: we split the user x item matrix into two small matrices

    R ~ U @ V.T

and solve U first (V fixed), then V (U fixed), step by step, until the
error is small. Each step is a linear system, so numpy solves it directly
and we do not need an extra library.

For implicit feedback we use confidence:
    c = 1 + alpha * w        (w = our weight)
A real "watched" fact is strong, a missing one is not sure (maybe the user
just never saw the item), so we do not push everything to zero.
"""
from __future__ import annotations

import numpy as np


class ALS:
    def __init__(self, factors: int = 32, iterations: int = 12,
                 regularization: float = 0.05, alpha: float = 20.0,
                 seed: int = 42, pop_prior: float = 0.0):
        self.factors = int(factors)
        self.iterations = int(iterations)
        self.reg = float(regularization)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.pop_prior = float(pop_prior)   # weight of the popularity prior
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._pop: np.ndarray | None = None

    # ------------------------------------------------------------ solve
    def _solve(self, groups, other: np.ndarray, n: int) -> np.ndarray:
        """groups = list of (key, other indices, weights)."""
        f = self.factors
        yty = other.T @ other
        ridge = np.eye(f, dtype=np.float32) * self.reg
        out = np.zeros((n, f), dtype=np.float32)
        for key, idx, vals in groups:
            y = other[idx]                       # (m, f)
            c = 1.0 + self.alpha * vals          # (m,)
            a = yty + ridge + y.T @ ((c - 1.0)[:, None] * y)
            b = y.T @ c
            out[key] = np.linalg.solve(a, b)
        return out

    def fit(self, ds) -> "ALS":
        rng = np.random.default_rng(self.seed)
        u = rng.normal(0.0, 0.01, (ds.n_users, self.factors)).astype(np.float32)
        v = rng.normal(0.0, 0.01, (ds.n_items, self.factors)).astype(np.float32)

        for _ in range(self.iterations):
            # update item factors with the user factors fixed
            v = self._solve(ds.matrix.group_cols(), u, ds.n_items)
            # and the other way round
            u = self._solve(ds.matrix.group_rows(), v, ds.n_users)

        self.user_factors, self.item_factors = u, v
        if self.pop_prior > 0:
            p = np.log1p(ds.pop_recent.astype(np.float64))
            self._pop = (p / max(float(p.max()), 1e-9)).astype(np.float32)
        return self

    # ------------------------------------------------------------ scoring
    def score(self, user_codes: np.ndarray) -> np.ndarray:
        """codes -> score matrix of shape (len(codes), n_items).

        If pop_prior > 0: every user's scores are scaled to [0, 1] and the
        recent-popularity prior is added (this helps with new content).
        """
        s = self.user_factors[user_codes] @ self.item_factors.T
        if self._pop is not None:
            lo = s.min(axis=1, keepdims=True)
            hi = s.max(axis=1, keepdims=True)
            s = (s - lo) / np.maximum(hi - lo, 1e-9) + self.pop_prior * self._pop
        return s
