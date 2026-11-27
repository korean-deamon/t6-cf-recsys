"""Item-kNN - "people who watched this also watched that".

We compute the cosine similarity between items and then recommend content
that is close to what the user already watched.

scipy is imported in one place only, and only when it is needed.
"""
from __future__ import annotations

import numpy as np


class ItemKNN:
    def __init__(self, neighbors: int = 200, shrink: float = 10.0,
                 block_items: int = 2000):
        self.neighbors = int(neighbors)
        self.shrink = float(shrink)
        self.block_items = int(block_items)
        self.x = None      # (users x items) CSR
        self.sim = None    # (items x items) top-N neighbours

    def fit(self, ds) -> "ItemKNN":
        from scipy import sparse

        m = ds.matrix
        self.x = sparse.csr_matrix(
            (m.val.astype(np.float32), (m.row, m.col)),
            shape=(ds.n_users, ds.n_items), dtype=np.float32,
        )
        norms = np.sqrt(np.asarray(self.x.multiply(self.x).sum(axis=0))).ravel()

        rows, cols, data = [], [], []
        for start in range(0, ds.n_items, self.block_items):
            stop = min(start + self.block_items, ds.n_items)
            block = self.x[:, start:stop]
            sims = np.asarray((self.x.T @ block).todense(), dtype=np.float32)

            denom = np.outer(norms, norms[start:stop]) + self.shrink
            sims = np.divide(sims, denom, out=np.zeros_like(sims),
                             where=denom > 0)
            # drop the similarity of an item with itself
            for j in range(stop - start):
                sims[start + j, j] = 0.0

            k = min(self.neighbors, max(sims.shape[0] - 1, 1))
            top = np.argpartition(-sims, k - 1, axis=0)[:k]      # (k, block)
            for j in range(stop - start):
                idx = top[:, j]
                keep = sims[idx, j] > 0
                rows.append(idx[keep])
                cols.append(np.full(keep.sum(), start + j, dtype=np.int32))
                data.append(sims[idx[keep], j])

        rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int32)
        cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int32)
        data = np.concatenate(data) if data else np.empty(0, dtype=np.float32)
        self.sim = sparse.csr_matrix(
            (data, (rows, cols)), shape=(ds.n_items, ds.n_items), dtype=np.float32
        )
        return self

    def score(self, user_codes: np.ndarray) -> np.ndarray:
        scored = self.x[user_codes] @ self.sim
        return np.asarray(scored.todense(), dtype=np.float32)
