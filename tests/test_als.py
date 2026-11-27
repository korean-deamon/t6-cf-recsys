"""Tests for select_top_k and ALS (no database)."""
import unittest

import numpy as np

from src.als import ALS
from src.data import build_dataset
from src.pipeline import select_top_k
from tests.helpers import make_events


class TestSelectTopK(unittest.TestCase):
    def test_excludes_seen(self):
        scores = np.array([10, 9, 8, 7], dtype=np.float32)
        items, vals = select_top_k(scores, [0, 1], k=2)
        self.assertEqual(items.tolist(), [2, 3])
        self.assertEqual(len(vals), 2)

    def test_exactly_k_with_fallback(self):
        scores = np.array([-np.inf, -np.inf, 5.0, -np.inf], dtype=np.float32)
        items, _ = select_top_k(scores, [], k=3,
                                fallback_items=np.array([2, 0, 1, 3]),
                                fallback_scores=np.array([5.0, 1.0, 0.5, 0.1],
                                                         dtype=np.float32))
        self.assertEqual(len(items), 3)
        self.assertEqual(len(set(items.tolist())), 3)
        self.assertNotIn(-1, items.tolist())

    def test_fallback_skips_seen(self):
        scores = np.full(4, -np.inf, dtype=np.float32)
        items, _ = select_top_k(scores, [0], k=2,
                                fallback_items=np.array([0, 1, 2, 3]),
                                fallback_scores=np.ones(4, dtype=np.float32))
        self.assertNotIn(0, items.tolist())
        self.assertEqual(len(items), 2)


class TestALS(unittest.TestCase):
    def test_recovers_cluster_structure(self):
        events = make_events(n_users=40, per_user=8, seed=1)
        cfg = {"weights": {"scheme": "A", "min_watch_pct": 0.3, "watch": 1.0,
                           "repeat": 0.5, "rating_scale": 0.2, "bookmark": 1.0,
                           "half_life_days": 180, "binary_watch": 1.0},
               "models": {"popularity": {"recent_days": 14, "pool_size": 500},
                          "als": {"factors": 8, "iterations": 8,
                                  "regularization": 0.05, "alpha": 20.0,
                                  "seed": 42}}}
        ds = build_dataset(events, None, None, cfg, ref_ts=events["ts"].max() + 1)
        self.assertGreater(len(ds.matrix), 0)

        model = ALS(factors=8, iterations=8, regularization=0.05, alpha=20.0).fit(ds)
        self.assertEqual(model.user_factors.shape, (ds.n_users, 8))
        self.assertEqual(model.item_factors.shape, (ds.n_items, 8))
        self.assertTrue(np.isfinite(model.user_factors).all())

        scores = model.score(np.arange(ds.n_users))
        self.assertEqual(scores.shape, (ds.n_users, ds.n_items))

        # scores inside a group must be higher than outside it
        same, other = [], []
        item_cluster = (ds.item_ids - 5000) // 10     # original item_id -> group
        for u in range(20):
            in_cluster = item_cluster == (u % 4)
            same.append(scores[u, in_cluster].mean())
            other.append(scores[u, ~in_cluster].mean())
        self.assertGreater(np.mean(same), np.mean(other))


if __name__ == "__main__":
    unittest.main()
