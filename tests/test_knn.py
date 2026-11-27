"""Tests for item-kNN (needs scipy)."""
import unittest

import numpy as np

from src.config import load_config
from src.knn import ItemKNN
from src.pipeline import run_pipeline
from tests.helpers import make_store

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:          # pragma: no cover
    HAS_SCIPY = False

K = 10


@unittest.skipUnless(HAS_SCIPY, "scipy is not installed")
class TestItemKNN(unittest.TestCase):
    def _fit(self):
        store = make_store()
        cfg = load_config()
        events = store.events
        from src.data import build_dataset
        ds = build_dataset(events, None, None, cfg, ref_ts=events["ts"].max() + 1)
        model = ItemKNN(neighbors=20, shrink=1.0).fit(ds)
        return ds, model

    def test_shapes_and_finiteness(self):
        ds, model = self._fit()
        self.assertEqual(model.sim.shape, (ds.n_items, ds.n_items))
        scores = model.score(np.arange(min(ds.n_users, 10)))
        self.assertEqual(scores.shape, (min(ds.n_users, 10), ds.n_items))
        self.assertTrue(np.isfinite(scores).all())

    def test_similar_items_share_cluster(self):
        ds, model = self._fit()
        sims = np.asarray(model.sim.todense())
        item_cluster = (ds.item_ids - 5000) // 10
        same, other = [], []
        for i in range(ds.n_items):
            mask = item_cluster == item_cluster[i]
            mask[i] = False
            same.extend(sims[i, mask].tolist())
            other.extend(sims[i, ~mask & (np.arange(ds.n_items) != i)].tolist())
        self.assertGreater(np.mean(same), np.mean(other))

    def test_pipeline_end_to_end(self):
        cfg = load_config()
        store = make_store()
        result = run_pipeline(store, cfg, model_name="itemknn",
                              run_name="test_knn", log=lambda *_: None)
        rows = store.runs[result["run_id"]]
        self.assertEqual(len(rows), len(store.eval_users) * K)

        seen = {}
        for u, i in zip(store.events["users"].tolist(), store.events["items"].tolist()):
            seen.setdefault(u, set()).add(i)
        by_user = {}
        for _, user_id, rank, item_id in rows:
            self.assertNotIn(item_id, seen.get(user_id, set()))
            by_user.setdefault(user_id, []).append(item_id)
        for items in by_user.values():
            self.assertEqual(len(set(items)), K)


if __name__ == "__main__":
    unittest.main()
