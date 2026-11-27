"""Tests for the offline validation logic (no database).

The rule from the task: relevant = FIRST view inside the test window with
watch_pct >= 0.3 and the item exists in train.
"""
import unittest

import numpy as np

from src.config import load_config
from src.data import build_dataset
from src.evaluate import build_relevance
from tests.helpers import CLUSTER_SIZE, make_events


def _val_events(pairs, pct_values, base=1.79e9):
    return {
        "users": np.array([u for u, _ in pairs], dtype=np.int64),
        "items": np.array([i for _, i in pairs], dtype=np.int64),
        "ts": np.arange(len(pairs), dtype=np.float64) + base,
        "watch_pct": np.array(pct_values, dtype=np.float64),
    }


class TestBuildRelevance(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.events = make_events(n_users=8, per_user=8, seed=3)
        self.ds = build_dataset(self.events, None, None, self.cfg,
                                ref_ts=self.events["ts"].max() + 1)
        self.seen = {}
        for u, i in zip(self.events["users"].tolist(), self.events["items"].tolist()):
            self.seen.setdefault(u, set()).add(i)

    def _next_unseen(self, user_id):
        cluster_start = 5000 + (CLUSTER_SIZE * ((user_id - 1000) % 4))
        for item in range(cluster_start, cluster_start + CLUSTER_SIZE):
            if item not in self.seen[user_id]:
                return item
        self.skipTest("no unused item found")

    def test_first_view_with_low_pct_is_excluded(self):
        u = 1000
        item = self._next_unseen(u)
        val = _val_events([(u, item), (u, item)], [0.1, 0.9])
        rel = build_relevance(val, self.ds, 0.3)
        # the first view has pct 0.1 -> not relevant
        self.assertNotIn(u, {self.ds.user_ids[k] for k in rel})

    def test_first_view_with_good_pct_is_included(self):
        u = 1000
        item = self._next_unseen(u)
        val = _val_events([(u, item)], [0.8])
        rel = build_relevance(val, self.ds, 0.3)
        user_codes = {int(self.ds.user_ids[k]): {int(self.ds.item_ids[i]) for i in v}
                      for k, v in rel.items()}
        self.assertIn(item, user_codes[u])

    def test_unknown_item_ignored(self):
        val = _val_events([(1000, 999_999)], [1.0])
        rel = build_relevance(val, self.ds, 0.3)
        self.assertEqual(rel, {})

    def test_unknown_user_ignored(self):
        val = _val_events([(777_777, 5000)], [1.0])
        rel = build_relevance(val, self.ds, 0.3)
        self.assertEqual(rel, {})


if __name__ == "__main__":
    unittest.main()
