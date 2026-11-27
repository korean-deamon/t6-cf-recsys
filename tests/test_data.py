"""Tests for how the signal weights are built."""
import unittest

import numpy as np

from src.config import load_config
from src.data import build_dataset

BASE = 1.79e9


def events(users, items, pct, ts=None):
    ts = ts if ts is not None else [BASE] * len(users)
    return {"users": np.array(users, dtype=np.int64),
            "items": np.array(items, dtype=np.int64),
            "ts": np.array(ts, dtype=np.float64),
            "watch_pct": np.array(pct, dtype=np.float64)}


def weight_of(ds, user_id, item_id):
    u = int(ds.user_codes(np.array([user_id]))[0])
    i = int(ds.item_codes(np.array([item_id]))[0])
    mask = (ds.matrix.row == u) & (ds.matrix.col == i)
    return float(ds.matrix.val[mask].sum()) if mask.any() else 0.0


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_low_watch_pct_ignored(self):
        ev = events([1, 1], [10, 11], [0.1, 0.9])
        ds = build_dataset(ev, None, None, self.cfg, ref_ts=BASE + 86400)
        self.assertEqual(weight_of(ds, 1, 10), 0.0)
        self.assertGreater(weight_of(ds, 1, 11), 0.0)

    def test_scheme_b_is_binary(self):
        cfg = dict(self.cfg)
        cfg["weights"] = dict(self.cfg["weights"], scheme="B")
        ev = events([1, 1], [10, 11], [0.4, 0.95])
        ds = build_dataset(ev, None, None, cfg, ref_ts=BASE + 86400)
        self.assertAlmostEqual(weight_of(ds, 1, 10), weight_of(ds, 1, 11),
                               places=6)

    def test_repeat_view_boosts_weight(self):
        single = events([1], [10], [0.8])
        repeat = events([1, 1, 1], [10, 10, 10], [0.8, 0.8, 0.8])
        ds_single = build_dataset(single, None, None, self.cfg, ref_ts=BASE + 86400)
        ds_repeat = build_dataset(repeat, None, None, self.cfg, ref_ts=BASE + 86400)
        self.assertGreater(weight_of(ds_repeat, 1, 10), weight_of(ds_single, 1, 10))

    def test_bookmark_only_pair_survives(self):
        ev = events([1], [10], [0.8])
        bm = {"users": np.array([1], dtype=np.int64),
              "items": np.array([99], dtype=np.int64),
              "ts": np.array([BASE], dtype=np.float64)}
        ds = build_dataset(ev, None, bm, self.cfg, ref_ts=BASE + 86400)
        self.assertGreater(weight_of(ds, 1, 99), 0.0)

    def test_old_interaction_decays(self):
        recent = events([1], [10], [0.8], ts=[BASE])
        old = events([1], [10], [0.8], ts=[BASE - 360 * 86400])
        ref = BASE + 86400
        recent_w = weight_of(build_dataset(recent, None, None, self.cfg, ref_ts=ref), 1, 10)
        old_w = weight_of(build_dataset(old, None, None, self.cfg, ref_ts=ref), 1, 10)
        self.assertGreater(recent_w, old_w)
        self.assertLess(old_w, recent_w * 0.9)

    def test_rating_adds_weight(self):
        ev = events([1], [10], [0.8])
        rt = {"users": np.array([1], dtype=np.int64),
              "items": np.array([10], dtype=np.int64),
              "rating": np.array([9.0]), "ts": np.array([BASE])}
        base_w = weight_of(build_dataset(ev, None, None, self.cfg, ref_ts=BASE + 86400), 1, 10)
        rated_w = weight_of(build_dataset(ev, rt, None, self.cfg, ref_ts=BASE + 86400), 1, 10)
        self.assertGreater(rated_w, base_w)

    def test_seen_includes_low_pct_views(self):
        """A short view still counts as "seen" - it must not come back."""
        ev = events([1], [10], [0.05])
        ds = build_dataset(ev, None, None, self.cfg, ref_ts=BASE + 86400)
        u = int(ds.user_codes(np.array([1]))[0])
        i = int(ds.item_codes(np.array([10]))[0])
        self.assertIn(i, ds.seen_of(u).tolist())


if __name__ == "__main__":
    unittest.main()
