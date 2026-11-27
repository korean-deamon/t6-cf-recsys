"""Test for the no-database (local file) mode: full flow on an export file."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import load_config
from src.pipeline import run_pipeline
from src.store import FileStore
from tests.helpers import make_events

K = 10


def write_npz(path: Path, events: dict, eval_users: np.ndarray, n_items: int) -> None:
    empty_i, empty_f = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    np.savez_compressed(
        path,
        events_users=events["users"], events_items=events["items"],
        events_ts=events["ts"], events_pct=events["watch_pct"],
        ratings_users=empty_i, ratings_items=empty_i,
        ratings_rating=empty_f, ratings_ts=empty_f,
        bookmarks_users=empty_i, bookmarks_items=empty_i, bookmarks_ts=empty_f,
        eval_users=eval_users, n_items=np.array([n_items], dtype=np.int64),
    )


class TestFileStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "raw.npz"
        self.events = make_events(n_users=30, per_user=6, seed=7)
        self.eval_users = np.array([1000 + u for u in range(30)] + [424_242],
                                   dtype=np.int64)
        write_npz(self.path, self.events, self.eval_users, n_items=40)
        self.cfg = load_config()
        self.cfg["models"]["als"] = {"factors": 8, "iterations": 3,
                                     "regularization": 0.05, "alpha": 20.0,
                                     "seed": 42}

    def test_reads_events(self):
        store = FileStore(self.path)
        ev = store.fetch_events(None, None)
        self.assertEqual(len(ev["users"]), len(self.events["users"]))
        self.assertEqual(store.count_items(), 40)
        self.assertEqual(len(store.fetch_eval_users()), len(self.eval_users))

    def test_time_filter(self):
        store = FileStore(self.path)
        mid = np.median(self.events["ts"])
        from datetime import datetime, timezone

        cut = datetime.fromtimestamp(float(mid), timezone.utc)
        before = store.fetch_events(None, cut)
        self.assertLess(len(before["users"]), len(self.events["users"]))
        self.assertTrue((before["ts"] < mid).all())

    def test_pipeline_writes_csv(self):
        store = FileStore(self.path)
        result = run_pipeline(store, self.cfg, model_name="popularity",
                              run_name="local_test", log=lambda *_: None)
        rows = len(store.fetch_eval_users()) * K
        self.assertEqual(result["rows"], rows)
        out = Path(store.output_dir) / f"recommendations_run{result['run_id']}.csv"
        self.assertTrue(out.exists())
        lines = out.read_text().strip().splitlines()
        self.assertEqual(len(lines), rows + 1)          # header + rows

    def test_missing_file_is_clear_error(self):
        with self.assertRaises(SystemExit):
            FileStore(Path(self.tmp.name) / "yoq.npz")


if __name__ == "__main__":
    unittest.main()
