"""End-to-end test: recommend with FakeStore (no database)."""
import unittest

from src.config import load_config
from src.pipeline import run_pipeline
from tests.helpers import make_store

K = 10


def run(model_name: str, dry_run: bool = False):
    cfg = load_config()
    cfg["models"]["als"] = {"factors": 8, "iterations": 4,
                            "regularization": 0.05, "alpha": 20.0, "seed": 42}
    store = make_store()
    result = run_pipeline(store, cfg, model_name=model_name,
                          run_name=f"test_{model_name}", dry_run=dry_run,
                          log=lambda *_: None)
    return store, result


class TestPipeline(unittest.TestCase):
    def _check_rows(self, rows, eval_users):
        self.assertEqual(len(rows), len(eval_users) * K)
        by_user = {}
        for run_id, user_id, rank, item_id in rows:
            by_user.setdefault(user_id, []).append((rank, item_id))
        self.assertEqual(set(by_user), set(eval_users.tolist()))
        for user_id, recs in by_user.items():
            ranks = sorted(r for r, _ in recs)
            items = [i for _, i in recs]
            self.assertEqual(ranks, list(range(1, K + 1)), f"user {user_id}")
            self.assertEqual(len(set(items)), K, f"user {user_id}: duplicate")

    def test_popularity_writes_valid_rows(self):
        store, result = run("popularity")
        self.assertEqual(result["rows"], len(store.eval_users) * K)
        self.assertIsNotNone(result["run_id"])
        self._check_rows(store.runs[result["run_id"]], store.eval_users)
        self.assertEqual(store.run_meta[result["run_id"]][1], "popularity")

    def test_als_never_recommends_seen_items(self):
        store, result = run("als")
        rows = store.runs[result["run_id"]]
        self._check_rows(rows, store.eval_users)

        seen = {}
        ev_users, ev_items = store.events["users"], store.events["items"]
        for u, i in zip(ev_users.tolist(), ev_items.tolist()):
            seen.setdefault(u, set()).add(i)
        for _, user_id, _, item_id in rows:
            self.assertNotIn(item_id, seen.get(user_id, set()),
                             f"user {user_id} got an item they already watched")

    def test_cold_start_user_gets_recommendations(self):
        store, result = run("popularity")
        cold = [r for r in store.runs[result["run_id"]] if r[1] == 999_999]
        self.assertEqual(len(cold), K)
        self.assertEqual(len({r[3] for r in cold}), K)

    def test_only_items_present_in_train(self):
        store, result = run("popularity")
        available = set(int(i) for i in set(store.events["items"].tolist()))
        for row in store.runs[result["run_id"]]:
            self.assertIn(row[3], available)

    def test_dry_run_does_not_create_run(self):
        store, result = run("popularity", dry_run=True)
        self.assertIsNone(result["run_id"])
        self.assertEqual(store.runs, {})
        self.assertEqual(result["rows"], len(store.eval_users) * K)


if __name__ == "__main__":
    unittest.main()
