"""A fake store and made-up data, so the tests run without a database.

Every user belongs to one "interest group" and watches content from that
group only. The model has to find this structure.
"""
from __future__ import annotations

import numpy as np

N_CLUSTERS = 4
CLUSTER_SIZE = 10


def make_events(n_users: int = 60, per_user: int = 8, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    base = 1.78e9
    users, items, when, pct = [], [], [], []
    for u in range(n_users):
        cluster = u % N_CLUSTERS
        pool = np.arange(cluster * CLUSTER_SIZE, (cluster + 1) * CLUSTER_SIZE)
        for item in rng.choice(pool, size=per_user, replace=False):
            users.append(1000 + u)
            items.append(5000 + int(item))
            when.append(base + rng.integers(0, 10_000_000))
            pct.append(rng.uniform(0.5, 1.0))
    return {
        "users": np.array(users, dtype=np.int64),
        "items": np.array(items, dtype=np.int64),
        "ts": np.array(when, dtype=np.float64),
        "watch_pct": np.array(pct, dtype=np.float64),
    }


class FakeStore:
    """Same interface as PgStore, but kept in memory."""

    def __init__(self, events: dict, eval_users: np.ndarray):
        self.events = events
        self.eval_users = eval_users
        self.runs: dict[int, list[tuple]] = {}
        self.run_meta: dict[int, tuple] = {}

    def fetch_events(self, start=None, end=None, exclude_bots: bool = False):
        return self.events

    def fetch_ratings(self, end=None):
        empty = np.empty(0)
        return {"users": empty.astype(np.int64), "items": empty.astype(np.int64),
                "rating": empty, "ts": empty}

    def fetch_bookmarks(self, end=None):
        empty = np.empty(0)
        return {"users": empty.astype(np.int64), "items": empty.astype(np.int64),
                "ts": empty}

    def fetch_eval_users(self):
        return self.eval_users

    def create_run(self, student_name: str, model_name: str, **kwargs) -> int:
        run_id = len(self.runs) + 1
        self.runs[run_id] = []
        self.run_meta[run_id] = (student_name, model_name)
        return run_id

    def insert_recommendations(self, run_id: int, rows, batch: int = 1000) -> int:
        data = list(rows)
        self.runs[run_id].extend(data)
        return len(data)


def make_store(seed: int = 0, n_users: int = 60) -> FakeStore:
    events = make_events(n_users=n_users, seed=seed)
    eval_users = np.array([1000 + u for u in range(n_users)] + [999_999],
                          dtype=np.int64)   # the last one is brand new (cold start)
    return FakeStore(events, eval_users)
