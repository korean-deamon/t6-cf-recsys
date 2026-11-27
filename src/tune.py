"""Pick the ALS hyperparameters with offline validation.

The dataset is built once, then the model is trained again for every
parameter set (nothing is written to the database). Output: reports/tuning.csv.

Usage:
  python -m src.tune --source local --sample 3000
  python -m src.tune --source local --sample 0      # all users
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from datetime import timedelta

import numpy as np

from .config import ROOT, load_config
from .data import build_dataset
from .evaluate import build_relevance, recommend_for
from .metrics import evaluate_all
from .models import build_model
from .store import open_store, ts

# (name, weights change, als parameters)
GRID = [
    ("pop06",       {}, dict(factors=32, iterations=12, alpha=20.0,
                             regularization=0.05, pop_prior=0.6)),
    ("pop09",       {}, dict(factors=32, iterations=12, alpha=20.0,
                             regularization=0.05, pop_prior=0.9)),
    ("pop12",       {}, dict(factors=32, iterations=12, alpha=20.0,
                             regularization=0.05, pop_prior=1.2)),
    ("pop09_reg02", {}, dict(factors=32, iterations=12, alpha=20.0,
                             regularization=0.02, pop_prior=0.9)),
    ("pop09_it8",   {}, dict(factors=32, iterations=8, alpha=20.0,
                             regularization=0.05, pop_prior=0.9)),
    ("pop09_a30",   {}, dict(factors=32, iterations=12, alpha=30.0,
                             regularization=0.05, pop_prior=0.9)),
    ("pop09_f48",   {}, dict(factors=48, iterations=16, alpha=20.0,
                             regularization=0.05, pop_prior=0.9)),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="ALS tuning")
    ap.add_argument("--source", choices=["db", "local"], default="local")
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--sample", type=int, default=3000)
    args = ap.parse_args()

    cfg = load_config()
    k = int(cfg["inference"]["top_k"])
    train_end = ts(cfg["data"]["train_end"])
    cutoff = train_end - timedelta(days=int(cfg["data"]["valid_last_days"]))
    exclude_bots = bool(cfg["data"].get("exclude_bots", True))

    store = open_store(args.source, args.data_file)
    try:
        t0 = time.time()
        events = store.fetch_events(ts(cfg["data"]["train_start"]), cutoff,
                                    exclude_bots=exclude_bots)
        val = store.fetch_events(cutoff, train_end, exclude_bots=exclude_bots)
        print(f"[data] {len(events['users']):,} train event ({time.time() - t0:.0f}s)")
        ds0 = build_dataset(events, None, None, cfg, ref_ts=cutoff.timestamp())

        cache: dict[str, object] = {"": ds0}

        def dataset_with(weights_delta: dict):
            key = repr(sorted(weights_delta.items()))
            if key not in cache:
                c = copy.deepcopy(cfg)
                c["weights"].update(weights_delta)
                cache[key] = build_dataset(events, None, None, c, ref_ts=cutoff.timestamp())
            return cache[key]

        rows = []
        for name, wdelta, als in GRID:
            c = copy.deepcopy(cfg)
            c["weights"].update(wdelta)
            c["models"]["als"].update(als)
            ds = dataset_with(wdelta)
            rel = build_relevance(val, ds, float(c["weights"]["min_watch_pct"]))
            users = np.array(sorted(rel), dtype=np.int64)
            if args.sample and len(users) > args.sample:
                rng = np.random.default_rng(42)
                users = np.sort(rng.choice(users, size=args.sample, replace=False))

            t1 = time.time()
            model = build_model("als", c).fit(ds)
            recs = recommend_for(model, ds, users, k)
            metric = evaluate_all(
                {int(ds.user_ids[u]): v for u, v in recs.items()},
                {int(ds.user_ids[u]): rel[u] for u in recs},
                ds.n_items, k,
            )
            secs = round(time.time() - t1, 1)
            rows.append([name, metric.get(f"ndcg_at_{k}", 0),
                         metric.get(f"recall_at_{k}", 0),
                         metric.get("catalog_coverage", 0), secs,
                         repr(wdelta), repr(als)])
            print(f"{name:<14} NDCG@{k}={rows[-1][1]:.4f}  "
                  f"Recall@{k}={rows[-1][2]:.4f}  coverage={rows[-1][3]:.3f}  {secs}s")

        rows.sort(key=lambda r: -r[1])
        out = ROOT / "reports" / "tuning.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["name", f"ndcg_at_{k}", f"recall_at_{k}",
                             "catalog_coverage", "seconds", "weights", "als"])
            writer.writerows(rows)
        print(f"\nBest: {rows[0][0]} (NDCG@{k}={rows[0][1]:.4f})")
        print(f"Table: {out.relative_to(ROOT)}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
