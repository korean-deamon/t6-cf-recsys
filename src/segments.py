"""Segment analysis: on which groups of users and content does the model do well.

The rubric asks for this ("Report and segment analysis"): one overall NDCG is
not enough - we have to show the segments where quality is low.

Segments:
  * by user activity (number of events in train): low / middle / high
  * by content popularity (14 days before validation): head / mid / tail
  * by user history: cold start / has history

Usage:
  python -m src.segments --source local
  python -m src.segments --source db --sample 4000
"""
from __future__ import annotations

import argparse
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


def _ndcg(recs: dict, rel: dict, ds, users: np.ndarray, k: int) -> float:
    if len(users) == 0:
        return float("nan")
    m = evaluate_all(
        {int(ds.user_ids[u]): recs[u] for u in users},
        {int(ds.user_ids[u]): rel[u] for u in users},
        ds.n_items, k,
    )
    return float(m.get(f"ndcg_at_{k}", float("nan")))


def main() -> None:
    ap = argparse.ArgumentParser(description="Segment analysis")
    ap.add_argument("--source", choices=["db", "local"], default="local")
    ap.add_argument("--data-file", default=None)
    ap.add_argument("--model", default="als")
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
        ds = build_dataset(events, None, None, cfg, ref_ts=cutoff.timestamp())
        rel = build_relevance(val, ds, float(cfg["weights"]["min_watch_pct"]))
        print(f"[data] {time.time() - t0:.0f}s - "
              f"{ds.n_users:,}x{ds.n_items:,} - {len(rel):,} users to score")

        users = np.array(sorted(rel), dtype=np.int64)
        if args.sample and len(users) > args.sample:
            rng = np.random.default_rng(42)
            users = np.sort(rng.choice(users, size=args.sample, replace=False))

        t1 = time.time()
        model = build_model(args.model, cfg).fit(ds)
        fit_s = time.time() - t1
        t2 = time.time()
        recs = recommend_for(model, ds, users, k)
        infer_s = time.time() - t2

        overall = _ndcg(recs, rel, ds, users, k)
        rows = [["all", "everyone", len(users), overall]]

        # 1) user activity: terciles of the event count in train
        ucodes, ucounts = np.unique(ds.user_codes(np.asarray(events["users"])),
                                    return_counts=True)
        user_events = {int(c): int(n) for c, n in zip(ucodes, ucounts) if c >= 0}
        activity = np.array([user_events.get(int(u), 0) for u in users])
        q1, q2 = np.quantile(activity, [1 / 3, 2 / 3])
        groups = [("activity: low", activity <= q1),
                  ("activity: middle", (activity > q1) & (activity <= q2)),
                  ("activity: high", activity > q2)]
        for name, mask in groups:
            rows.append(["user", name, int(mask.sum()),
                         _ndcg(recs, rel, ds, users[mask], k)])

        # 2) content popularity: terciles of pop_recent
        order = np.argsort(-ds.pop_recent, kind="stable")
        rank_of = np.empty(ds.n_items, dtype=np.int64)
        rank_of[order] = np.arange(ds.n_items)
        n = ds.n_items
        rec_mat = np.array([[rank_of[i] for i in recs[int(u)]] for u in users],
                           dtype=np.int64)
        head = float(np.mean(rec_mat < n / 3))
        mid = float(np.mean((rec_mat >= n / 3) & (rec_mat < 2 * n / 3)))
        tail = float(np.mean(rec_mat >= 2 * n / 3))
        rows.append(["content", "head share (top-10)", len(users), head])
        rows.append(["content", "mid share (top-10)", len(users), mid])
        rows.append(["content", "tail share (top-10)", len(users), tail])

        # 3) cold start: users that never appear in train
        known = ds.user_codes(users) >= 0
        if (~known).any():
            rows.append(["cold start", "not in train", int((~known).sum()),
                         _ndcg(recs, rel, ds, users[~known], k)])
        rows.append(["cold start", "has history", int(known.sum()),
                     _ndcg(recs, rel, ds, users[known], k)])

        print(f"\n{args.model}: NDCG@{k} by segment "
              f"(fit {fit_s:.1f}s, inference {infer_s:.2f}s, "
              f"{len(users) / max(infer_s, 1e-9):,.0f} user/s)\n")
        for kind, name, n_users, value in rows:
            print(f"  {kind:<12} {name:<24} n={n_users:>6,}  {value:.4f}")

        out = ROOT / "reports" / "segments.csv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["segment_turi", "segment", "n_users", f"ndcg_at_{k}"])
            writer.writerows(rows)
        print(f"\nTable: {out.relative_to(ROOT)}")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
