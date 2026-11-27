"""Offline validation - compare models without touching the leaderboard.

The task asks for this: take the last 14 days of train and repeat the test
rule exactly (2026-10-04 .. 10-17).

Usage:
  python -m src.evaluate --models popularity itemknn als --sample 3000
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import timedelta

import numpy as np

from .config import ROOT, load_config
from .data import build_dataset
from .metrics import already_seen_share, evaluate_all
from .models import build_model
from .pipeline import select_top_k
from .store import open_store, ts


def build_relevance(val: dict, ds, min_pct: float) -> dict:
    """Relevant content in the test window: first view + watch_pct threshold."""
    uc = ds.user_codes(val["users"])
    ic = ds.item_codes(val["items"])
    keep = (uc >= 0) & (ic >= 0)
    if "event_type" in val:                 # only views can be relevant
        keep &= np.asarray(val["event_type"]) == 0
    uc, ic = uc[keep], ic[keep]
    pct, when = val["watch_pct"][keep], val["ts"][keep]
    if len(uc) == 0:
        return {}

    key = uc * ds.n_items + ic
    order = np.argsort(when, kind="stable")
    keys = key[order]
    first = np.concatenate(([True], keys[1:] != keys[:-1]))
    sel = order[first]
    sel = sel[pct[sel] >= min_pct]
    sel = sel[ds.pop_all[ic[sel]] > 0]      # items missing from train are not useful

    rel: dict[int, set[int]] = {}
    for u, i in zip(uc[sel], ic[sel]):
        rel.setdefault(int(u), set()).add(int(i))
    return rel


def recommend_for(model, ds, users: np.ndarray, k: int, chunk: int = 2000) -> dict:
    valid = ds.pop_all > 0
    fallback_items = np.argsort(-ds.pop_recent, kind="stable")
    fallback_scores = ds.pop_recent[fallback_items]
    out: dict[int, list[int]] = {}
    for start in range(0, len(users), chunk):
        uc = users[start:start + chunk]
        scores = model.score(uc)
        scores[:, ~valid] = -np.inf
        for i, code in enumerate(uc):
            items, _ = select_top_k(scores[i], ds.seen_of(int(code)), k,
                                    fallback_items, fallback_scores)
            out[int(code)] = items.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline validation")
    ap.add_argument("--models", nargs="+", default=["popularity", "itemknn", "als"])
    ap.add_argument("--sample", type=int, default=3000,
                    help="how many users to measure (0 = all)")
    ap.add_argument("--source", choices=["db", "local"], default="db",
                    help="db = university database, local = data/raw.npz")
    ap.add_argument("--data-file", default=None, help="path to the local file")
    args = ap.parse_args()

    cfg = load_config()
    k = int(cfg["inference"]["top_k"])
    train_end = ts(cfg["data"]["train_end"])
    cutoff = train_end - timedelta(days=int(cfg["data"]["valid_last_days"]))

    store = open_store(args.source, args.data_file)
    try:
        print(f"Train: {cfg['data']['train_start']} .. {cutoff.date()}")
        print(f"Validation: {cutoff.date()} .. {train_end.date()}")

        exclude_bots = bool(cfg["data"].get("exclude_bots", True))
        events = store.fetch_events(ts(cfg["data"]["train_start"]), cutoff,
                                    exclude_bots=exclude_bots)
        val = store.fetch_events(cutoff, train_end, exclude_bots=exclude_bots)
        ratings = store.fetch_ratings(cutoff)
        bookmarks = store.fetch_bookmarks(cutoff)

        ds = build_dataset(events, ratings, bookmarks, cfg,
                           ref_ts=cutoff.timestamp())
        print(f"Train matrix: {ds.n_users:,} x {ds.n_items:,}, "
              f"{len(ds.matrix):,} pairs")

        rel = build_relevance(val, ds, float(cfg["weights"]["min_watch_pct"]))
        users = np.array(sorted(rel), dtype=np.int64)
        rng = np.random.default_rng(42)
        if args.sample and len(users) > args.sample:
            users = np.sort(rng.choice(users, size=args.sample, replace=False))
        avg_rel = np.mean([len(v) for v in rel.values()])
        print(f"Users we score: {len(users):,} "
              f"(about {avg_rel:.1f} relevant items each)\n")

        results, table = [], []
        for name in args.models:
            t0 = time.time()
            model = build_model(name, cfg).fit(ds)
            recs = recommend_for(model, ds, users, k)
            metric = evaluate_all(
                {int(ds.user_ids[u]): v for u, v in recs.items()},
                {int(ds.user_ids[u]): rel[u] for u in recs},
                ds.n_items, k,
            )
            seen_sets = [set(ds.seen_of(int(u)).tolist()) for u in users]
            metric["already_seen_share"] = already_seen_share(
                list(recs.values()), [set(ds.item_ids[list(s)].tolist()) if s else set()
                                      for s in seen_sets]
            )
            metric["seconds"] = round(time.time() - t0, 1)
            metric["model"] = name
            results.append(metric)
            row = [name, metric.get(f"ndcg_at_{k}", 0), metric.get(f"recall_at_{k}", 0),
                   metric.get("catalog_coverage", 0), metric.get("already_seen_share", 0),
                   metric["seconds"]]
            table.append(row)
            print(f"{name:<12} NDCG@{k}={row[1]:.4f}  Recall@{k}={row[2]:.4f}  "
                  f"coverage={row[3]:.3f}  seen={row[4]:.4f}  {row[5]}s")

        out_csv = ROOT / "reports" / f"offline_models.csv"
        out_csv.parent.mkdir(exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([f"model", f"ndcg_at_{k}", f"recall_at_{k}",
                             "catalog_coverage", "already_seen_share", "seconds"])
            writer.writerows(table)
        print(f"\nTable saved: {out_csv.relative_to(ROOT)}")

        best = max(results, key=lambda r: r.get(f"ndcg_at_{k}", 0))
        print(f"Best model: {best['model']} "
              f"(NDCG@{k} = {best.get(f'ndcg_at_{k}'):.4f})")
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
