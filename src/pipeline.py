"""Main flow: connect -> data -> model -> top-10 -> write to the database.

`select_top_k` is a pure function (tested): it removes seen items, drops
items that never appear in train and returns exactly 10 items.
"""
from __future__ import annotations

import time

import numpy as np

from .data import build_dataset
from .models import build_model
from .store import ts


def select_top_k(scores: np.ndarray, seen, k: int,
                 fallback_items: np.ndarray | None = None,
                 fallback_scores: np.ndarray | None = None):
    """Top-k items for one user.

    scores         - score for every item (-inf = not possible)
    seen           - item codes seen in train (removed from the list)
    fallback_items - if 10 is not enough (new user), fill in this order
    """
    s = np.asarray(scores, dtype=np.float32).copy()
    seen_arr = np.asarray(seen, dtype=np.int64) if seen is not None else np.empty(0, np.int64)
    if seen_arr.size:
        s[seen_arr] = -np.inf

    chosen: list[int] = []
    kk = min(k, int(np.isfinite(s).sum()))
    if kk > 0:
        idx = np.argpartition(-s, kk - 1)[:kk]
        idx = idx[np.lexsort((idx, -s[idx]))]        # equal scores: order by item_id
        chosen = [int(i) for i in idx if s[i] > -np.inf]

    seen_set = set(seen_arr.tolist())
    if len(chosen) < k and fallback_items is not None:
        picked = set(chosen)
        for cand, val in zip(fallback_items, fallback_scores):
            if len(chosen) >= k:
                break
            cand = int(cand)
            if cand in picked or cand in seen_set:
                continue
            chosen.append(cand)
            picked.add(cand)
            s[cand] = float(val)

    items = np.asarray(chosen, dtype=np.int64)
    return items, s[items] if items.size else np.empty(0, dtype=np.float32)


def run_pipeline(store, cfg: dict, model_name: str = "als", run_name: str = "run",
                 student_name: str = "team", limit_users: int | None = None,
                 dry_run: bool = False, log=print) -> dict:
    t0 = time.time()
    data_cfg = cfg["data"]
    train_end = ts(data_cfg["train_end"])
    train_start = ts(data_cfg["train_start"])

    events = store.fetch_events(
        train_start, train_end,
        exclude_bots=bool(data_cfg.get("exclude_bots", True)),
    )
    ratings = store.fetch_ratings(train_end)
    bookmarks = store.fetch_bookmarks(train_end)
    eval_users = store.fetch_eval_users()
    if limit_users:
        eval_users = eval_users[:limit_users]

    log(f"[data] {len(events['users']):,} event · {len(eval_users):,} eval user")
    if "event_type" in events:
        types, counts = np.unique(np.asarray(events["event_type"]), return_counts=True)
        names = {0: "view", 1: "like", 2: "share"}
        log("[data] " + " · ".join(f"{names.get(int(t), t)}: {int(c):,}"
                                   for t, c in zip(types, counts)))

    ds = build_dataset(events, ratings, bookmarks, cfg,
                       ref_ts=train_end.timestamp())
    log(f"[data] matrix {ds.n_users:,} x {ds.n_items:,}, "
        f"{len(ds.matrix):,} pairs "
        f"(density {len(ds.matrix) / max(ds.n_users * ds.n_items, 1):.3%})")

    t1 = time.time()
    model = build_model(model_name, cfg)
    model.fit(ds)
    log(f"[{model_name}] fit {time.time() - t1:.1f}s")

    codes = ds.user_codes(eval_users)
    k = int(cfg["inference"]["top_k"])
    chunk = int(cfg["inference"]["user_chunk"])

    valid_items = ds.pop_all > 0
    fallback_items = np.argsort(-ds.pop_recent, kind="stable")
    fallback_scores = ds.pop_recent[fallback_items]

    def row_stream(run_id: int):
        for start in range(0, len(eval_users), chunk):
            uc = codes[start:start + chunk]
            scores = np.zeros((len(uc), ds.n_items), dtype=np.float32)
            known = uc >= 0
            if known.any():
                scores[known] = model.score(uc[known])
            if (~known).any():                      # a totally new user
                scores[~known] = ds.pop_recent
            scores[:, ~valid_items] = -np.inf       # items that never appear in train
            for i in range(len(uc)):
                items, vals = select_top_k(
                    scores[i], ds.seen_of(int(uc[i])), k,
                    fallback_items, fallback_scores,
                )
                user_id = int(eval_users[start + i])
                for rank, item in enumerate(items, start=1):
                    yield (run_id, user_id, rank, int(ds.item_ids[item]))

    run_id = None
    written = 0
    if dry_run:
        written = sum(1 for _ in row_stream(0))
        log(f"[dry-run] {written:,} rows would be written (database not touched)")
    else:
        run_id = store.create_run(
            student_name, model_name, run_name=run_name,
            params=dict(cfg["models"].get(model_name) or {}),
            feature_list=["views", "watch_pct", "likes", "shares"],
            train_from=train_start.date(), train_to=train_end.date(),
        )
        written = store.insert_recommendations(run_id, row_stream(run_id))
        log(f"[write] run_id={run_id} - {written:,} rows written")

    return {
        "run_id": run_id,
        "model": model_name,
        "rows": written,
        "users": len(eval_users),
        "seconds": round(time.time() - t0, 1),
    }
