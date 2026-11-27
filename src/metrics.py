"""Metrics that measure recommendation quality (binary relevance).

All of them are pure functions: no database, easy to test.
"""
from __future__ import annotations

import numpy as np


def dcg_at_k(recs, relevant: set, k: int) -> float:
    total = 0.0
    for i, item in enumerate(recs[:k]):
        if item in relevant:
            total += 1.0 / np.log2(i + 2)
    return total


def ndcg_at_k(recs, relevant: set, k: int = 10) -> float:
    """The main metric. Ideal DCG = all relevant items at the top."""
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg_at_k(recs, relevant, k) / ideal if ideal > 0 else 0.0


def precision_at_k(recs, relevant: set, k: int = 10) -> float:
    if k == 0:
        return 0.0
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    return len(set(recs[:k]) & relevant) / min(len(relevant), k)


def hit_rate_at_k(recs, relevant: set, k: int = 10) -> float:
    return 1.0 if set(recs[:k]) & relevant else 0.0


def average_precision_at_k(recs, relevant: set, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, item in enumerate(recs[:k]):
        if item in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)


def catalog_coverage(all_recs, n_items: int) -> float:
    """How many different items appear in the recommendations (0..1)."""
    if n_items <= 0 or not all_recs:
        return 0.0
    seen = set()
    for recs in all_recs:
        seen.update(recs)
    return len(seen) / n_items


def already_seen_share(all_recs, seen_sets) -> float:
    """Share of recommended items already seen in train (must be 0)."""
    total = hit = 0
    for recs, seen in zip(all_recs, seen_sets):
        total += len(recs)
        hit += len(set(recs) & seen)
    return hit / total if total else 0.0


def evaluate_all(recs_by_user: dict, relevant_by_user: dict, n_items: int,
                 k: int = 10) -> dict:
    """Compute every metric for one model."""
    users = [u for u in recs_by_user if relevant_by_user.get(u)]
    if not users:
        return {"users": 0}
    ndcg, prec, rec, hr, ap = [], [], [], [], []
    for u in users:
        r, rel = recs_by_user[u], relevant_by_user[u]
        ndcg.append(ndcg_at_k(r, rel, k))
        prec.append(precision_at_k(r, rel, k))
        rec.append(recall_at_k(r, rel, k))
        hr.append(hit_rate_at_k(r, rel, k))
        ap.append(average_precision_at_k(r, rel, k))
    return {
        "users": len(users),
        f"ndcg_at_{k}": float(np.mean(ndcg)),
        f"recall_at_{k}": float(np.mean(rec)),
        f"map_at_{k}": float(np.mean(ap)),
        f"precision_at_{k}": float(np.mean(prec)),
        f"hit_rate_at_{k}": float(np.mean(hr)),
        "catalog_coverage": catalog_coverage(list(recs_by_user.values()), n_items),
        "users_covered": len(recs_by_user) / max(len(users), 1),
    }
