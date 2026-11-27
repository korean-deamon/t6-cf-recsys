"""Data audit: quality, distribution, sparsity, anomalies.

Usage:
  python -m src.audit                 # university database
  python -m src.audit --source local  # data/raw.npz (no database)
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from .config import load_config
from .store import FileStore, open_store

QUERIES = [
    ("Total size", """
        SELECT count(*) AS events, count(DISTINCT user_id) AS users,
               count(DISTINCT content_id) AS contents,
               min(event_ts)::date AS first_day, max(event_ts)::date AS last_day
          FROM raw.events"""),
    ("Event types", """
        SELECT event_type, count(*) AS events,
               count(DISTINCT user_id) AS users,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
          FROM raw.events GROUP BY 1 ORDER BY 2 DESC"""),
    ("Watch_pct distribution (view)", """
        SELECT CASE WHEN p < 0.1 THEN '0.0-0.1'
                    WHEN p < 0.3 THEN '0.1-0.3'
                    WHEN p < 0.6 THEN '0.3-0.6'
                    WHEN p < 0.9 THEN '0.6-0.9'
                    ELSE '0.9-1.0' END AS bucket,
               count(*) AS events,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct
          FROM (SELECT LEAST(GREATEST(COALESCE(e.watch_seconds,0)::numeric
                              / GREATEST(c.duration_s,1), 0), 1) AS p
                  FROM raw.events e JOIN raw.content c USING (content_id)
                 WHERE e.event_type = 'view') t
         GROUP BY 1 ORDER BY 1"""),
    ("Duplicates (user, content, time)", """
        SELECT count(*) FROM (
            SELECT user_id, content_id, event_ts, count(*) c
              FROM raw.events GROUP BY 1,2,3 HAVING count(*) > 1) t"""),
    ("Top 10 most active users", """
        SELECT user_id, count(*) AS events, count(DISTINCT content_id) AS contents,
               count(DISTINCT event_ts::date) AS days
          FROM raw.events GROUP BY 1 ORDER BY 2 DESC LIMIT 10"""),
    ("Content never watched", """
        SELECT count(*) FROM raw.content c
         WHERE NOT EXISTS (SELECT 1 FROM raw.events e
                            WHERE e.content_id = c.content_id AND e.event_type='view')"""),
    ("Sparsity", """
        SELECT count(DISTINCT user_id)::bigint * count(DISTINCT content_id) AS cells,
               count(*) AS filled,
               round(count(*)::numeric / (count(DISTINCT user_id)::numeric
                     * count(DISTINCT content_id)), 6) AS density
          FROM raw.events"""),
    ("Test users", """
        SELECT count(*) AS test_users FROM raw.test_users"""),
    ("Activity of the last 14 days", """
        SELECT event_ts::date AS day, count(*) AS events,
               count(DISTINCT user_id) AS users
          FROM raw.events
         WHERE event_ts >= (SELECT as_of FROM raw.data_clock) - 14
         GROUP BY 1 ORDER BY 1"""),
    ("Clock and round", """
        SELECT d.as_of, d.last_update_at, r.round_id, r.test_from, r.test_to, r.n_keys
          FROM raw.data_clock d CROSS JOIN raw.current_round r"""),
    ("Suspicious bots (sandbox)", """
        SELECT count(*) AS bots, max(n_events) AS max_events
          FROM sandbox.t6_bots"""),
]

BUCKETS = [(0.0, 0.1, "0.0-0.1"), (0.1, 0.3, "0.1-0.3"), (0.3, 0.6, "0.3-0.6"),
           (0.6, 0.9, "0.6-0.9"), (0.9, 1.01, "0.9-1.0")]
TYPE_NAMES = {0: "view", 1: "like", 2: "share"}


def audit_db() -> None:
    store = open_store("db")
    try:
        for title, sql in QUERIES:
            print(f"### {title}")
            try:
                rows = store.query(sql)
            except Exception as exc:
                store.conn.rollback()
                print(f"  ! error: {str(exc).splitlines()[0]}\n")
                continue
            if not rows:
                print("  (empty)\n")
                continue
            for row in rows[:15]:
                print("  " + " | ".join(_fmt(v) for v in row))
            print()
    finally:
        store.close()


def audit_local(cfg: dict, data_file: str | None = None) -> None:
    store = FileStore(data_file)
    ev = store.fetch_events(None, None)
    pct_all = ev["watch_pct"].astype(np.float64)
    etype = np.asarray(ev.get("event_type", np.zeros(len(pct_all), np.int8)))
    is_view = etype == 0
    pct = pct_all[is_view]
    n_events = len(ev["users"])
    users, user_counts = np.unique(ev["users"], return_counts=True)
    items = np.unique(ev["items"])
    n_items = store.count_items()

    print(f"Source: {store.path}\n")
    print("### Total size")
    print(f"  events={n_events:,} | users={len(users):,} | contents={len(items):,} | "
          f"catalogue={n_items:,}")
    print(f"  period: {_day(ev['ts'].min())} .. {_day(ev['ts'].max())}\n")

    print("### Event types")
    for code, count in zip(*np.unique(etype, return_counts=True)):
        print(f"  {TYPE_NAMES.get(int(code), code):<6} {count:>9,}  "
              f"({100 * count / n_events:5.2f}%)")
    print()

    print("### Watch_pct distribution (view)")
    for lo, hi, label in BUCKETS:
        m = (pct >= lo) & (pct < hi)
        print(f"  {label:<8} {m.mean() * 100:6.2f}%  ({int(m.sum()):,})")
    print(f"  mean watch_pct = {pct.mean():.3f}\n")

    print("### Top 10 most active users")
    for idx in np.argsort(-user_counts)[:10]:
        mask = ev["users"] == users[idx]
        print(f"  user {users[idx]}: {user_counts[idx]:,} event, "
              f"{len(np.unique(ev['items'][mask])):,} content")
    print()

    print("### Content never watched")
    viewed = np.unique(ev["items"][is_view])
    print(f"  {n_items - len(viewed):,} / {n_items:,}\n")

    print("### Sparsity")
    cells = len(users) * len(items)
    print(f"  cells={cells:,} | filled={n_events:,} | "
          f"density={n_events / max(cells, 1):.6f}\n")

    print("### Test users")
    print(f"  {len(store.fetch_eval_users()):,}")


def _day(epoch: float) -> str:
    return datetime.fromtimestamp(float(epoch), timezone.utc).date().isoformat()


def _fmt(value) -> str:
    if value is None:
        return "-"
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def main() -> None:
    ap = argparse.ArgumentParser(description="Data audit")
    ap.add_argument("--source", choices=["db", "local"], default="db")
    ap.add_argument("--data-file", default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.source == "local":
        audit_local(cfg, args.data_file)
    else:
        audit_db()


if __name__ == "__main__":
    main()
