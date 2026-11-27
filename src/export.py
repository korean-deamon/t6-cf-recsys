"""Export the data from the database to a local file (data/raw.npz).

Run it ONCE while on the university network. After that you can work
anywhere (at home, online) with `--source local`.

Usage:
  python -m src.export              # data/raw.npz
  python -m src.export --out data/raw.npz
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import ROOT, db_params, load_config
from .store import PgStore, ts


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the data to a local file")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw.npz"))
    ap.add_argument("--plain", action="store_true",
                    help="write without compression (faster, bigger file)")
    args = ap.parse_args()

    cfg = load_config()
    train_start = ts(cfg["data"]["train_start"])
    train_end = ts(cfg["data"]["train_end"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with PgStore(db_params()) as store:
        print("Loading events ...")
        events = store.fetch_events(train_start, train_end)
        print(f"  {len(events['users']):,} events ({time.time() - t0:.0f}s)")

        print("Test users, catalogue, clock ...")
        eval_users = store.fetch_eval_users()
        n_items = store.count_items()
        as_of = store.as_of()
        rnd = store.current_round()
        try:
            bots = np.array([r[0] for r in store.query(
                "SELECT user_id FROM sandbox.t6_bots")], dtype=np.int64)
        except Exception:
            store.conn.rollback()
            bots = np.empty(0, dtype=np.int64)
        print(f"  bots: {len(bots)}")

    # compact dtypes to save memory
    arrays = {
        "events_users": events["users"].astype(np.int64),
        "events_items": events["items"].astype(np.int64),
        "events_ts": events["ts"].astype(np.float64),
        "events_pct": events["watch_pct"].astype(np.float32),
        "events_type": events["event_type"].astype(np.int8),
        "eval_users": eval_users.astype(np.int64),
        "n_items": np.array([n_items], dtype=np.int64),
        "as_of": np.array([datetime.combine(as_of, datetime.min.time(),
                                            timezone.utc).timestamp()]),
        "round_id": np.array([rnd["round_id"]], dtype=np.int64),
        "bots": bots,
    }

    if args.plain:
        np.savez(out, **arrays)
    else:
        np.savez_compressed(out, **arrays)

    # check: read the file back and compare the counts
    check = np.load(out)
    assert len(check["events_users"]) == len(events["users"]), "event count does not match"
    assert len(check["eval_users"]) == len(eval_users), "eval user count does not match"
    size_mb = out.stat().st_size / 1e6

    print(f"\nDone: {out}  ({size_mb:.1f} MB, {time.time() - t0:.0f}s)")
    print(f"  events        : {len(events['users']):,}")
    print(f"  test users    : {len(eval_users):,}")
    print(f"  clock (as_of) : {as_of}")
    print(f"  catalogue     : {n_items:,}")
    print("\nNow you can work without the database:")
    print("  python -m src.evaluate --models popularity itemknn als --source local")
    print("  python -m src.recommend --run-name als_v1 --model als --source local")


if __name__ == "__main__":
    main()
