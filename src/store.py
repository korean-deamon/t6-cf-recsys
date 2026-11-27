"""Data layer: Postgres (`raw` schema) or a local file.

Real schema (checked):
  raw.events(user_id, content_id, event_ts, event_type, watch_seconds)
      event_type: view | like | share
  raw.content(content_id, duration_s, ...)  - watch_pct = watch_seconds / duration_s
  raw.test_users(user_id)                    - users we score
  raw.data_clock(as_of)                      - which day is "now"
  raw.current_round(round_id, as_of, ...)    - the current eval round
  ml.model_runs / ml.recommendations         - we write here
  eval.score_run(run_id)                     - scoring (database only)
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

BATCH = 100_000
DEFAULT_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "raw.npz"
EVENT_VIEW, EVENT_LIKE, EVENT_SHARE = 0, 1, 2


def ts(value: str) -> datetime:
    """'2026-10-17 00:00+00' -> timezone-aware datetime."""
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------- Postgres
class PgStore:
    def __init__(self, params: dict):
        import psycopg2  # lazy: the tests run without a database

        self.conn = psycopg2.connect(**params)
        self.conn.autocommit = False
        self._seq = 0

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    # ------------------------------------------------------------ internals
    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()

    def _arrays(self, sql: str, params: tuple, ncols: int) -> list[np.ndarray]:
        """Read a big result in chunks and collect it into numpy arrays."""
        cols: list[list] = [[] for _ in range(ncols)]
        self._seq += 1
        with self.conn.cursor(name=f"t6_{self._seq}") as cur:
            cur.itersize = BATCH
            cur.execute(sql, params)
            while True:
                rows = cur.fetchmany(BATCH)
                if not rows:
                    break
                for row in rows:
                    for bucket, value in zip(cols, row):
                        bucket.append(value)
        return [np.asarray(c) for c in cols]

    # ------------------------------------------------------------ read
    def fetch_events(self, start=None, end=None, exclude_bots: bool = False) -> dict:
        """View / like / share, with watch_pct scaled to [0..1].

        With exclude_bots=True the suspicious accounts in `sandbox.t6_bots`
        are left out of training (they break the popularity signal).
        """
        bot_filter = """
               AND NOT EXISTS (SELECT 1 FROM sandbox.t6_bots b
                                WHERE b.user_id = e.user_id)
        """ if exclude_bots else ""
        sql = """
            SELECT e.user_id, e.content_id,
                   CASE e.event_type WHEN 'view' THEN 0 WHEN 'like' THEN 1 ELSE 2 END,
                   CASE WHEN e.event_type = 'view'
                        THEN LEAST(GREATEST(COALESCE(e.watch_seconds, 0)::float8
                             / GREATEST(c.duration_s, 1), 0), 1)
                        ELSE 1.0 END,
                   EXTRACT(EPOCH FROM e.event_ts)::float8
              FROM raw.events e
              JOIN raw.content c ON c.content_id = e.content_id
             WHERE (%(start)s::timestamptz IS NULL OR e.event_ts >= %(start)s::timestamptz)
               AND (%(end)s::timestamptz   IS NULL OR e.event_ts <  %(end)s::timestamptz)
        """ + bot_filter
        u, i, t, p, w = self._arrays(sql, {"start": start, "end": end}, 5)
        return {
            "users": u.astype(np.int64),
            "items": i.astype(np.int64),
            "event_type": t.astype(np.int8),
            "watch_pct": p.astype(np.float32),
            "ts": w.astype(np.float64),
        }

    def fetch_ratings(self, end=None) -> dict:
        """This database has no ratings table, so we return empty (same interface)."""
        empty = np.empty(0, dtype=np.int64)
        return {"users": empty, "items": empty,
                "rating": np.empty(0, dtype=np.float32),
                "ts": np.empty(0, dtype=np.float64)}

    def fetch_bookmarks(self, end=None) -> dict:
        """No bookmark signal either - the 'share' event is used instead."""
        empty = np.empty(0, dtype=np.int64)
        return {"users": empty, "items": empty, "ts": np.empty(0, dtype=np.float64)}

    def fetch_eval_users(self) -> np.ndarray:
        return np.asarray([r[0] for r in self.query(
            "SELECT DISTINCT user_id FROM raw.test_users ORDER BY 1")], dtype=np.int64)

    def count_items(self) -> int:
        return int(self.query("SELECT count(*) FROM raw.content")[0][0])

    def as_of(self) -> date:
        return self.query("SELECT as_of FROM raw.data_clock")[0][0]

    def current_round(self) -> dict:
        row = self.query("""SELECT round_id, as_of, test_from, test_to, n_keys
                              FROM raw.current_round ORDER BY round_id DESC LIMIT 1""")[0]
        return {"round_id": int(row[0]), "as_of": row[1],
                "test_from": row[2], "test_to": row[3], "n_keys": int(row[4])}

    # ------------------------------------------------------------ write
    def create_run(self, student_name: str, model_name: str, *, run_name: str = "",
                   params: dict | None = None, feature_list: list[str] | None = None,
                   train_from=None, train_to=None) -> int:
        from psycopg2.extras import Json

        rnd = self.current_round()
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO ml.model_runs
                (student_name, model_name, model_version, params, feature_list,
                 train_from, train_to, round_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING run_id
        """, (student_name, model_name, run_name or "v1",
              Json(params or {}), feature_list, train_from, train_to,
              rnd["round_id"]))
        run_id = int(cur.fetchone()[0])
        self.conn.commit()
        return run_id

    def insert_recommendations(self, run_id: int, rows: Iterable[tuple],
                               batch: int = 20_000) -> int:
        from psycopg2.extras import execute_values

        cur = self.conn.cursor()
        cur.execute("DELETE FROM ml.recommendations WHERE run_id = %s", (run_id,))
        total, buf = 0, []
        for row in rows:
            buf.append(tuple(row))
            if len(buf) >= batch:
                execute_values(cur,
                               "INSERT INTO ml.recommendations"
                               " (run_id, user_id, rank, content_id) VALUES %s", buf)
                total += len(buf)
                buf.clear()
        if buf:
            execute_values(cur,
                           "INSERT INTO ml.recommendations"
                           " (run_id, user_id, rank, content_id) VALUES %s", buf)
            total += len(buf)
        self.conn.commit()
        cur.execute("ANALYZE ml.recommendations")
        self.conn.commit()
        return total

    def score_run(self, run_id: int):
        cur = self.conn.cursor()
        cur.execute("SELECT metric, value, info FROM eval.score_run(%s)", (run_id,))
        rows = cur.fetchall()
        self.conn.commit()
        return rows


# ---------------------------------------------------------------- local file
class FileStore:
    """No-database mode: reads `data/raw.npz` written by `python -m src.export`."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or DEFAULT_DATA_FILE)
        if not self.path.exists():
            raise SystemExit(
                f"{self.path} not found.\nOn the university network run: python -m src.export")
        self._d = np.load(self.path)
        self._runs = 0
        self.output_dir = self.path.parent

    def _mask(self, prefix: str, start=None, end=None) -> np.ndarray:
        when = self._d[f"{prefix}_ts"]
        mask = np.ones(len(when), dtype=bool)
        if start is not None:
            mask &= when >= start.timestamp()
        if end is not None:
            mask &= when < end.timestamp()
        return mask

    def fetch_events(self, start=None, end=None, exclude_bots: bool = False) -> dict:
        m = self._mask("events", start, end)
        if exclude_bots and "bots" in self._d:
            m &= ~np.isin(self._d["events_users"], self._d["bots"])
        out = {
            "users": self._d["events_users"][m],
            "items": self._d["events_items"][m],
            "ts": self._d["events_ts"][m],
            "watch_pct": self._d["events_pct"][m],
        }
        if "events_type" in self._d:
            out["event_type"] = self._d["events_type"][m]
        return out

    def fetch_ratings(self, end=None) -> dict:
        empty = np.empty(0, dtype=np.int64)
        return {"users": empty, "items": empty,
                "rating": np.empty(0, dtype=np.float32),
                "ts": np.empty(0, dtype=np.float64)}

    def fetch_bookmarks(self, end=None) -> dict:
        empty = np.empty(0, dtype=np.int64)
        return {"users": empty, "items": empty, "ts": np.empty(0, dtype=np.float64)}

    def fetch_eval_users(self) -> np.ndarray:
        return self._d["eval_users"].astype(np.int64)

    def count_items(self) -> int:
        return int(np.asarray(self._d["n_items"]).ravel()[0])

    def as_of(self):
        return datetime.fromtimestamp(float(self._d["as_of"]), timezone.utc).date() \
            if "as_of" in self._d else None

    def current_round(self) -> dict:
        return {"round_id": -1, "as_of": self.as_of(), "test_from": None,
                "test_to": None, "n_keys": len(self.fetch_eval_users())}

    def create_run(self, student_name: str, model_name: str, **kwargs) -> int:
        self._runs += 1
        return self._runs

    def insert_recommendations(self, run_id: int, rows, batch: int = 20_000) -> int:
        path = self.output_dir / f"recommendations_run{run_id}.csv"
        count = 0
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["run_id", "user_id", "rank", "content_id"])
            for row in rows:
                writer.writerow(row)
                count += 1
        print(f"[local] {count:,} rows -> {path}")
        return count

    def score_run(self, run_id: int):
        raise SystemExit(
            "Scoring works on the university database only (eval.score_run).")


def open_store(source: str = "db", data_file: str | None = None):
    """source='db' -> Postgres, source='local' -> local file."""
    if source == "local":
        return FileStore(data_file)
    from .config import db_params

    return PgStore(db_params())
