"""Build the user x item matrix from the data.

Three signals are merged here:
  view (watch_pct) + rating + bookmark,
then a time decay is applied.

The result is a sparse matrix (existing pairs only) plus the "seen" list
(used to remove items from the recommendations).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAY = 86_400.0
EVENT_VIEW, EVENT_LIKE, EVENT_SHARE = 0, 1, 2


# ---------------------------------------------------------------- sparse matrix
class Matrix:
    """Sparse matrix in COO form plus small grouping helpers.

    No scipy needed: ALS works with this form, and item-kNN converts it
    to CSR only when it has to.
    """

    def __init__(self, row: np.ndarray, col: np.ndarray, val: np.ndarray,
                 n_rows: int, n_cols: int):
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)

        # the same (user, item) pair is added together
        key = row.astype(np.int64) * self.n_cols + col.astype(np.int64)
        if len(key) == 0:                       # every weight fell to zero
            self.row = np.empty(0, dtype=np.int32)
            self.col = np.empty(0, dtype=np.int32)
            self.val = np.empty(0, dtype=np.float32)
            return
        order = np.argsort(key, kind="stable")
        key, val = key[order], val[order].astype(np.float32)
        starts = np.concatenate(([0], np.flatnonzero(key[1:] != key[:-1]) + 1))
        self.val = np.add.reduceat(val, starts)
        key = key[starts]

        self.row = (key // self.n_cols).astype(np.int32)
        self.col = (key % self.n_cols).astype(np.int32)

    def __len__(self) -> int:
        return len(self.val)

    # grouping: which columns belong to one user
    def _bounds(self, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        change = np.flatnonzero(codes[1:] != codes[:-1]) + 1
        return codes[np.concatenate(([0], change))], \
            np.concatenate(([0], change)), \
            np.concatenate((change, [len(codes)]))

    def group_rows(self):
        """(user_code, item_codes, weights) for ALS.

        The matrix is sorted by (row, col), so only the columns need sorting.
        """
        keys, starts, ends = self._bounds(self.row)
        for k, a, b in zip(keys, starts, ends):
            yield int(k), self.col[a:b], self.val[a:b]

    def group_cols(self):
        order = np.argsort(self.col, kind="stable")
        col_sorted = self.col[order]
        keys, starts, ends = self._bounds(col_sorted)
        for k, a, b in zip(keys, starts, ends):
            yield int(k), self.row[order[a:b]], self.val[order[a:b]]

    def col_sums(self) -> np.ndarray:
        return np.bincount(self.col, weights=self.val,
                           minlength=self.n_cols).astype(np.float32)


# ---------------------------------------------------------------- dataset
@dataclass
class Dataset:
    user_ids: np.ndarray          # code -> original user_id
    item_ids: np.ndarray          # code -> original item_id
    matrix: Matrix
    seen_items: np.ndarray        # item codes, sorted by user code
    seen_starts: np.ndarray       # start index per user (n_users+1)
    pop_all: np.ndarray           # total weight per item (marks valid items)
    pop_recent: np.ndarray        # unique users in the last N days
    n_users: int = 0
    n_items: int = 0
    extra: dict = field(default_factory=dict)

    # coding helpers
    def user_codes(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.user_ids, ids)
        pos = np.clip(pos, 0, max(self.n_users - 1, 0))
        return np.where(self.user_ids[pos] == ids, pos, -1).astype(np.int64)

    def item_codes(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        pos = np.searchsorted(self.item_ids, ids)
        pos = np.clip(pos, 0, max(self.n_items - 1, 0))
        return np.where(self.item_ids[pos] == ids, pos, -1).astype(np.int64)

    def seen_of(self, user_code: int) -> np.ndarray:
        if user_code < 0:
            return np.empty(0, dtype=np.int32)
        return self.seen_items[self.seen_starts[user_code]:
                               self.seen_starts[user_code + 1]]


def _codes(sorted_ids: np.ndarray, ids: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(sorted_ids, ids)
    pos = np.clip(pos, 0, max(len(sorted_ids) - 1, 0))
    return pos.astype(np.int64)


def _event_pct(events: dict) -> np.ndarray:
    """watch_pct: use it if given, else watch_seconds / duration_s
    (this is the form the real database returns)."""
    if "watch_pct" in events:
        return np.asarray(events["watch_pct"], dtype=np.float64)
    pct = np.divide(events["watch_seconds"].astype(np.float64),
                    np.maximum(events["duration_s"].astype(np.float64), 1.0),
                    out=np.zeros(len(events["users"]), dtype=np.float64))
    return np.clip(pct, 0.0, 1.0)


def build_dataset(events: dict, ratings: dict | None, bookmarks: dict | None,
                  cfg: dict, ref_ts: float) -> Dataset:
    """Merge all signals into one matrix.

    Signals: view (watch_pct), like, share - these are the three in the real database.
    `ratings` / `bookmarks` are optional (kept for the old interface).

    cfg - the whole config.yaml dict.
    ref_ts - the time we treat as "now" (end of train), decay is counted from it.
    """
    w = cfg["weights"]
    scheme = str(w.get("scheme", "A")).upper()
    half_life_days = max(float(w.get("half_life_days", 180)), 1e-9)
    recent_ts = ref_ts - float(cfg["models"]["popularity"]["recent_days"]) * DAY

    u_all, i_all = [events["users"]], [events["items"]]
    for src in (ratings, bookmarks):
        if src is not None and len(src["users"]):
            u_all.append(src["users"])
            i_all.append(src["items"])
    user_ids = np.unique(np.concatenate(u_all))
    item_ids = np.unique(np.concatenate(i_all))
    n_users, n_items = len(user_ids), len(item_ids)

    etype = events.get("event_type")
    etype = (np.zeros(len(events["users"]), dtype=np.int8) if etype is None
             else np.asarray(etype, dtype=np.int8))
    is_view = etype == EVENT_VIEW
    pct = _event_pct(events)

    # --- views: sum up per (user, item)
    v_idx = np.flatnonzero(is_view)
    v_key = (_codes(user_ids, events["users"][v_idx]) * n_items
             + _codes(item_ids, events["items"][v_idx]))
    q = pct[v_idx]
    if scheme == "B":
        q = (q >= float(w["min_watch_pct"])).astype(np.float64)
    else:
        q = np.where(q >= float(w["min_watch_pct"]), q, 0.0)

    if len(v_key):
        order = np.argsort(v_key, kind="stable")
        key_s = v_key[order]
        starts = np.concatenate(([0], np.flatnonzero(key_s[1:] != key_s[:-1]) + 1))
        uniq = key_s[starts]
        n_views = np.diff(np.concatenate((starts, [len(key_s)])))
        q_sum = np.add.reduceat(q[order], starts)
        last_ts = np.maximum.reduceat(events["ts"][v_idx][order], starts)
    else:
        uniq = np.empty(0, dtype=np.int64)
        n_views = np.empty(0, dtype=np.int64)
        q_sum = np.empty(0, dtype=np.float64)
        last_ts = np.empty(0, dtype=np.float64)

    # --- ratings
    r_key = r_val = r_ts = np.empty(0)
    if ratings is not None and len(ratings["users"]):
        r_key = _codes(user_ids, ratings["users"]) * n_items + _codes(item_ids, ratings["items"])
        order = np.argsort(r_key, kind="stable")
        r_key = r_key[order]
        r_val = ratings["rating"][order]
        r_ts = ratings["ts"][order]
        rst = np.concatenate(([0], np.flatnonzero(r_key[1:] != r_key[:-1]) + 1))
        r_key, r_val, r_ts = r_key[rst], r_val[rst], r_ts[rst]

    # --- bookmarks
    b_key = np.empty(0)
    if bookmarks is not None and len(bookmarks["users"]):
        b_key = _codes(user_ids, bookmarks["users"]) * n_items + _codes(item_ids, bookmarks["items"])
        b_key = np.unique(b_key)

    # --- like / share: each is its own signal, no watch_pct threshold
    x_idx = np.flatnonzero(~is_view)
    x_key = (_codes(user_ids, events["users"][x_idx]) * n_items
             + _codes(item_ids, events["items"][x_idx]))
    x_w = np.where(etype[x_idx] == EVENT_LIKE,
                   float(w.get("like", 0.5)),
                   float(w.get("share", 1.0))).astype(np.float64)
    if len(x_key):
        order = np.argsort(x_key, kind="stable")
        x_key_s = x_key[order]
        x_starts = np.concatenate(([0], np.flatnonzero(x_key_s[1:] != x_key_s[:-1]) + 1))
        x_uniq = x_key_s[x_starts]
        x_sum = np.add.reduceat(x_w[order], x_starts)
        x_ts = np.maximum.reduceat(events["ts"][x_idx][order], x_starts)
    else:
        x_uniq = np.empty(0, dtype=np.int64)
        x_sum = np.empty(0, dtype=np.float64)
        x_ts = np.empty(0, dtype=np.float64)

    # --- merge all keys
    all_keys = np.concatenate([uniq, r_key, b_key, x_uniq]).astype(np.int64)
    keys, inv = np.unique(all_keys, return_inverse=True)
    weight = np.zeros(len(keys), dtype=np.float64)
    stamp = np.full(len(keys), -np.inf)

    # view share (mean) * repeat bonus
    mean_q = np.divide(q_sum, n_views, out=np.zeros_like(q_sum), where=n_views > 0)
    quality = (np.full(len(mean_q), float(w.get("binary_watch", 1.0)))
               if scheme == "B" else mean_q)
    view_w = float(w.get("watch", 1.0)) * quality * (
        1.0 + float(w.get("repeat", 0.5)) * np.log1p(n_views - 1))

    n_uniq, n_rat, n_bm, n_x = len(uniq), len(r_key), len(b_key), len(x_uniq)
    if n_uniq:
        np.add.at(weight, inv[:n_uniq], view_w)
        np.maximum.at(stamp, inv[:n_uniq], last_ts)
    if n_rat:
        np.add.at(weight, inv[n_uniq:n_uniq + n_rat],
                  float(w.get("rating_scale", 0.2)) * r_val)
        np.maximum.at(stamp, inv[n_uniq:n_uniq + n_rat], r_ts)
    if n_bm:
        np.add.at(weight, inv[n_uniq + n_rat:n_uniq + n_rat + n_bm],
                  float(w.get("bookmark", 1.0)))
    if n_x:
        np.add.at(weight, inv[len(inv) - n_x:], x_sum)
        np.maximum.at(stamp, inv[len(inv) - n_x:], x_ts)

    # time decay - only for pairs with a known timestamp
    # (a bookmark without a view stays as it is)
    known_time = np.isfinite(stamp)
    age_days = np.clip(ref_ts - np.where(known_time, stamp, ref_ts), 0, None) / DAY
    decay = np.where(known_time, 0.5 ** (age_days / half_life_days), 1.0)
    weight = weight * decay

    keep = weight > 0
    keys, weight = keys[keep], weight[keep]

    row = (keys // n_items).astype(np.int32)
    col = (keys % n_items).astype(np.int32)
    matrix = Matrix(row, col, weight.astype(np.float32), n_users, n_items)

    # --- "seen" set: any interaction at all, whatever the pct
    seen_user = _codes(user_ids, events["users"]).astype(np.int64)
    seen_item = _codes(item_ids, events["items"]).astype(np.int32)
    order = np.argsort(seen_user, kind="stable")
    seen_item = seen_item[order]
    seen_user = seen_user[order]
    seen_starts = np.concatenate(([0], np.cumsum(np.bincount(seen_user,
                                                             minlength=n_users)).astype(np.int64)))

    # --- popularity: how many unique users watched it in the last N days
    recent_mask = events["ts"] >= recent_ts
    if recent_mask.any():
        pair = np.unique(seen_user[recent_mask] * n_items + seen_item[recent_mask])
    else:
        pair = np.empty(0, dtype=np.int64)
    pop_recent = np.bincount(pair % n_items, minlength=n_items).astype(np.float32)

    pop_all = np.bincount(matrix.col, weights=matrix.val,
                          minlength=n_items).astype(np.float32)

    return Dataset(
        user_ids=user_ids,
        item_ids=item_ids,
        matrix=matrix,
        seen_items=seen_item,
        seen_starts=seen_starts,
        pop_all=pop_all,
        pop_recent=pop_recent,
        n_users=n_users,
        n_items=n_items,
        extra={"scheme": scheme, "ref_ts": ref_ts},
    )
