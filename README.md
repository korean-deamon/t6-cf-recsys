# T6 - Collaborative filtering recommender

A "For you" recommender for a video streaming platform: for every user, the
**10 new items** they will watch in the next 14 days. Only behaviour data is
used (no genre, description or actors), so this is a pure collaborative
filtering problem.

- Database: `ml_t6_recsys_cf` (user `t6_student`, login kept in `.env`)
- Main metric: **NDCG@10** - pass `>= 0.0811`, excellent `>= 0.1032`
  (the baseline 0.05751 times 1.4103 / 1.7949)
- Limit: 15 minutes, 15 000 users x 10 = 150 000 rows

**Result: NDCG@10 = 0.18210 (baseline 0.05751, +216.6%), catalogue coverage
0.1043, status `excellent`.** Details: `reports/REPORT.md`, how to test:
`TESTING.md`.

## Quick start

```bash
# 1) libraries
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) connection values (from the mentor)
cp .env.example .env      # then put PGPASSWORD in it

# 3) check the data (needs the university network)
python -m src.audit

# 4) compare the models offline (no leaderboard)
python -m src.evaluate --models popularity itemknn als --sample 3000

# 4b) choose the ALS hyperparameters -> reports/tuning.csv
python -m src.tune --sample 3000

# 4c) segment analysis -> reports/segments.csv
python -m src.segments --sample 3000

# 5) compute the recommendations, write them and score
python -m src.recommend --run-name pop_v1 --model popularity --score
python -m src.recommend --run-name als_v1 --model als --score

# 6) tests (they run without a database)
python -m unittest discover -s tests -t .
```

## About the database

The database `10.20.9.24:5434` is on the internal network, so it is reachable
only from the university or a VPN. Every command works directly against this
database: there is no local copy and no local mode. `eval.score_run()` also
works on this database only.

Useful flags for a quick check: `--limit-users 200 --dry-run` (trains the model
and prints the row count, writes nothing).

## Code layout

| File | What it does |
|---|---|
| `src/config.py` | reads `config.yaml` and `.env` (no password in the code) |
| `src/store.py` | the only database layer: read (server-side cursor) and write |
| `src/data.py` | merges the 3 signals into one user x item matrix |
| `src/models.py` | model registry (one place to add a model) |
| `src/als.py` | ALS - implicit matrix factorisation (numpy, no scipy) |
| `src/knn.py` | item-kNN - cosine similarity (scipy) |
| `src/pipeline.py` | main flow plus `select_top_k` (removes seen items) |
| `src/recommend.py` | CLI: compute -> `ml.recommendations` |
| `src/evaluate.py` | offline validation: last 14 days, model x metric table |
| `src/audit.py` | data audit queries |
| `tests/` | metrics, `select_top_k`, ALS, item-kNN, the whole flow |

## Model strategy (and why in this order)

1. **popularity (baseline)** - the content that most different people watched
   in the last 14 days. NDCG about 0.04. Goal: prove the pipeline works.
2. **item-kNN** - "people who watched this also watched that". Simple, easy to
   explain, usually much better than the baseline.
3. **ALS** - splits the matrix into `U @ V.T` and solves the user and item
   factors one after the other. For implicit feedback it uses
   `confidence = 1 + alpha * w`.
4. **tuning** - `factors`, `regularization`, `alpha`, `neighbors`.
5. Bonus (if time is left): two-stage pipeline (candidates -> LightGBM ranker).

## Implicit signal weights

`config.yaml -> weights.scheme` switches between two schemes:

- **A** - continuous: the `watch_pct` value (0 below 0.3), a repeat bonus
  `1 + 0.5*log1p(n-1)`, rating `0.2*rating`, bookmark `+1`.
- **B** - binary: 1 point if `watch_pct >= 0.3`, everything else like A.

Both use a time decay: `0.5 ** (age / 180 days)`.

## Checking a run

After writing, only `eval.score_run(run_id)` shows the real truth:

```sql
SELECT * FROM eval.score_run(<run_id>);
SELECT * FROM eval.leaderboard;
```

Expected checks: `already_seen_share = 0` (items seen in train are never
recommended), `users_covered = 1.0`, and exactly 10 different items per user.

## Results

| Run | Model | NDCG@10 | Baseline | Change | Coverage | Status |
|---|---|---|---|---|---|---|
| 43 | `als` (no prior, alpha=20) | 0.07172 | 0.05751 | +24.7% | 0.369 | below_pass |
| 44 | `als` + pop_prior 0.9, alpha=30 | **0.18756** | 0.05751 | **+226.1%** | 0.0855 | excellent |
| **45** | `als` + pop_prior 0.6, alpha=30 | **0.18210** | 0.05751 | **+216.6%** | **0.1043** | **excellent** |
| 46 | same model, re-run on the real DB | 0.18210 | 0.05751 | +216.6% | 0.1043 | excellent |

Run 44 scores a little higher, but its coverage 0.0855 is below the 0.10 the
rubric asks for, so run 45 is the submitted one. Run 46 was a fresh end-to-end
check on the live database and reproduced the same numbers.

## Rules (breaking them means disqualification)

- The password is never written in the code; `.env` follows the same rule.
- Filling content metadata from an outside source is not allowed - CF only.
- The `raw` schema is read-only: do not run `INSERT/UPDATE` on it.
- Sharing code between teams is not allowed.

## Short version for the teacher

1. **Problem**: 7k users, 5k items, behaviour only. Every user needs 10 *new*
   items -> a ranking problem with implicit feedback.
2. **Why popularity is not enough**: the same list for everyone, no
   personalisation, low coverage, NDCG 0.04.
3. **Data**: views (`watch_pct`), likes, shares -> a weighted matrix; the time
   decay makes old interest weaker.
4. **Model**: ALS factorises the matrix, item-kNN works through neighbours,
   both learn from "who watched what".
5. **Validation**: take the last 14 days of train and repeat the test rule
   exactly (a cut by date, so nothing leaks).
6. **Filter**: items seen in train are removed, `watch_pct < 0.3` is ignored,
   items missing from train are never recommended.
7. **Metric**: NDCG@10 - how high the correct items sit in the list.
