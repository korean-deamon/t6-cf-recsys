# How to test this project

Everything below runs from the project root. The test suite needs no database,
so it works at home as well.

## 1. Setup (once)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The file `.env` already holds the database login. If it is missing:

```bash
cp .env.example .env             # then fill in the values from the mentor
```

## 2. Unit tests (no database, about 1 second)

```bash
.venv/bin/python -m unittest discover -s tests -t .
```

Expected result:

```
Ran 36 tests in 0.4s

OK
```

What is covered:

| File | What it checks |
|---|---|
| `tests/test_metrics.py` | NDCG, recall, precision, MAP, coverage |
| `tests/test_data.py` | signal weights, scheme A/B, repeat bonus, time decay, "seen" list |
| `tests/test_als.py` | ALS shapes, cluster recovery, `select_top_k` rules |
| `tests/test_knn.py` | item-kNN similarity, end-to-end pipeline |
| `tests/test_offline.py` | the relevance rule (first view + `watch_pct >= 0.3`) |
| `tests/test_local.py` | the local file store and CSV output |
| `tests/test_pipeline.py` | ranks 1..10, no seen items, cold start, dry run |

One file only:

```bash
.venv/bin/python -m unittest tests.test_pipeline -v
```

## 3. Data audit (local export)

```bash
.venv/bin/python -m src.audit --source local
```

Expected, with the shipped `data/raw.npz`:

- events: `1,325,744`
- test users: `4,577`
- bots left out of training: `30`
- clock (`as_of`): `2026-11-02`

The same audit against the live database (university network only):

```bash
.venv/bin/python -m src.audit
```

## 4. Offline validation (compare the models)

```bash
.venv/bin/python -m src.evaluate --models popularity itemknn als --sample 3000 --source local
```

Expected: `als` is clearly the best model, `popularity` is the weakest one.
The numbers are written to `reports/offline_models.csv`, so you can compare the
file with the console output.

## 5. Full flow without the database

```bash
.venv/bin/python -m src.recommend --run-name demo --model als --source local
```

This writes `data/recommendations_run1.csv` and prints the row count.
It does not touch `ml.recommendations`.

## 6. Real run and leaderboard score

Needs the university network. **Each `--score` call uses one of the 10 daily
scoring attempts** (3 used so far, 7 left).

```bash
.venv/bin/python -m src.recommend --run-name als_pop06_v1 --model als --score
```

Expected score table (this is run 45):

```
ndcg_at_10      0.18210
baseline        0.05751
improvement     2.166x  (+216.6%)
status          excellent
```

A cheaper dry run first, to be sure the data loads:

```bash
.venv/bin/python -m src.recommend --run-name check --model als --limit-users 200 --dry-run
```

## 7. Check what was written to the database

Run `sql/01_checks.sql` with `<run_id>` replaced by your run number:

```bash
psql -h 10.20.9.24 -p 5434 -U t6_student -d ml_t6_recsys_cf -f sql/01_checks.sql
```

Expected for run 45:

| Check | Expected |
|---|---|
| rows / users | `45,770` / `4,577` |
| users without exactly ranks 1..10 | `0` |
| duplicate content | `0` |
| already seen in train | `0` |
| catalogue coverage | `0.1043` (522 of 5,004 items) |

No `psql` on the machine? Any PostgreSQL client works, or a short Python call:

```bash
.venv/bin/python -c "
import psycopg2
from src.config import db_params, load_dotenv
load_dotenv()
c = psycopg2.connect(**db_params()); q = c.cursor()
q.execute('select run_id, primary_metric, status from eval.results order by run_id desc limit 3')
print(q.fetchall()); c.close()"
```

## 8. Tuning and segments (optional, slower)

```bash
.venv/bin/python -m src.tune --source local --sample 3000      # -> reports/tuning.csv
.venv/bin/python -m src.segments --source local                # -> reports/segments.csv
```

`tune` trains 7 parameter sets, so it takes a few minutes. `segments` reports
NDCG per user activity group, per content popularity group and for cold start.

## 9. Common problems

| Message | Reason | Fix |
|---|---|---|
| `timeout expired` | database is under load | wait 1-2 minutes, try again |
| `No route to host` | wrong network | connect to the university Wi-Fi |
| `password authentication failed` | wrong password | copy it from `.env`, mind `O` and `k` |
| `psql: command not found` | no client installed | use the Python snippet above |
| `scipy is not installed` | skip note from `test_knn` | `pip install -r requirements.txt` |
| `data/raw.npz not found` | export missing | on campus: `python -m src.export` |

## 10. Expected summary

| Item | Value |
|---|---|
| unit tests | 36 passed |
| leaderboard run | 45 |
| NDCG@10 | 0.18210 |
| baseline (popularity) | 0.05751 |
| improvement | +216.6% |
| catalogue coverage | 0.1043 |
| status | excellent |
