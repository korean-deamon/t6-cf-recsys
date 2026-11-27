# T6 - Collaborative filtering recommender: report

**Database:** `ml_t6_recsys_cf` (`10.20.9.24:5434`), user `t6_student`
**Round:** 3 · `as_of` = 2026-11-02 · test window 2026-11-03 .. 2026-11-16
**Metric:** NDCG@10 (bigger is better), also: Recall@10, catalogue coverage

---

## 1. Data audit

| Number | Value |
|---|---|
| Events | 1,331,903 |
| Users | 7,031 |
| Content (seen in train) | 4,882 |
| Catalogue (`raw.content`) | 5,003 |
| Period | 2026-03-01 .. 2026-11-03 |
| Test users (`raw.test_users`) | 4,577 |
| Matrix density | 0.0388 (3.9%) |

**Event types**

| Type | Count | Share | Users |
|---|---|---|---|
| view | 1,193,484 | 89.61% | 7,031 |
| like | 117,084 | 8.79% | 6,451 |
| share | 21,335 | 1.60% | 4,247 |

**Watch pct distribution (view)**

| Range | Share |
|---|---|
| 0.0-0.1 | 14.65% |
| 0.1-0.3 | 18.82% |
| 0.3-0.6 | 37.40% |
| 0.6-0.9 | 27.54% |
| 0.9-1.0 | 1.59% |

The mean watch share is about 0.44, and about a third of the views are below
the `min_watch_pct = 0.3` line. Those are not counted as a "good watch", but
they do go into the "seen" list, so the same item is not recommended again.

**Anomalies**

- 21,335 duplicates on `(user_id, content_id, event_ts)` - a share event is
  written in the same second as the view. At model level the `(user, item)`
  pair is summed up, so this has no effect.
- 121 items were never watched (2.4% of the catalogue); the model never
  considers them.
- `sandbox.t6_bots` holds **30 suspicious accounts** (the biggest one has
  19,903 events and 4,091 items). They break the popularity signal, so they are
  left out of training with `data.exclude_bots: true`.

---

## 2. Signals and model

| Signal | Weight | Note |
|---|---|---|
| view | `1.0 * watch_pct` | `watch_seconds / duration_s`, 0 below 0.3 |
| like | `0.6` | sign of intent |
| share | `1.2` | the strongest sign of intent |
| repeat view | `* (1 + 0.5*log1p(n-1))` | |
| time decay | `0.5^(age/180 days)` | old interest matters less |

The second scheme (the task asks for two) is `weights.scheme: "B"`: a binary
threshold instead of `watch_pct` (`watch_pct >= 0.3 -> 1`). The comparison is
in `src/tune.py`.

Models:

1. **Popularity** - unique users in the last 14 days (baseline).
2. **Item-kNN** - cosine similarity with shrink, 200 neighbours.
3. **ALS** (implicit, `confidence = 1 + alpha*w`) plus a recent-popularity
   prior: `score = minmax(ALS) + 0.6 * log1p(pop_recent)`.

Inside the top-10: seen items are removed, items missing from train are
removed, and if 10 is not enough the list is filled by popularity (cold start).

---

## 3. Offline validation

Train: 2026-03-01 .. 2026-10-19 · validation: 2026-10-19 .. 2026-11-02
(3,000 users, about 4.8 relevant items each, bots removed).

| Model | NDCG@10 | Recall@10 | Coverage |
|---|---|---|---|
| popularity | 0.0232 | 0.0320 | 0.005 |
| itemknn | 0.0301 | 0.0405 | 0.357 |
| **als** | **0.0652** | **0.0853** | 0.494 |

### Hyperparameter choice (`reports/tuning.csv`)

| Variant | NDCG@10 | Coverage |
|---|---|---|
| ALS 32 factors, alpha=20 (no prior) | 0.0652 | 0.494 |
| + pop_prior 0.6 | 0.0932 | 0.209 |
| + pop_prior 0.9 | 0.0963 | 0.175 |
| **+ pop_prior 0.9, alpha=30** | **0.0971** | 0.177 |
| 64 factors, alpha=20 | 0.0632 | 0.626 |
| scheme B | 0.0509 | 0.728 |
| half-life 90 days | 0.0596 | 0.634 |

What it means: quality comes from the **popularity prior**, not from a bigger
model. More factors overfit a sparse matrix, while the prior adds the signal
"what will many people watch in the next 14 days". Coverage drops from 0.494
to 0.177, but stays above the 0.10 the task asks for.

---

## 4. Leaderboard results

| Run | Model | NDCG@10 | Baseline | Change | Catalogue cov. | Status |
|---|---|---|---|---|---|---|
| 43 | `als` (no prior, alpha=20) | 0.07172 | 0.05751 | +24.7% | 0.369 | below_pass |
| 44 | `als` + pop_prior 0.9, alpha=30 | **0.18756** | 0.05751 | **+226.1%** | 0.0855 | excellent |
| **45** | `als` + pop_prior 0.6, alpha=30 | **0.18210** | 0.05751 | **+216.6%** | **0.1043** | **excellent** |
| 46 | the same model, re-run end-to-end | 0.18210 | 0.05751 | +216.6% | 0.1043 | excellent |

Round 3 thresholds (relative): pass = baseline x 1.4103 = **0.0811**,
excellent = baseline x 1.7949 = **0.1032**. Runs 44 and 45 are both about two
times above the line. The rubric also needs **coverage >= 0.10** for excellent;
run 44 gives 0.0855 (not enough), so the submitted variant is **run 45**:
NDCG 0.1821 with coverage 0.1043.

Every run: coverage 1.000 (4,577 of 4,577 users, 45,770 rows), hit rate @10 =
0.573, and 3,407 users have at least one label.

Leaderboard (`eval.leaderboard`): 1st place, `5-guruh t6`, run 45,
`improvement_pct = 216.65`, best attempt 226.13% (run 44).

Daily quota: 4 of 10 used (6 left) - run 46 was a full re-check on the live
database.

### Why did the prior help so much?

The test label is "which NEW item will this user watch in the next 14 days".
For this target the recent popularity of an item is a very strong signal: ALS
gives "what people like you watched", the prior adds "what the whole audience
is watching right now". They complete each other. If the prior is too big
(1.5+), the lists become the same for everyone and coverage falls to about
0.14.

---

## 5. Segment analysis (`reports/segments.csv`)

Run 45 model, validation window, 3,000 users:

| Segment | N | NDCG@10 |
|---|---|---|
| all | 3,000 | 0.0956 |
| activity: low (<=33%) | 1,001 | **0.1105** |
| activity: middle | 999 | 0.0888 |
| activity: high (top 33%) | 1,000 | 0.0874 |
| cold start (not in train) | 1,792 | 0.0928 |
| has history | 1,208 | 0.0997 |

Popularity of the recommended items: **head 99.9%**, mid 0.09%, tail 0.03%.

**What it shows:**

- Users with little activity get **better** quality (0.1105): their taste is
  "mainstream" and the prior works well there.
- The most active 33% get a lower score (0.0874): most of what they watched is
  already in the "seen" list, so the choice is narrow.
- Cold start users still do well (0.0928), which shows the popularity fallback
  works.
- **Main limit:** the recommendations are almost only head content (top 33% by
  popularity). That is why catalogue coverage is 0.1043 - close to the line.
  With prior 0.9 the score is 0.1876 but coverage falls to 0.0855 (below the
  excellent condition). Next step: prior plus MMR / diversity re-rank to show
  more tail content.

## 6. Inference time and the final run

| Step | Time |
|---|---|
| Read the events from the database (758,862 rows, join) | ~65 s |
| Build the matrix (6,983 x 3,726, 346,093 pairs) | ~10 s |
| Train ALS (32 factors, 12 iterations) | 8.0 s |
| Top-10 for 4,577 users and write | ~19 s |
| **Total (run 44)** | **102.8 s** |
| **Total (run 46, end-to-end check)** | **204.3 s** |
| Inference (measured in the segment analysis) | 6,040 users/s |

The task limit is 15 minutes and 30 connections. The flow uses one connection
and takes about 2 minutes for 5,000 users, so it stays far below the limit.

**Submitted run: `run_id = 45`**, `model_name = als`, `model_version =
als_pop06_v1`, round 3, NDCG@10 = 0.182101, coverage = 0.1043, status =
excellent. Run 46 trained the same model again on the real database and gave
exactly the same numbers, so the pipeline is reproducible.

Hyperparameter log: `reports/tuning.csv` (8 + 7 variants: factors, iterations,
alpha, regularisation, scheme A/B, half-life, pop_prior).

## 7. How to repeat it

```bash
cp .env.example .env          # password from the mentor
python -m src.audit
python -m src.evaluate --models popularity itemknn als --sample 3000
python -m src.tune --sample 3000
python -m src.segments --sample 3000
python -m src.recommend --run-name als_pop06_v1 --model als --score
python -m unittest discover -s tests -t .
```

More detail: `TESTING.md`.

## 8. Limitations

- Offline validation uses one 14-day window; if the distribution drifts between
  rounds, the score will differ.
- `watch_seconds` is `NULL` in some views and was taken as 0 (14% low share).
- Database quota: 10 scorings per day, and each run is scored once.
- Bots were removed, so the popularity signal is based on a clean audience.
