# Presentation plan (10 minutes)

Goal: explain the problem, what we built, and why it works. Numbers are from
the real database (round 3, run 45, re-checked as run 46).

## Slide 1 - The problem (1 min)

"For every user, which **new** item will they watch in the next 14 days?"

- Data: 1.33M events, 7,031 users, 5,003 items, behaviour only
  (view / like / share). No genre, no description, no actors.
- Task: a top-10 list for each of the 4,577 test users.
- Metric: **NDCG@10** - correct items should sit high in the list.
  Second metric: catalogue coverage - we must not show the same 20 items to
  everybody.
- Limits: 15 minutes of compute, one database connection.

## Slide 2 - Data audit (1.5 min)

What we checked before modelling:

- 89.6% views, 8.8% likes, 1.6% shares. Density 3.9% - the matrix is very empty.
- `watch_pct` (watched seconds / duration): a third of the views are below 0.3,
  so we need a "good watch" threshold, otherwise a 5-second look counts as
  interest.
- 30 bot accounts in `sandbox.t6_bots` (one has 19,903 events) - they would
  break the popularity signal, so we removed them from training.
- 121 items were never watched.
- 21,335 duplicate rows on (user, item, second) - a share is written together
  with the view. We sum at (user, item) level, so this is harmless.

## Slide 3 - Our representation (1.5 min)

We turn behaviour into one weighted matrix:

```
w = 1.0 * watch_pct + 0.6 * like + 1.2 * share
repeat bonus: w * (1 + 0.5 * log1p(views - 1))
time decay:   w * 0.5 ** (age_days / 180)
```

- share is the strongest sign of intent, so it has the biggest weight.
- The time decay makes an old interest weaker than a fresh one.
- Second scheme (the task asks for two): binary, `watch_pct >= 0.3 -> 1`.
- Result: 6,983 x 3,726 matrix with 346,093 pairs.

## Slide 4 - Three models (2 min)

| Model | What it does | NDCG@10 (offline) | Coverage |
|---|---|---|---|
| popularity | most watched items of the last 14 days | 0.0232 | 0.005 |
| item-kNN | "who watched this also watched that" | 0.0301 | 0.357 |
| ALS | matrix factorisation, implicit feedback | 0.0652 | 0.494 |
| **ALS + popularity prior** | ALS plus "what is hot right now" | **0.0971** | 0.177 |

- ALS: `confidence = 1 + alpha * w`, pure numpy, 32 factors, 12 iterations,
  trains in 8 seconds.
- Our own score computation, no library for the model itself.

## Slide 5 - The main solution: popularity prior (1 min)

- The target is "next 14 days", so recent audience demand is a strong signal.
- `score = minmax(ALS) + pop_prior * log1p(unique viewers last 14 days)`.
- With prior 0.9 we get NDCG 0.1876 but coverage only 0.0855.
- With prior **0.6** we get **0.1821 and coverage 0.1043** - both excellent
  conditions are met. This trade-off is the key decision of the project.

## Slide 6 - Filters that keep the output correct (1 min)

Before writing, for every user:

1. remove everything they already watched (even a 5-second view),
2. remove items that never appear in train (no data to learn from),
3. if fewer than 10 items are left, fill the rest with popular items
   (this is what saves cold start users).

Then we check the result in SQL: exactly 10 rows, ranks 1..10, no duplicates,
no seen items.

## Slide 7 - Leaderboard result (1 min)

| Run | Model | NDCG@10 | Change vs baseline | Coverage | Status |
|---|---|---|---|---|---|
| 43 | ALS, no prior | 0.07172 | +24.7% | 0.369 | below_pass |
| 44 | ALS + prior 0.9 | 0.18756 | +226.1% | 0.0855 | excellent |
| **45** | **ALS + prior 0.6** | **0.18210** | **+216.6%** | **0.1043** | **excellent** |
| 46 | same model, re-run | 0.18210 | +216.6% | 0.1043 | excellent |

Baseline 0.05751, pass line 0.0811, excellent line 0.1032. We are at 1st place,
45,770 rows (4,577 users x 10), and run 46 shows the result is reproducible.

## Slide 8 - Weak points and next steps (1 min)

Segment analysis (`reports/segments.csv`):

| Segment | NDCG@10 |
|---|---|
| low activity | 0.1105 |
| middle | 0.0888 |
| high activity | 0.0874 |
| cold start | 0.0928 |

- 99.9% of the recommended items are head content, so coverage is only 0.1043.
- Weakest group: the most active users - most of what they watched is filtered
  out, so the choice is narrow.
- Next steps: MMR / diversity re-rank, tail boost, two-stage ranking with
  LightGBM.

## Slide 9 - Demo and code (0.5 min)

```
python -m src.recommend --run-name als_pop06_v1 --model als --score
python -m unittest discover -s tests -t .
```

- `src/`: data, als, knn, models, pipeline, train, recommend, evaluate, audit,
  tune, segments, store.
- 36 unit tests, they run without a database.
- Password only in `.env`, never in the code; only the database is used, no
  local copy and no outside metadata.

## Likely questions

**Why is item-kNN weak?** On a sparse matrix cosine similarity is noisy. Shrink
helped, but ALS latent factors are much more stable.

**Why is coverage low?** The prior pushes popular items up. We chose 0.6 as the
balance; diversity re-ranking is the next step.

**How did you find the bots?** The `sandbox.t6_bots` table: 30 users with 19k+
events, active 247 days. They would inflate popularity, so we removed them.

**Is there data leakage?** No. Train stops at `as_of` (2026-11-02), validation
is the 14 days before it, and the test window (11-03 .. 11-16) is never used.

**Why no rating or bookmark signal?** This database has no such tables, so we
used like and share instead. That also follows the rule "CF only, no outside
metadata".

**What if the database is slow?** Every run uses one connection and about two
minutes; with `--limit-users 200 --dry-run` we can check the pipeline in
seconds without writing anything.
