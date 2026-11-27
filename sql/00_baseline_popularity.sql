-- Baseline from the task: the most popular content of the last 14 days (SQL only).
-- The main flow is `src/recommend.py`; this file is a fallback and a cross-check.
--
-- Logic: top-200 candidates by how many unique users watched them in the last
-- 14 days; every test user gets the 10 items they have not watched yet.

WITH rnd AS (
    SELECT round_id, as_of FROM raw.current_round ORDER BY round_id DESC LIMIT 1
), pop AS (
    SELECT e.content_id, count(DISTINCT e.user_id) AS users
      FROM raw.events e, rnd r
     WHERE e.event_type = 'view'
       AND e.event_ts >= r.as_of - interval '14 days'
       AND e.event_ts <  r.as_of
       AND NOT EXISTS (SELECT 1 FROM sandbox.t6_bots b WHERE b.user_id = e.user_id)
     GROUP BY 1
     ORDER BY 2 DESC
     LIMIT 200
), run AS (
    INSERT INTO ml.model_runs
        (student_name, model_name, model_version, round_id, train_from, train_to)
    SELECT '5-guruh t6', 'baseline_sql', 'v1', r.round_id,
           r.as_of - interval '14 days', r.as_of
      FROM rnd r
    RETURNING run_id
), cand AS (
    SELECT t.user_id, p.content_id,
           row_number() OVER (PARTITION BY t.user_id
                              ORDER BY p.users DESC, p.content_id) AS rn
      FROM raw.test_users t
     CROSS JOIN pop p
     WHERE NOT EXISTS (
           SELECT 1 FROM raw.events i
            WHERE i.user_id = t.user_id
              AND i.content_id = p.content_id
              AND i.event_type = 'view')
)
INSERT INTO ml.recommendations(run_id, user_id, rank, content_id)
SELECT run.run_id, c.user_id, c.rn, c.content_id
  FROM run, cand c
 WHERE c.rn <= 10;

-- Scoring (this uses one of the daily quota):
--   SELECT * FROM eval.score_run((SELECT max(run_id) FROM ml.model_runs));
