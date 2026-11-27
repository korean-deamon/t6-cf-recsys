-- Queries that check a written run. Replace <run_id> with your number.
-- Expected: 4 577 users x 10 = 45 770 rows.

-- 1) Row count and users
SELECT count(*) AS rows, count(DISTINCT user_id) AS users
  FROM ml.recommendations WHERE run_id = <run_id>;

-- 2) Exactly 10 items per user, ranks 1..10 (must return 0)
SELECT count(*) AS not_ok_users FROM (
    SELECT user_id
      FROM ml.recommendations WHERE run_id = <run_id>
     GROUP BY 1
    HAVING count(*) <> 10 OR count(DISTINCT rank) <> 10
        OR min(rank) <> 1 OR max(rank) <> 10) t;

-- 3) Duplicate content (must be 0)
SELECT count(*) AS duplicates FROM (
    SELECT user_id, content_id
      FROM ml.recommendations WHERE run_id = <run_id>
     GROUP BY 1, 2 HAVING count(*) > 1) t;

-- 4) Items the user already watched in train (must be 0).
--    The time filter is required: `raw.events` also holds events from the test
--    window (after as_of), and without the filter the query reports a wrong
--    "441 already seen".
SELECT count(*) AS seen_rows
  FROM ml.recommendations r
  JOIN raw.events e
    ON e.user_id = r.user_id AND e.content_id = r.content_id
   AND e.event_type = 'view'
 WHERE r.run_id = <run_id>
   AND e.event_ts < (SELECT as_of FROM raw.data_clock);

-- 5) Catalogue coverage (the task needs >= 0.10 for excellent)
SELECT count(DISTINCT r.content_id)::numeric
       / (SELECT count(*) FROM raw.content) AS catalog_coverage
  FROM ml.recommendations r WHERE r.run_id = <run_id>;

-- 6) Does every recommended item really exist
SELECT count(*) AS invalid_content
  FROM ml.recommendations r
  LEFT JOIN raw.content c ON c.content_id = r.content_id
 WHERE r.run_id = <run_id> AND c.content_id IS NULL;

-- 7) Score the run and look at the leaderboard
SELECT * FROM eval.score_run(<run_id>);
SELECT * FROM eval.leaderboard;
