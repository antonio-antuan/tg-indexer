-- SPEC v1.12 §4: `recent` lists windows by their last message, newest first, and pages on
-- (end_date, window_id); this index serves the range filter, the order and the cursor.
-- A plain CREATE INDEX (the runner wraps each migration in a transaction, which CONCURRENTLY
-- cannot run in): it blocks window writes while it builds, so migrate while no sync runs.
CREATE INDEX window_end_idx ON "window" (end_date DESC, window_id DESC);
