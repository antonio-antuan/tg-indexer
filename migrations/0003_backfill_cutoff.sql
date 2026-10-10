-- SPEC v1.11 §4: the backfill cutoff in force when the chat's backfill last finished; null
-- when it never finished or finished before this column existed (the next sync reopens it).
ALTER TABLE chat ADD COLUMN backfill_cutoff timestamptz;
