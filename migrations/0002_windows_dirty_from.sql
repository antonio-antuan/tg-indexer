-- SPEC v1.6 §4: persisted window-rebuild marker (decision #632). The lowest message_id
-- written, edited, tombstoned, restored or purged since the chat's windows were rebuilt;
-- null when they are current.
ALTER TABLE chat ADD COLUMN windows_dirty_from bigint;
