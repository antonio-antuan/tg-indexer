"""QA probes for T6 round 2 (59b0666): empty-page guard, wholly deleted window, marker order."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tgi import chats, ingest
from tgi.config import Settings
from tgi.td_fake import FakeTdClient

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACC = "+34600000001"
BASE = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=0)
A = -1001000000501
MODES = ["tombstone", "purge"]


class Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        pass


def msgs(n: int, step: timedelta) -> list[dict[str, Any]]:
    return [{"id": i, "date": (NOW - (n - i) * step).isoformat(), "text": f"SECRET {i}"}
            for i in range(1, n + 1)]  # fmt: skip


def make(
    n: int, step: timedelta, runs: list[dict[str, Any]], sizes: list[int] | None = None
) -> FakeTdClient:
    chat = {"chat_id": A, "title": "t", "kind": "supergroup", "messages": msgs(n, step),
            "page_sizes": sizes or []}  # fmt: skip
    return FakeTdClient({"chats": [chat], "runs": runs})


def sync(conn: Any, td: FakeTdClient, mode: str, **kw: Any) -> Any:
    return ingest.sync(conn, lambda _: td, replace(BASE, on_delete=mode), Clock(), **kw)


def gone(conn: Any, mode: str) -> list[int]:
    """Ids (server) the archive treats as deleted."""
    if mode == "purge":
        have = {r[0] >> 20 for r in conn.execute("SELECT message_id FROM message")}
        return sorted(set(range(1, 251)) - have)
    return sorted(r[0] >> 20 for r in conn.execute(
        "SELECT message_id FROM message WHERE deleted_at IS NOT NULL"))  # fmt: skip


def total(conn: Any) -> int:
    return conn.execute("SELECT count(*) FROM message").fetchone()[0]  # type: ignore[no-any-return]


DEL = [{"chat_id": A, "id": i} for i in (200, 201, 240)]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("where", "sizes", "empty_calls"),
    [("first", None, [1]), ("later", [50] * 10, [2]), ("later-deep", [10] * 40, [5]),
     ("full-sync-first", None, [1, 2, 3])],
)  # fmt: skip
def test_spurious_empty_page_deletes_nothing_then_next_run_deletes(
    conn: Any,
    caplog: pytest.LogCaptureFixture,
    mode: str,
    where: str,
    sizes: Any,
    empty_calls: list[int],
) -> None:
    caplog.set_level("DEBUG")
    faults = [{"call": k, "empty": True} for k in empty_calls]
    td = make(250, timedelta(hours=1), [{"delete": DEL, "faults": faults}, {}], sizes)
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    td.next_run()
    if where.startswith("full-sync"):
        res = sync(conn, td, mode)
    else:  # resweep alone: call 1 is the sweep's first page
        res = sync(conn, td, mode, resweep=(A, 7))
    assert res.status == "ok"
    assert gone(conn, mode) == [], where
    assert total(conn) == 250
    assert "SECRET" not in caplog.text and ACC not in caplog.text
    assert any(str(A) in r.getMessage() and "incomplete" in r.getMessage() for r in caplog.records)
    td.next_run()
    assert sync(conn, td, mode).status == "ok"
    assert gone(conn, mode) == [200, 201, 240]


@pytest.mark.parametrize("mode", MODES)
def test_real_empty_page_at_young_chat_start_is_trusted(conn: Any, mode: str) -> None:
    """Whole history inside the window; a mid-chat deletion; the real empty page ends the sweep."""
    td = make(250, timedelta(minutes=1), [{"delete": [{"chat_id": A, "id": 120}]}], [30] * 9)
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    td.next_run()
    sync(conn, td, mode)
    assert gone(conn, mode) == [120]


@pytest.mark.parametrize("mode", MODES)
def test_young_chat_oldest_deleted_is_left_for_later(conn: Any, mode: str) -> None:
    """Documented limit (SPEC §5.3 step 3): oldest messages of an all-in-window chat wait."""
    td = make(250, timedelta(minutes=1), [{"delete": [{"chat_id": A, "id": 1}]}])
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    td.next_run()
    sync(conn, td, mode)
    assert gone(conn, mode) == []


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sizes", [None, [37, 1, 64, 100, 9] * 20, [1] * 1000])
def test_fully_deleted_window_above_older_survivors_in_one_sweep(
    conn: Any, mode: str, sizes: Any
) -> None:
    window = [{"chat_id": A, "id": i} for i in range(83, 251)]  # every message in the 7 days
    td = make(250, timedelta(hours=1), [{"delete": window}], sizes)
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    assert total(conn) == 250
    td.next_run()
    res = sync(conn, td, mode)
    assert res.status == "ok"
    assert gone(conn, mode) == list(range(83, 251))
    assert total(conn) == (82 if mode == "purge" else 250)


TRIG = """
CREATE TABLE wlog (op text, message_id bigint, dirty bigint);
CREATE FUNCTION wl() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE d bigint; mid bigint;
BEGIN
  IF TG_OP = 'DELETE' THEN mid := OLD.message_id; ELSE mid := NEW.message_id; END IF;
  SELECT windows_dirty_from INTO d FROM chat WHERE chat_id = OLD.chat_id;
  INSERT INTO wlog VALUES (TG_OP, mid, d);
  IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER wl_upd BEFORE UPDATE OF deleted_at ON message FOR EACH ROW EXECUTE FUNCTION wl();
CREATE TRIGGER wl_del BEFORE DELETE ON message FOR EACH ROW EXECUTE FUNCTION wl();
"""


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sizes", [None, [7] * 40])
def test_marker_set_before_tombstone_purge_and_restore(conn: Any, mode: str, sizes: Any) -> None:
    back = [{"chat_id": A, "id": 240, "date": (NOW - timedelta(hours=10)).isoformat(), "text": "x"}]
    td = make(250, timedelta(hours=1), [{"delete": DEL}, {"add": back}, {"delete": DEL}], sizes)
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    conn.execute(TRIG)
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")
    td.next_run()
    sync(conn, td, mode)
    rows = conn.execute("SELECT op, message_id >> 20, dirty >> 20 FROM wlog").fetchall()
    assert {r[1] for r in rows} == {200, 201, 240}
    assert all(r[2] is not None and r[2] <= 200 for r in rows), rows
    assert conn.execute("SELECT windows_dirty_from >> 20 FROM chat").fetchone() == (200,)
    if mode == "tombstone":  # restore: reappearing message marks itself before the update
        conn.execute("UPDATE chat SET windows_dirty_from = NULL")
        conn.execute("DELETE FROM wlog")
        td.next_run()
        sync(conn, td, mode)
        rows = conn.execute("SELECT op, message_id >> 20, dirty >> 20 FROM wlog").fetchall()
        assert rows == [("UPDATE", 240, 240)], rows
        # a resweep that finds nothing new leaves the marker alone
        conn.execute("UPDATE chat SET windows_dirty_from = NULL")
        sync(conn, td, mode, resweep=(A, 7))
        assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
@pytest.mark.parametrize("mode", MODES)
def test_marker_survives_a_kill_after_the_sweep_deleted(conn: Any, mode: str) -> None:
    """Marker and deletion commit together; later failure keeps both (marker never lost)."""
    td = make(250, timedelta(hours=1), [{"delete": DEL}])
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")
    td.next_run()
    sync(conn, td, mode)
    assert conn.execute("SELECT windows_dirty_from >> 20 FROM chat").fetchone() == (200,)
    assert gone(conn, mode) == [200, 201, 240]


@pytest.mark.parametrize("mode", MODES)
def test_resweep_edit_older_than_backfill_years_is_applied(conn: Any, mode: str) -> None:
    old = NOW - timedelta(days=4 * 365)
    ms = msgs(250, timedelta(hours=1))
    ms[4]["date"] = old.isoformat()  # id 5 is 4 years old: beyond BACKFILL_YEARS=3
    ms[5]["date"] = old.isoformat()
    td = FakeTdClient({"chats": [{"chat_id": A, "title": "t", "kind": "supergroup",
                                  "messages": ms}], "runs": [
        {"edit": [{"chat_id": A, "id": 5, "text": "EDITED", "edit_date": NOW.isoformat()}],
         "add": [{"chat_id": A, "id": 251, "date": old.isoformat(), "text": "n"}]}]})  # fmt: skip
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td, mode)
    # force the old rows into the archive as if stored under a longer lookback
    conn.execute(
        "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
        " VALUES (%s, %s, 5, %s, 'old')",
        (A, 5 << 20, old),
    )
    td.next_run()
    res = sync(conn, td, mode, resweep=(A, 5 * 365))
    assert res.status == "ok"
    text = conn.execute("SELECT text FROM message WHERE message_id = %s", (5 << 20,)).fetchone()
    assert text == ("EDITED",)
    # a message new to the archive and older than the lookback is still not stored
    assert conn.execute(
        "SELECT count(*) FROM message WHERE message_id = %s", (251 << 20,)
    ).fetchone() == (0,)
    # and without the stored copy the old one is not resurrected either
    assert conn.execute(
        "SELECT count(*) FROM message WHERE message_id = %s", (6 << 20,)
    ).fetchone() == (0,)
