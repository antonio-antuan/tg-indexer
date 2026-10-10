"""T6: the edit/delete sweep against the fake TDLib and a real Postgres (A3 minus windows)."""

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_ingest import ACCOUNT, NOW, SETTINGS, A, FakeClock, chat, dirty_from, enable, stored
from typer.testing import CliRunner

from tgi import ingest
from tgi.cli import app
from tgi.config import OnDelete
from tgi.td_fake import FakeTdClient

# The window step clears the marker; these tests check the marker the writers leave
# (`synced()` asserts it): all tests here but the chat-start one, so it is set module-wide.
pytestmark = pytest.mark.usefixtures("no_window_step")

Conn = psycopg.Connection[tuple[object, ...]]
EDITED = list(range(601, 606))
DELETED_IN = list(range(610, 617))
DELETED_OUT = [490, 100, 50]  # 18, 408 and 458 days old: outside the 7-day sweep


def history() -> list[dict[str, Any]]:
    """Ids 1..500 one per day (500 is 8 days old), then 501..640 hourly up to a day ago."""
    daily = [(i, NOW - timedelta(days=8 + 500 - i)) for i in range(1, 501)]
    hourly = [(i, NOW - timedelta(days=1, hours=640 - i)) for i in range(501, 641)]
    return [{"id": i, "date": d.isoformat(), "text": f"message {i}"} for i, d in daily + hourly]


def a3_changes(**extra: Any) -> dict[str, Any]:
    """A3: 50 new messages, 5 edited and 7 deleted inside the window, 3 deleted outside."""
    return {
        "add": [
            {"chat_id": A, "id": i, "date": (NOW - timedelta(minutes=690 - i)).isoformat(),
             "text": f"message {i}"}
            for i in range(641, 691)
        ],
        "edit": [
            {"chat_id": A, "id": i, "text": f"message {i} (edited)",
             "edit_date": (NOW - timedelta(minutes=5)).isoformat()}
            for i in EDITED
        ],
        "delete": [{"chat_id": A, "id": i} for i in DELETED_IN + DELETED_OUT],
        **extra,
    }  # fmt: skip


def sync(
    conn: Conn, td: FakeTdClient, on_delete: OnDelete, resweep: tuple[int, int] | None = None
) -> ingest.SyncResult:
    settings = replace(SETTINGS, on_delete=on_delete)
    return ingest.sync(conn, lambda _: td, settings, FakeClock(), resweep=resweep)


def synced(conn: Conn, on_delete: OnDelete, runs: list[dict[str, Any]]) -> FakeTdClient:
    td = FakeTdClient({"chats": [chat(A, history())], "runs": runs})
    enable(conn, td, A)
    first = sync(conn, td, on_delete)
    assert first.status == "ok"
    assert (first.stats["tombstoned"], first.stats["purged"], first.stats["edited"]) == (0, 0, 0)
    assert stored(conn, A) == list(range(1, 641))
    assert dirty_from(conn, A) == 1
    windows_rebuilt(conn)
    td.next_run()
    return td


def windows_rebuilt(conn: Conn) -> None:
    """Stand-in for the T7 window step: the marker is cleared once windows are current."""
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")


def deleted(conn: Conn) -> list[int]:
    rows = conn.execute(
        "SELECT server_message_id FROM message WHERE chat_id = %s AND deleted_at IS NOT NULL"
        " ORDER BY 1",
        (A,),
    )
    return [r[0] for r in rows]


def texts(conn: Conn, ids: list[int]) -> list[str]:
    rows = conn.execute(
        "SELECT text FROM message WHERE chat_id = %s AND server_message_id = ANY(%s) ORDER BY 1",
        (A, ids),
    )
    return [r[0] for r in rows]


@pytest.mark.parametrize("on_delete", ["tombstone", "purge"])
def test_a3_sweep_applies_edits_and_deletions_inside_the_window(
    conn: Conn, on_delete: OnDelete
) -> None:
    td = synced(conn, on_delete, [a3_changes()])
    result = sync(conn, td, on_delete)

    assert result.status == "ok"
    assert result.stats["edited"] == 5
    assert texts(conn, EDITED) == [f"message {i} (edited)" for i in EDITED]
    kept = [i for i in range(1, 691) if i not in DELETED_IN]
    if on_delete == "tombstone":
        assert result.stats["tombstoned"] == 7
        assert stored(conn, A) == list(range(1, 691))
        assert deleted(conn) == DELETED_IN
    else:
        assert result.stats["purged"] == 7
        assert stored(conn, A) == kept
        assert deleted(conn) == []
    # The 3 outside deletions are still stored and live; the 50 new ones are stored.
    assert set(DELETED_OUT + list(range(641, 691))) <= set(stored(conn, A)) - set(deleted(conn))
    # The first edit (601) is below the first deletion (610) and the first new message (641).
    assert dirty_from(conn, A) == 601


def test_truncated_sweep_tombstones_nothing_until_a_complete_one(conn: Conn) -> None:
    # Call 1 is catch-up, call 2 the sweep's first page; its second page floods 3 times.
    flood = [{"call": n, "flood_wait": 1} for n in (3, 4, 5)]
    td = synced(conn, "tombstone", [a3_changes(faults=flood), {}])
    result = sync(conn, td, "tombstone")
    assert result.status == "flood_limited"
    assert result.stats["edited"] == 5  # pages received before the stop are still applied
    assert result.stats["tombstoned"] == 0
    assert deleted(conn) == []

    td.next_run()
    result = sync(conn, td, "tombstone")
    assert (result.status, result.stats["tombstoned"]) == ("ok", 7)
    assert deleted(conn) == DELETED_IN


def test_resweep_365_days_tombstones_one_outside_deletion(conn: Conn) -> None:
    td = synced(conn, "tombstone", [a3_changes()])
    assert sync(conn, td, "tombstone").status == "ok"
    windows_rebuilt(conn)

    result = sync(conn, td, "tombstone", resweep=(A, 365))
    assert (result.status, result.stats["tombstoned"]) == ("ok", 1)
    # 490 is 18 days old; 100 and 50 lie before the 365-day cutoff and stay live.
    assert deleted(conn) == sorted(DELETED_IN + [490])
    assert dirty_from(conn, A) == 490


def test_reappearing_message_is_untombstoned(conn: Conn) -> None:
    back = {"chat_id": A, "id": 612, "date": (NOW - timedelta(days=1, hours=28)).isoformat(),
            "text": "message 612"}  # fmt: skip
    td = synced(conn, "tombstone", [a3_changes(), {"add": [back]}])
    sync(conn, td, "tombstone")
    windows_rebuilt(conn)
    td.next_run()
    result = sync(conn, td, "tombstone")
    assert (result.stats["untombstoned"], result.stats["tombstoned"]) == (1, 0)
    assert deleted(conn) == [i for i in DELETED_IN if i != 612]
    assert dirty_from(conn, A) == 612


def test_cli_resweep_days(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    td = synced(conn, "tombstone", [a3_changes()])
    sync(conn, td, "tombstone")
    monkeypatch.setenv("DATABASE_URL", db_url)
    # The CLI's fresh fake holds only the original 640 messages: 610..616 reappear and
    # 641..690 vanish.
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"chats": [chat(A, history())]}))
    base = ["sync", "--scenario", str(scenario), "--resweep-days"]
    runner = CliRunner()

    result = runner.invoke(app, [*base, "30", str(A)])
    assert result.exit_code == 0, result.output
    assert "tombstoned=50" in result.output
    assert "untombstoned=7" in result.output
    assert ACCOUNT not in result.output
    assert runner.invoke(app, [*base, "30"]).exit_code == 1  # CHAT_ID missing
    unknown = runner.invoke(app, [*base, "30", "-1001999999999"])
    assert unknown.exit_code == 1
    assert "not in the archive" in unknown.output


def test_sweep_without_changes_leaves_the_marker_clear(conn: Conn) -> None:
    td = synced(conn, "tombstone", [{}])
    assert sync(conn, td, "tombstone").status == "ok"
    assert dirty_from(conn, A) is None


def test_deep_resweep_edits_stored_messages_older_than_the_backfill_cutoff(conn: Conn) -> None:
    """Backfill 1 year, a stored 2-year-old message, resweep 1000 days (review, PR #6)."""
    old = {"id": 1, "date": (NOW - timedelta(days=730)).isoformat(), "text": "old"}
    recent = {"id": 2, "date": (NOW - timedelta(days=1)).isoformat(), "text": "recent"}
    edit = {"chat_id": A, "id": 1, "text": "old (edited)",
            "edit_date": (NOW - timedelta(minutes=5)).isoformat()}  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, [old, recent])], "runs": [{"edit": [edit]}]})
    enable(conn, td, A)
    assert sync(conn, td, "tombstone").status == "ok"  # BACKFILL_YEARS=3 stores both
    conn.execute("UPDATE chat SET backfill_years = 1, windows_dirty_from = NULL")
    td.next_run()

    result = sync(conn, td, "tombstone", resweep=(A, 1000))
    assert (result.status, result.stats["edited"]) == ("ok", 1)
    assert texts(conn, [1]) == ["old (edited)"]
    assert dirty_from(conn, A) == 1


@pytest.mark.parametrize("on_delete", ["tombstone", "purge"])
@pytest.mark.parametrize("call", [2, 3], ids=["first-page", "later-page"])
def test_untrusted_empty_page_deletes_nothing_until_the_next_run(
    conn: Conn, on_delete: OnDelete, call: int, caplog: pytest.LogCaptureFixture
) -> None:
    """SPEC v1.7: a spurious empty page with live stored window messages behind it."""
    # Call 1 is catch-up; call 2 is the sweep's first page and call 3 its second.
    td = synced(conn, on_delete, [a3_changes(faults=[{"call": call, "empty": True}]), {}])
    caplog.set_level("INFO")
    result = sync(conn, td, on_delete)
    assert result.status == "ok"
    assert (result.stats["tombstoned"], result.stats["purged"]) == (0, 0)
    assert set(DELETED_IN) <= set(stored(conn, A)) and deleted(conn) == []
    assert f"sweep chat_id={A} incomplete: untrusted empty page" in caplog.text
    assert "message " not in caplog.text

    td.next_run()
    result = sync(conn, td, on_delete)
    assert result.status == "ok"
    assert result.stats["purged" if on_delete == "purge" else "tombstoned"] == 7
    assert not set(DELETED_IN) & (set(stored(conn, A)) - set(deleted(conn)))


def test_fully_deleted_window_above_older_messages_is_tombstoned_in_one_sweep(
    conn: Conn,
) -> None:
    window = list(range(501, 641))  # every message inside the 7-day window
    td = synced(conn, "tombstone", [{"delete": [{"chat_id": A, "id": i} for i in window]}])
    result = sync(conn, td, "tombstone")
    assert (result.status, result.stats["tombstoned"]) == ("ok", 140)
    assert deleted(conn) == window
    assert dirty_from(conn, A) == 501


def test_empty_page_at_the_chat_start_is_trusted(conn: Conn) -> None:
    young = [{"id": i, "date": (NOW - timedelta(hours=40 - i)).isoformat(), "text": f"m {i}"}
             for i in range(1, 31)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, young)], "runs": [{"delete": [{"chat_id": A, "id": 15}]}]})
    enable(conn, td, A)
    assert sync(conn, td, "tombstone").status == "ok"
    td.next_run()
    result = sync(conn, td, "tombstone")
    assert (result.status, result.stats["tombstoned"]) == ("ok", 1)
    assert deleted(conn) == [15]
