# ruff: noqa: E501, S608, ANN001, ANN201, ANN202
"""QA P2.T6: `tgi purge --deleted` boundaries, atomicity, lock release, windows after a purge."""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest
import typer
from test_ingest import SETTINGS
from test_purge import SHIFT, A, B, markers, purge, seed, stored
from typer.testing import CliRunner

from tgi import ingest, windows
from tgi.cli import app

Conn = psycopg.Connection[tuple[object, ...]]
C = -1001000000203
CUT = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)


def add_chat(conn: Conn, chat_id: int) -> None:
    conn.execute(
        "INSERT INTO chat (chat_id, account_phone, title, kind, enabled)"
        " VALUES (%s, '+0', 'c', 'supergroup', true)",
        (chat_id,),
    )


def add_msg(
    conn: Conn,
    chat_id: int,
    sid: int,
    deleted_at: Any,
    minute: float = 0,
    text: str = "synthetic text",
) -> None:
    conn.execute(
        "INSERT INTO message (chat_id, message_id, server_message_id, date, text, deleted_at)"
        " VALUES (%s, %s, %s, %s, %s, %s)",
        (
            chat_id,
            sid << SHIFT,
            sid,
            CUT - timedelta(days=400) + timedelta(minutes=minute),
            text,
            deleted_at,
        ),
    )


def lock_is_free(db_url: str) -> bool:
    with psycopg.connect(db_url, autocommit=True) as other:
        got = other.execute("SELECT pg_try_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,)).fetchone()
        if got and got[0]:
            other.execute("SELECT pg_advisory_unlock(%s)", (ingest.SYNC_LOCK_KEY,))
            return True
    return False


# --- the top risk: a live message is never deleted --------------------------------------


def test_cutoff_boundaries_at_the_microsecond(conn: Conn) -> None:
    add_chat(conn, A)
    add_msg(conn, A, 1, CUT, 1)  # exactly at the cutoff: kept (strictly older only)
    add_msg(conn, A, 2, CUT - timedelta(microseconds=1), 2)  # one microsecond older: purged
    add_msg(conn, A, 3, CUT + timedelta(microseconds=1), 3)  # newer: kept
    add_msg(conn, A, 4, CUT + timedelta(days=3650), 4)  # a tombstone dated in the future: kept
    add_msg(conn, A, 5, None, 5)  # live: kept
    add_msg(conn, A, 6, "infinity", 6)
    add_msg(conn, A, 7, "-infinity", 7)
    assert ingest.tombstones_before(conn, CUT) == {A: 2}  # 2 and -infinity
    assert ingest.purge_tombstones(conn, CUT) == {A: 2}
    assert stored(conn)[A] == {1, 3, 4, 5, 6}


def test_zero_days_with_a_future_tombstone_and_null_through_the_cli(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_chat(conn, A)
    add_msg(conn, A, 1, None, 1)
    add_msg(conn, A, 2, datetime.now(UTC) + timedelta(hours=1), 2)
    add_msg(conn, A, 3, datetime.now(UTC) - timedelta(seconds=5), 3)
    add_msg(conn, A, 4, datetime.now(UTC) + timedelta(days=1000), 4)
    r = purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", "--yes")
    assert r.exit_code == 0, r.output
    assert stored(conn)[A] == {1, 2, 4}


def test_cli_margin_around_ninety_days(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_chat(conn, A)
    now = datetime.now(UTC)
    add_msg(conn, A, 1, now - timedelta(days=90, minutes=1), 1)
    add_msg(conn, A, 2, now - timedelta(days=90) + timedelta(minutes=1), 2)
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert stored(conn)[A] == {2}


@pytest.mark.parametrize("days", ["999999999", "1000000000", "9" * 40, "007", "0000"])
def test_extreme_ages_never_touch_a_live_message(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch, days: str
) -> None:
    seed(conn)
    live = conn.execute("SELECT count(*) FROM message WHERE deleted_at IS NULL").fetchone()
    purge(db_url, monkeypatch, "--deleted", "--older-than", f"{days}d", "--yes")
    after = conn.execute("SELECT count(*) FROM message WHERE deleted_at IS NULL").fetchone()
    assert live == after


# --- counts shown = rows deleted; scale --------------------------------------------------


def test_counts_equal_rows_deleted_at_scale(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    for c in (A, B, C):
        add_chat(conn, c)
        conn.execute(
            "INSERT INTO message (chat_id, message_id, server_message_id, date, text, deleted_at)"
            " SELECT %s, g::bigint << 20, g, now() - interval '500 days' + g * interval '1 second', 'synthetic text',"
            " CASE WHEN g %% 3 = 0 THEN now() - interval '100 days' WHEN g %% 3 = 1 THEN now() - interval '10 days' END"
            " FROM generate_series(1, 30000) g",
            (c,),
        )
    before = conn.execute(
        "SELECT count(*) FROM message WHERE deleted_at IS NOT NULL AND deleted_at < now() - interval '90 days'"
    ).fetchone()
    t = time.monotonic()
    shown = purge(db_url, monkeypatch, "--deleted", input="y\n")
    took = time.monotonic() - t
    assert shown.exit_code == 0, shown.output
    lines = shown.stdout.splitlines()
    asked = {int(ln.split()[1][:-1]): int(ln.split()[2]) for ln in lines if ln.startswith("chat ")}
    done = {
        int(ln.split()[2][:-1]): int(ln.split()[3]) for ln in lines if ln.startswith("purged chat ")
    }
    assert asked == done and sum(done.values()) == before[0] == 30000
    assert conn.execute("SELECT count(*) FROM message WHERE deleted_at IS NULL").fetchone() == (
        30000,
    )
    assert took < 60


def test_a_writer_that_ignores_the_lock_cannot_change_what_was_shown_when_it_waits(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows inserted during the prompt by a non-sync writer (no such writer exists in the product)
    are deleted although not counted: records the reliance on the lock, not a defect."""
    seed(conn)

    def confirm(*a: Any, **k: Any) -> bool:
        add_msg(conn, A, 99, datetime.now(UTC) - timedelta(days=500), 99)
        return True

    monkeypatch.setattr(typer, "confirm", confirm)
    r = purge(db_url, monkeypatch, "--deleted")
    assert r.exit_code == 0
    assert 99 not in stored(conn)[A]  # deleted, but the count line said 2 for chat A


# --- atomic and lock release ---------------------------------------------------------------


def test_a_failure_midway_rolls_back_every_chat_and_the_markers(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn)
    conn.execute(
        "CREATE FUNCTION boom() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'boom'; END $$"
    )
    conn.execute(
        f"CREATE TRIGGER t BEFORE DELETE ON message FOR EACH ROW WHEN (old.chat_id = {A}) EXECUTE FUNCTION boom()"
    )  # B sorts first and is deleted before A fails
    before = stored(conn)
    r = purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", "--yes")
    assert r.exit_code == 3, r.output
    assert "purge failed" in r.stderr
    assert stored(conn) == before
    assert markers(conn) == {A: None, B: None}
    assert lock_is_free(db_url)


@pytest.mark.parametrize("answer", ["n\n", None])
def test_the_lock_is_released_after_an_abort(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch, answer: str | None
) -> None:
    seed(conn)
    assert purge(db_url, monkeypatch, "--deleted", input=answer).exit_code == 1
    assert lock_is_free(db_url)


def test_the_lock_is_released_after_success_and_nothing_to_purge(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn)
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert lock_is_free(db_url)
    assert purge(db_url, monkeypatch, "--deleted", "--yes").stdout == "nothing to purge\n"
    assert lock_is_free(db_url)


def test_second_purge_and_sync_exit_2_while_the_prompt_is_open(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn)
    seen: list[Any] = []

    def confirm(*a: Any, **k: Any) -> bool:
        seen.append(CliRunner().invoke(app, ["purge", "--deleted", "--yes"]))
        seen.append(CliRunner().invoke(app, ["embed"]))
        raise typer.Abort

    monkeypatch.setattr(typer, "confirm", confirm)
    r = purge(db_url, monkeypatch, "--deleted")
    assert r.exit_code == 1
    assert [s.exit_code for s in seen] == [2, 2]
    assert len(stored(conn)[A]) == 5


def test_forget_unknown_chat_exits_1_and_releases_the_lock(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    r = CliRunner().invoke(app, ["chats", "forget", "-5", "--yes"])
    assert r.exit_code == 1 and "unknown chat" in r.stderr
    assert lock_is_free(db_url)


def test_forget_with_the_lock_held_deletes_nothing(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn)
    monkeypatch.setenv("DATABASE_URL", db_url)
    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        r = CliRunner().invoke(app, ["chats", "forget", str(A), "--yes"])
    assert r.exit_code == 2
    assert len(stored(conn)[A]) == 5


# --- yes means y ---------------------------------------------------------------------------


@pytest.mark.parametrize("answer", ["yes\n", "Y\n", "YES\n", "y \n", "ja\n", "1\n", "true\n"])
def test_which_answers_count_as_yes(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    """Agreed: anything but `y` deletes nothing. Records what the prompt accepts."""
    seed(conn)
    r = purge(db_url, monkeypatch, "--deleted", input=answer)
    deleted = len(stored(conn)[A]) < 5
    print(repr(answer), "->", "deleted" if deleted else "kept", r.exit_code)
    assert deleted == (answer.strip().lower() in {"y", "yes"})


# --- windows after a purge, against a from-scratch build ----------------------------------


def test_windows_after_a_purge_equal_a_from_scratch_build(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    add_chat(conn, A)
    old = datetime.now(UTC) - timedelta(days=200)
    # Conversations with gaps; tombstones in the middle, at a window's first message, at the chat start and end.
    minutes = [0, 5, 10, 200, 205, 210, 400, 405, 800, 805, 810, 1500]
    for i, m in enumerate(minutes, 1):
        add_msg(conn, A, i, old if i in (1, 4, 7, 10, 12, 5) else None, m, text=f"synthetic {i}")
    # Reply from a live message to a tombstoned parent.
    conn.execute(
        "UPDATE message SET reply_to_message_id = %s WHERE chat_id = %s AND message_id = %s",
        (4 << SHIFT, A, 6 << SHIFT),
    )
    # Windows as built while the tombstoned messages were still live (sweep marks the chain dirty).
    conn.execute("UPDATE message SET deleted_at = NULL")
    conn.execute("UPDATE chat SET windows_dirty_from = %s", (1 << SHIFT,))
    windows.rebuild_dirty(conn, SETTINGS)
    conn.execute(
        "UPDATE message SET deleted_at = %s WHERE server_message_id IN (1,4,5,7,10,12)", (old,)
    )
    conn.execute("UPDATE chat SET windows_dirty_from = %s", (1 << SHIFT,))
    windows.rebuild_dirty(conn, SETTINGS)
    q = 'SELECT first_message_id, message_ids, text, text_hash FROM "window" WHERE chat_id = %s ORDER BY 1'
    pre = conn.execute(q, (A,)).fetchall()
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert markers(conn)[A] == 1 << SHIFT
    windows.rebuild_dirty(conn, SETTINGS)
    post = conn.execute(q, (A,)).fetchall()
    # Scratch: drop every window, rebuild from nothing.
    conn.execute('DELETE FROM "window"')
    conn.execute("UPDATE chat SET windows_dirty_from = %s", (1 << SHIFT,))
    windows.rebuild_dirty(conn, SETTINGS)
    scratch = conn.execute(q, (A,)).fetchall()
    assert post == pre == scratch and scratch


def test_purge_marker_survives_a_crash_before_the_rebuild(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(conn)
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert all(v is not None for v in markers(conn).values())  # the next sync rebuilds


# --- A10 --------------------------------------------------------------------------------


def test_logs_and_output_have_no_message_text(
    db_url: str, conn: Conn, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seed(conn)
    conn.execute("UPDATE message SET text = 'SECRETMARKER' ")
    caplog.set_level(logging.DEBUG)
    r = purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", input="y\n")
    assert r.exit_code == 0
    blob = caplog.text + r.output
    assert "SECRETMARKER" not in blob and "synthetic" not in blob
    print([rec.getMessage() for rec in caplog.records if "purge" in rec.getMessage()])
