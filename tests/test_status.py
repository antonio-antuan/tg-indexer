"""`tgi status` and the MCP `status` tool: how current the archive is."""

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import psycopg
import pytest
from test_embed import sync
from test_ingest import ACCOUNT, NOW, A, B, chat, enable, messages
from test_mcp import in_process
from typer.testing import CliRunner

from tgi import chats, ingest, status
from tgi.cli import app
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]


@pytest.fixture
def archive(conn: Conn) -> None:
    td = FakeTdClient({
        "chats": [chat(A, messages(1, 30, timedelta(hours=1)), username="a_chat"),
                  chat(B, messages(1, 5, timedelta(hours=1)))],
        "runs": [],
    })  # fmt: skip
    enable(conn, td, A, B)
    conn.execute("UPDATE chat SET backfill_years = 2 WHERE chat_id = %s", (B,))
    assert sync(conn, td).status == "ok"
    conn.execute("INSERT INTO chat_tag (chat_id, tag, status) VALUES (%s, 'v', 'confirmed')", (A,))


def by_chat(answer: dict) -> dict[int, dict]:  # type: ignore[type-arg]
    return {c["chat_id"]: c for c in answer["chats"]}


def test_status_reports_runs_chats_and_the_embedding_backlog(archive: None, conn: Conn) -> None:
    answer = status.status(conn, now=NOW, default_years=3)
    assert answer["now"] == NOW.isoformat() and answer["sync_running"] is False
    assert answer["last_run"]["status"] == "ok"
    assert answer["last_run"]["stats"]["upserted"] == 35
    assert answer["last_ok_run"] == answer["last_run"]
    assert answer["windows"]["waiting_for_embedding"] == 0
    a, b = by_chat(answer)[A], by_chat(answer)[B]
    assert a["tags"] == ["v"] and a["history_complete"] is True and a["enabled"] is True
    assert a["newest_message_at"] == NOW.isoformat()
    assert a["oldest_message_at"] == (NOW - timedelta(hours=29)).isoformat()
    assert (a["backfill_years"], b["backfill_years"]) == (3, 2)  # default vs the chat's own
    assert a["last_sync_at"] is not None


def test_status_shows_pending_embeddings_and_an_incomplete_run(archive: None, conn: Conn) -> None:
    conn.execute('UPDATE "window" SET stale = true WHERE chat_id = %s', (A,))
    conn.execute("INSERT INTO sync_run (status, finished_at) VALUES ('time_limited', now())")
    answer = status.status(conn, now=NOW)
    assert answer["windows"]["waiting_for_embedding"] > 0
    assert by_chat(answer)[A]["windows_waiting_for_embedding"] > 0
    assert by_chat(answer)[B]["windows_waiting_for_embedding"] == 0
    assert answer["last_run"]["status"] == "time_limited"
    assert answer["last_ok_run"]["status"] == "ok"  # the earlier one


def test_status_sees_a_running_sync(archive: None, conn: Conn, db_url: str) -> None:
    with psycopg.connect(db_url) as other, ingest.sync_lock(other):
        assert status.status(conn)["sync_running"] is True
    assert status.status(conn)["sync_running"] is False


def test_status_filters_chats_and_never_shows_the_account(archive: None, conn: Conn) -> None:
    answer = status.status(conn, chat_ids=[B])
    assert [c["chat_id"] for c in answer["chats"]] == [B]
    assert ACCOUNT not in json.dumps(status.status(conn))


def test_status_on_an_empty_archive(conn: Conn) -> None:
    answer = status.status(conn)
    assert answer["last_run"] is None and answer["last_ok_run"] is None
    assert answer["chats"] == [] and answer["windows"] == {"total": 0, "waiting_for_embedding": 0}


def test_status_tool_over_mcp(archive: None, db_url: str) -> None:
    everything, tagged, unknown = in_process(db_url, [
        ("status", {}),
        ("status", {"chat_ids": [A]}),
        ("status", {"chat_ids": [123]}),
    ])  # fmt: skip
    assert {c["chat_id"] for c in everything["chats"]} == {A, B}
    assert [c["chat_id"] for c in tagged["chats"]] == [A]
    assert unknown["error"]["code"] == "unknown_chat"


def test_cli_status(
    archive: None, db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    out = CliRunner().invoke(app, ["status", "--json"])
    assert out.exit_code == 0, out.output
    assert {c["chat_id"] for c in json.loads(out.output)["chats"]} == {A, B}
    text = CliRunner().invoke(app, ["status"])
    assert text.exit_code == 0 and f"chat {A}" in text.output and "last run: ok" in text.output


def test_a_deeper_lookback_reads_incomplete_before_the_next_sync(archive: None, conn: Conn) -> None:
    assert by_chat(status.status(conn, now=NOW, default_years=3))[A]["history_complete"] is True
    # a larger BACKFILL_YEARS for a chat without its own years
    assert by_chat(status.status(conn, now=NOW, default_years=5))[A]["history_complete"] is False
    # a larger --years on the chat itself, through `tgi chats enable`
    td = FakeTdClient({"chats": [chat(B, [])], "runs": []})
    chats.enable(conn, td.get_chat(B), ACCOUNT, 4, default_years=3)
    answer = by_chat(status.status(conn, now=NOW, default_years=3))
    assert answer[B]["history_complete"] is False and answer[B]["backfill_years"] == 4
    assert answer[A]["history_complete"] is True


def test_a_lock_in_another_database_is_not_a_running_sync(
    archive: None, conn: Conn, create_database: Callable[[], str]
) -> None:
    with psycopg.connect(create_database()) as elsewhere, ingest.sync_lock(elsewhere):
        assert status.status(conn)["sync_running"] is False


def test_the_connection_holding_the_lock_sees_it(archive: None, conn: Conn) -> None:
    with ingest.sync_lock(conn):
        assert status.status(conn)["sync_running"] is True


def test_dates_skip_tombstones_and_tags_skip_proposals(archive: None, conn: Conn) -> None:
    newest = conn.execute("SELECT max(message_id) FROM message WHERE chat_id = %s", (A,)).fetchone()
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s",
        (A, newest[0] if newest else None),
    )
    conn.execute("INSERT INTO chat_tag (chat_id, tag, status) VALUES (%s, 'p', 'proposed')", (A,))
    oldest = conn.execute("SELECT min(message_id) FROM message WHERE chat_id = %s", (A,)).fetchone()
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s",
        (A, oldest[0] if oldest else None),
    )
    a = by_chat(status.status(conn, now=NOW))[A]
    assert a["newest_message_at"] == (NOW - timedelta(hours=1)).isoformat()
    assert a["oldest_message_at"] == (NOW - timedelta(hours=28)).isoformat()
    assert a["tags"] == ["v"]


def test_messages_waiting_for_the_window_step_are_flagged(archive: None, conn: Conn) -> None:
    assert by_chat(status.status(conn))[A]["windows_need_rebuild"] is False
    conn.execute("UPDATE chat SET windows_dirty_from = 1 WHERE chat_id = %s", (A,))
    answer = by_chat(status.status(conn))
    assert answer[A]["windows_need_rebuild"] is True and answer[B]["windows_need_rebuild"] is False


def test_the_schedule_is_reported_when_known(archive: None, conn: Conn) -> None:
    assert status.status(conn, now=NOW)["schedule"] == {"every_minutes": None, "next_run_at": None}
    hourly = status.status(conn, now=NOW + timedelta(minutes=10), sync_every_minutes=60)
    assert hourly["schedule"] == {
        "every_minutes": 60, "next_run_at": (NOW + timedelta(hours=1)).isoformat()
    }  # fmt: skip


def test_cli_status_filters_chats(
    archive: None, db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    out = CliRunner().invoke(app, ["status", "--chat", str(B), "--json"])
    assert [c["chat_id"] for c in json.loads(out.output)["chats"]] == [B]
    text = CliRunner().invoke(app, ["status", "--chat", str(B)])
    assert f"chat {A}" not in text.output and "history complete" in text.output


def test_history_complete_with_a_null_cutoff_or_unknown_default(archive: None, conn: Conn) -> None:
    # a chat finished before migration 0003: done, but no cutoff recorded
    conn.execute("UPDATE chat SET backfill_cutoff = NULL")
    known = by_chat(status.status(conn, now=NOW, default_years=3))
    assert known[A]["history_complete"] is False and known[B]["history_complete"] is False
    # a caller that does not know BACKFILL_YEARS: A (no years of its own) keeps backfill_done,
    # B (its own years) is still judged against its cutoff
    unknown = by_chat(status.status(conn, now=NOW))
    assert unknown[A]["history_complete"] is True and unknown[A]["backfill_years"] is None
    assert unknown[B]["history_complete"] is False
