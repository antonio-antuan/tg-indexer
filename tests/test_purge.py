"""P2.T6: `tgi purge --deleted --older-than Nd` (D-TGI-3, §4 marker rule)."""

from typing import Any

import psycopg
import pytest
import typer
from test_ingest import SETTINGS
from typer.testing import CliRunner

from tgi import ingest, windows
from tgi.cli import app

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
A, B = -1001000000201, -1001000000202
# chat → (server id, days since tombstoned or None for live); B is disabled.
MESSAGES = {
    A: [(1, None), (2, 30), (3, 91), (4, None), (5, 200)],
    B: [(1, 200), (2, 10), (3, None)],
}


def seed(conn: Conn) -> None:
    for chat_id, enabled in ((A, True), (B, False)):
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind, enabled)"
            " VALUES (%s, '+0', 'chat', 'supergroup', %s)",
            (chat_id, enabled),
        )
        for sid, age in MESSAGES[chat_id]:
            conn.execute(
                "INSERT INTO message (chat_id, message_id, server_message_id, date, text,"
                " deleted_at) VALUES (%s, %s, %s, now() - interval '300 days' + %s * interval"
                " '1 minute', 'synthetic text', now() - %s * interval '1 day')",
                (chat_id, sid << SHIFT, sid, sid, age),
            )


def stored(conn: Conn) -> dict[int, set[int]]:
    """chat → server ids still stored."""
    out: dict[int, set[int]] = {A: set(), B: set()}
    for chat_id, sid in conn.execute("SELECT chat_id, server_message_id FROM message"):
        out[chat_id].add(sid)  # type: ignore[index]
    return out


def markers(conn: Conn) -> dict[Any, Any]:
    return dict(conn.execute("SELECT chat_id, windows_dirty_from FROM chat").fetchall())


def purge(db_url: str, monkeypatch: pytest.MonkeyPatch, *args: str, input: str | None = None):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATABASE_URL", db_url)
    return CliRunner().invoke(app, ["purge", *args], input=input)


@pytest.fixture
def seeded(conn: Conn) -> Conn:
    seed(conn)
    return conn


def test_purges_exactly_the_tombstones_older_than_the_age(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", "--older-than", "90d", "--yes")
    assert result.exit_code == 0, result.output
    assert stored(seeded) == {A: {1, 2, 4}, B: {2, 3}}  # 91 and 200 days gone, 30 kept
    # By chat id: B (-…202) sorts before A (-…201).
    assert result.stdout.splitlines() == [f"purged chat {B}: 1", f"purged chat {A}: 2", "purged 3"]
    # The marker is each chat's lowest purged id (§4), so the next sync rebuilds from there.
    assert markers(seeded) == {A: 3 << SHIFT, B: 1 << SHIFT}


def test_the_default_age_is_90_days(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert stored(seeded) == {A: {1, 2, 4}, B: {2, 3}}


def test_zero_days_purges_every_tombstone_and_no_live_message(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", "--yes")
    assert result.exit_code == 0, result.output
    assert stored(seeded) == {A: {1, 4}, B: {3}}
    assert "purged 5" in result.stdout


def test_a_held_lock_exits_2_and_deletes_nothing(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        result = purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", "--yes")
    assert result.exit_code == 2
    assert "another tgi sync is running" in result.stderr
    assert len(stored(seeded)[A]) == 5 and markers(seeded) == {A: None, B: None}


def test_the_marker_only_moves_down(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded.execute("UPDATE chat SET windows_dirty_from = %s WHERE chat_id = %s", (1 << SHIFT, A))
    assert purge(db_url, monkeypatch, "--deleted", "--yes").exit_code == 0
    assert markers(seeded)[A] == 1 << SHIFT


def test_the_rebuild_after_a_purge_changes_no_window(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tombstones belong to no window, so the marker's rebuild is a no-op (PLAN T6)."""
    seeded.execute("UPDATE chat SET windows_dirty_from = %s", (1 << SHIFT,))
    windows.rebuild_dirty(seeded, SETTINGS)
    seeded.execute('UPDATE "window" SET stale = false')
    query = 'SELECT window_id, chat_id, message_ids, text_hash, stale FROM "window" ORDER BY 1'
    before = seeded.execute(query).fetchall()
    assert before

    assert purge(db_url, monkeypatch, "--deleted", "--older-than", "0d", "--yes").exit_code == 0
    windows.rebuild_dirty(seeded, SETTINGS)
    assert seeded.execute(query).fetchall() == before
    assert markers(seeded) == {A: None, B: None}


def test_asks_first_showing_the_counts_and_yes_purges(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", input="y\n")
    assert result.exit_code == 0, result.output
    assert result.stdout.startswith(f"chat {B}: 1\nchat {A}: 2\nDelete these 3 tombstoned")
    assert stored(seeded) == {A: {1, 2, 4}, B: {2, 3}}


@pytest.mark.parametrize("answer", ["n\n", "\n", None], ids=["no", "default", "no-input"])
def test_anything_but_yes_deletes_nothing(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch, answer: str | None
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", input=answer)
    assert result.exit_code == 1
    assert "Aborted" in result.output
    assert len(stored(seeded)[A]) == 5 and markers(seeded) == {A: None, B: None}


def test_the_lock_is_held_while_the_prompt_waits(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    held: list[bool] = []

    def confirm(*args: Any, **kwargs: Any) -> bool:
        row = seeded.execute("SELECT pg_try_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        held.append(not row.fetchone()[0])  # type: ignore[index]
        return True

    monkeypatch.setattr(typer, "confirm", confirm)
    assert purge(db_url, monkeypatch, "--deleted").exit_code == 0
    assert held == [True]


def test_nothing_to_purge_does_not_ask(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", "--older-than", "365d")
    assert result.exit_code == 0
    assert result.stdout == "nothing to purge\n"
    huge = purge(db_url, monkeypatch, "--deleted", "--older-than", "9" * 30 + "d")
    assert huge.exit_code == 0 and huge.stdout == "nothing to purge\n"
    assert len(stored(seeded)[A]) == 5


def test_without_deleted_it_is_a_usage_error(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = purge(db_url, monkeypatch, "--older-than", "0d", "--yes")
    assert result.exit_code == 1
    assert "--deleted is required" in result.stderr
    assert len(stored(seeded)[A]) == 5


@pytest.mark.parametrize(
    "age", ["77", "7w", "-7d", "1.5d", "d", "", " 77d", "٣d", "7" * 5000 + "d", "SECRET77d"]
)
def test_older_than_takes_whole_days_and_never_echoes_the_value(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch, age: str
) -> None:
    result = purge(db_url, monkeypatch, "--deleted", "--older-than", age, "--yes")
    assert result.exit_code == 1
    assert "whole number of days" in result.stderr
    if len(age.strip()) > 1:
        assert age.strip() not in result.output
    assert len(stored(seeded)[A]) == 5


def test_forget_exits_2_while_a_purge_prompt_is_open(
    db_url: str, seeded: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`chats forget` takes the sync lock, so it exits 2 while purge's prompt is open."""
    forgets: list[Any] = []

    def confirm(*args: Any, **kwargs: Any) -> bool:
        forgets.append(CliRunner().invoke(app, ["chats", "forget", str(B), "--yes"]))
        return True

    monkeypatch.setattr(typer, "confirm", confirm)
    result = purge(db_url, monkeypatch, "--deleted")
    assert result.exit_code == 0, result.output
    assert forgets[0].exit_code == 2
    assert "another tgi sync is running" in forgets[0].stderr
    assert result.stdout.splitlines()[-3:] == [
        f"purged chat {B}: 1", f"purged chat {A}: 2", "purged 3"
    ]  # fmt: skip
    assert stored(seeded) == {A: {1, 2, 4}, B: {2, 3}}  # B was not forgotten

    # With the lock free, forget works as before.
    done = CliRunner().invoke(app, ["chats", "forget", str(B), "--yes"])
    assert done.exit_code == 0 and stored(seeded)[B] == set()
