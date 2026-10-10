"""T5: catch-up and backfill against the fake TDLib and a real Postgres (A2, A3, A4, A10)."""

import json
import logging
import threading
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import psycopg
import pytest
from typer.testing import CliRunner

from tgi import chats, cli, embed, ingest
from tgi.cli import app
from tgi.config import Settings
from tgi.td_fake import FakeTdClient
from tgi.td_port import Chat, TdError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACCOUNT = "+34600000001"
SETTINGS = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=300)
A, B, C = -1001000000101, -1001000000102, -1001000000103


class FakeClock:
    def __init__(self) -> None:
        self.mono = 0.0
        self.slept: list[float] = []

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.mono += seconds


Msgs = list[dict[str, Any]]


def messages(first: int, last: int, step: timedelta, top: int | None = None) -> Msgs:
    """Server ids first..last; the message with id `top` (default `last`) is dated NOW."""
    top = last if top is None else top
    return [
        {
            "id": i,
            "date": (NOW - (top - i) * step).isoformat(),
            "text": f"message {i}",
            "sender_id": 7,
            "sender_name": "Ana",
        }
        for i in range(first, last + 1)
    ]


def chat(chat_id: int, msgs: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {"chat_id": chat_id, "title": f"chat {chat_id}", "kind": "supergroup",
            "messages": msgs, **extra}  # fmt: skip


def enable(conn: psycopg.Connection[tuple[object, ...]], td: FakeTdClient, *ids: int) -> None:
    for chat_id in ids:
        chats.enable(conn, td.get_chat(chat_id), ACCOUNT)


def run(
    conn: psycopg.Connection[tuple[object, ...]],
    td: FakeTdClient,
    clock: FakeClock | None = None,
    max_minutes: float = 150,
) -> ingest.SyncResult:
    return ingest.sync(conn, lambda _: td, SETTINGS, clock or FakeClock(), max_minutes)


def stored(conn: psycopg.Connection[tuple[object, ...]], chat_id: int) -> list[int]:
    rows = conn.execute(
        "SELECT server_message_id FROM message WHERE chat_id = %s ORDER BY 1", (chat_id,)
    )
    return [r[0] for r in rows]


def chat_row(conn: psycopg.Connection[tuple[object, ...]], chat_id: int) -> tuple[Any, ...]:
    row = conn.execute(
        "SELECT newest_message_id >> 20, oldest_message_id >> 20, backfill_done, last_sync_at"
        " FROM chat WHERE chat_id = %s",
        (chat_id,),
    ).fetchone()
    assert row is not None
    return row


def dirty_from(conn: psycopg.Connection[tuple[object, ...]], chat_id: int) -> int | None:
    row = conn.execute(
        "SELECT windows_dirty_from >> 20 FROM chat WHERE chat_id = %s", (chat_id,)
    ).fetchone()
    assert row is not None
    return cast(int | None, row[0])


def before_sweep(td: FakeTdClient) -> list[int]:
    """`from_message_id`s of one chat's calls before its sweep, which restarts from 0 last."""
    froms = [f for _, f, _ in td.history_calls]
    return froms[: len(froms) - froms[::-1].index(0) - 1]


UNEVEN = [100, 37, 1, 64, 100, 9] * 100


@pytest.mark.parametrize("inclusive", [False, True], ids=["exclusive", "inclusive"])
def test_a2_backfill_three_chats_with_crash_and_rerun(
    conn: psycopg.Connection[tuple[object, ...]], inclusive: bool
) -> None:
    # C spans ~5 years, so only its last 3 years are inside the lookback.
    old_step = timedelta(hours=18)
    td = FakeTdClient({
        "from_inclusive": inclusive,
        "chats": [
            chat(A, messages(1, 2500, timedelta(minutes=10)), page_sizes=UNEVEN),
            chat(B, messages(1, 2500, timedelta(minutes=7)), page_sizes=UNEVEN[3:]),
            chat(C, messages(1, 2500, old_step), page_sizes=UNEVEN[1:]),
        ],
        "faults": [{"call": 41, "error": "killed"}],
        "runs": [{}],
    })  # fmt: skip
    enable(conn, td, A, B, C)

    with pytest.raises(TdError):
        run(conn, td)
    status = conn.execute("SELECT status, error FROM sync_run").fetchone()
    assert status == ("error", "TdError 500")
    partial = sum(len(stored(conn, c)) for c in (A, B, C))
    assert 0 < partial < 7000

    td.next_run()
    assert run(conn, td).status == "ok"

    cutoff = NOW.replace(year=2023)
    in_lookback = [i for i in range(1, 2501) if NOW - (2500 - i) * old_step >= cutoff]
    assert 0 < len(in_lookback) < 2500
    assert stored(conn, A) == list(range(1, 2501))
    assert stored(conn, B) == list(range(1, 2501))
    assert stored(conn, C) == in_lookback
    total = conn.execute("SELECT count(*), count(DISTINCT (chat_id, message_id)) FROM message")
    assert total.fetchone() == (5000 + len(in_lookback),) * 2
    for chat_id in (A, B, C):
        newest, oldest, done, last_sync = chat_row(conn, chat_id)
        assert (newest, done) == (2500, True)
        assert last_sync is not None
    assert chat_row(conn, A)[1] == 1
    assert chat_row(conn, C)[1] < in_lookback[0]
    row = conn.execute("SELECT min(date) FROM message WHERE chat_id = %s", (C,)).fetchone()
    assert row is not None and row[0] >= cutoff


def test_inclusive_one_message_pages_do_not_loop(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({
        "from_inclusive": True,
        "chats": [chat(A, messages(1, 30, timedelta(minutes=1)), page_sizes=[1] * 500)],
    })  # fmt: skip
    enable(conn, td, A)
    result = run(conn, td)
    assert result.status == "ok"
    assert stored(conn, A) == list(range(1, 31))
    assert chat_row(conn, A)[:3] == (30, 1, True)
    # Each page is the boundary alone, then one message below it: about two calls each.
    assert len(before_sweep(td)) <= 2 * 30 + 2


def test_first_run_takes_one_page_then_backfills(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 250, timedelta(minutes=1)))]})
    enable(conn, td, A)
    run(conn, td)
    assert [f >> 20 for f in before_sweep(td)] == [0, 151, 51, 1]
    assert stored(conn, A) == list(range(1, 251))
    assert chat_row(conn, A)[:3] == (250, 1, True)


def test_per_chat_years_override(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 400, timedelta(days=3)))]})
    chats.enable(conn, td.get_chat(A), ACCOUNT, years=1)
    run(conn, td)
    ids = stored(conn, A)
    assert ids[-1] == 400
    assert len(ids) == len([i for i in range(1, 401) if (400 - i) * 3 <= 365])
    # Re-enabling without --years keeps the stored override.
    chats.enable(conn, td.get_chat(A), ACCOUNT)
    row = conn.execute("SELECT backfill_years FROM chat WHERE chat_id = %s", (A,)).fetchone()
    assert row == (1,)
    chats.enable(conn, td.get_chat(A), ACCOUNT, years=2)
    row = conn.execute("SELECT backfill_years FROM chat WHERE chat_id = %s", (A,)).fetchone()
    assert row == (2,)


def synced_chat(
    conn: psycopg.Connection[tuple[object, ...]], count: int, runs: list[dict[str, Any]]
) -> FakeTdClient:
    msgs = messages(1, count, timedelta(minutes=5), top=600)
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": runs})
    enable(conn, td, A)
    assert run(conn, td).status == "ok"
    assert stored(conn, A) == list(range(1, count + 1))
    td.history_calls.clear()
    td.next_run()
    return td


def added(chat_id: int, first: int, last: int) -> list[dict[str, Any]]:
    return [{"chat_id": chat_id, **m} for m in messages(first, last, timedelta(minutes=5), top=600)]


def test_a3_deleted_checkpoint_stops_one_page_past_it(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    deleted = [{"chat_id": A, "id": 300}]
    td = synced_chat(conn, 300, [{"delete": deleted, "add": added(A, 301, 350)}])
    assert run(conn, td).status == "ok"
    assert before_sweep(td) == [0]
    assert stored(conn, A) == list(range(1, 351))
    assert chat_row(conn, A)[0] == 350


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
def test_a3_interrupted_catch_up_keeps_checkpoint_and_leaves_no_gap(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = synced_chat(
        conn, 100, [{"add": added(A, 101, 350), "faults": [{"call": 2, "error": "killed"}]}, {}]
    )
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")  # windows current (T7)
    with pytest.raises(TdError):
        run(conn, td)
    assert chat_row(conn, A)[0] == 100
    assert stored(conn, A) == list(range(1, 101)) + list(range(251, 351))
    assert dirty_from(conn, A) == 251  # the committed page is marked despite the crash
    td.next_run()
    assert run(conn, td).status == "ok"
    assert stored(conn, A) == list(range(1, 351))
    assert chat_row(conn, A)[0] == 350
    assert dirty_from(conn, A) == 101


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
def test_backfill_lowers_the_window_marker(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 250, timedelta(minutes=1)))]})
    enable(conn, td, A)
    assert dirty_from(conn, A) is None
    assert run(conn, td, max_minutes=0).status == "time_limited"
    assert dirty_from(conn, A) is None  # nothing written, nothing marked
    assert run(conn, td).status == "ok"
    assert dirty_from(conn, A) == 1


def test_skipped_ids_are_logged_by_id_only(
    conn: psycopg.Connection[tuple[object, ...]], caplog: pytest.LogCaptureFixture
) -> None:
    secret = "SECRET-PAYLOAD продление"  # noqa: S105 - message text, not a credential
    msgs = messages(1, 5, timedelta(minutes=1))
    for m in msgs:
        m["text"] = f"{secret} {m['id']}"
    local = {"date": NOW.isoformat(), "text": secret}
    msgs += [{**local, "message_id": (3 << 20) + 1}, {**local, "message_id": 0}]
    td = FakeTdClient({"chats": [chat(A, msgs)]})
    enable(conn, td, A)
    caplog.set_level(logging.DEBUG)
    result = run(conn, td)
    assert result.status == "ok"
    assert result.stats["skipped"] >= 2  # backfill and sweep see them again
    assert stored(conn, A) == [1, 2, 3, 4, 5]
    assert f"message_id={(3 << 20) + 1}" in caplog.text
    assert "message_id=0" in caplog.text
    for forbidden in ("SECRET", "продлен", ACCOUNT, ACCOUNT.lstrip("+")):
        assert forbidden not in caplog.text


def test_a4_two_flood_waits_sleep_and_complete(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({
        "chats": [chat(A, messages(1, 150, timedelta(minutes=1)))],
        "faults": [{"call": 2, "flood_wait": 2}, {"call": 3, "flood_wait": 2}],
    })  # fmt: skip
    enable(conn, td, A)
    clock = FakeClock()
    result = run(conn, td, clock)
    assert result.status == "ok"
    assert result.stats["flood_waits"] == 2
    assert clock.slept.count(2) == 2
    assert all(s <= 0.3 for s in clock.slept if s != 2)
    assert stored(conn, A) == list(range(1, 151))


def test_a4_three_consecutive_flood_waits_end_the_run(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({
        "chats": [chat(A, messages(1, 10, timedelta(minutes=1))),
                  chat(B, messages(1, 10, timedelta(minutes=1)))],
        "faults": [{"call": n, "flood_wait": 5} for n in (1, 2, 3)],
    })  # fmt: skip
    enable(conn, td, A, B)
    clock = FakeClock()
    result = run(conn, td, clock)
    assert result.status == "flood_limited"
    assert clock.slept == [5, 5]
    assert conn.execute("SELECT status FROM sync_run").fetchone() == ("flood_limited",)
    assert stored(conn, A) == stored(conn, B) == []
    # Never-synced chats go by chat_id, so B (-…102) goes first and is cut off.
    assert chat_row(conn, B)[3] is not None
    assert chat_row(conn, A) == (None, None, False, None)  # A untouched


def test_flood_wait_past_the_deadline_ends_the_run_without_sleeping(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({
        "chats": [chat(A, messages(1, 150, timedelta(minutes=1)))],
        "faults": [{"call": 2, "flood_wait": 3600}],
    })  # fmt: skip
    enable(conn, td, A)
    clock = FakeClock()
    result = run(conn, td, clock, max_minutes=1)
    assert result.status == "time_limited"
    assert 3600 not in clock.slept
    assert clock.mono < 60
    assert stored(conn, A) == list(range(51, 151))  # the first page is kept


def test_request_spacing_past_the_deadline_ends_the_run_without_sleeping(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 150, timedelta(minutes=1)))]})
    enable(conn, td, A)
    clock = FakeClock()
    slow = replace(SETTINGS, page_delay_ms=120_000)  # 2 min between pages, 1 min budget
    result = ingest.sync(conn, lambda _: td, slow, clock, max_minutes=1)
    assert (result.status, result.stats["requests"]) == ("time_limited", 1)
    assert clock.slept == []


def test_a4_stuck_chat_completes_without_starving_others(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    cut = [{"call": n, "flood_wait": 1} for n in (5, 6, 7)]  # 4 page requests per run
    first_run = {"add": added(A, 101, 1100) + added(B, 101, 150), "faults": cut}
    td = FakeTdClient({
        "chats": [chat(A, messages(1, 100, timedelta(minutes=1), top=600)),
                  chat(B, messages(1, 100, timedelta(minutes=1), top=600))],
        "runs": [first_run] + [{"faults": cut}] * 6,
    })  # fmt: skip
    enable(conn, td, A, B)
    assert run(conn, td).status == "ok"
    conn.execute("UPDATE chat SET last_sync_at = '2026-01-01' WHERE chat_id = %s", (A,))
    conn.execute("UPDATE chat SET last_sync_at = '2026-01-02' WHERE chat_id = %s", (B,))

    statuses = []
    for n in range(1, 7):
        td.next_run()
        statuses.append(run(conn, td).status)
        if n == 1:
            assert chat_row(conn, A)[0] == 100  # cut off: checkpoint unchanged
            assert chat_row(conn, A)[3] > chat_row(conn, B)[3]  # A moved behind B
        if n == 2:
            assert stored(conn, B) == list(range(1, 151))
            assert chat_row(conn, B)[0] == 150
        if chat_row(conn, A)[0] == 1100:
            break
    assert chat_row(conn, A)[0] == 1100, statuses
    assert stored(conn, A) == list(range(1, 1101))
    assert statuses[0] == "flood_limited"


def test_sync_lock_and_time_budget(
    conn: psycopg.Connection[tuple[object, ...]], db_url: str
) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 10, timedelta(minutes=1)))]})
    enable(conn, td, A)
    assert run(conn, td, max_minutes=0).status == "time_limited"
    assert stored(conn, A) == []

    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        with pytest.raises(ingest.SyncLocked):
            run(conn, td)
    assert run(conn, td).status == "ok"


def test_stale_running_rows_become_interrupted(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    conn.execute("INSERT INTO sync_run (status) VALUES ('running'), ('ok')")
    result = run(conn, FakeTdClient({"chats": []}))
    rows = conn.execute(
        "SELECT run_id, status, finished_at IS NOT NULL FROM sync_run ORDER BY run_id"
    ).fetchall()
    assert [r[1:] for r in rows] == [("interrupted", True), ("ok", False), ("ok", True)]
    assert rows[2][0] == result.run_id


def cli_sync(db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
             scenario: dict[str, Any], *args: str) -> Any:  # fmt: skip
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    path = tmp_path / "s.json"
    path.write_text(json.dumps(scenario))
    return CliRunner().invoke(app, ["sync", "--scenario", str(path), *args])


def test_cli_partial_runs_exit_0_and_a_held_lock_exits_2(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    msgs = messages(1, 10, timedelta(minutes=1))
    enable(conn, FakeTdClient({"chats": [chat(A, msgs)]}), A)
    flood = {"chats": [chat(A, msgs)],
             "faults": [{"call": n, "flood_wait": 0} for n in (1, 2, 3)]}  # fmt: skip
    result = cli_sync(db_url, tmp_path, monkeypatch, flood)
    assert (result.exit_code, "flood_limited" in result.stdout) == (0, True), result.output
    result = cli_sync(db_url, tmp_path, monkeypatch, {"chats": [chat(A, msgs)]},
                      "--max-minutes", "0")  # fmt: skip
    assert (result.exit_code, "time_limited" in result.stdout) == (0, True), result.output

    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        locked = cli_sync(db_url, tmp_path, monkeypatch, {"chats": [chat(A, msgs)]})
    assert locked.exit_code == 2
    assert "another tgi sync is running" in locked.stderr
    assert conn.execute("SELECT count(*) FROM sync_run").fetchone() == (2,)


def test_cli_usage_errors_exit_1() -> None:
    """Click's usage errors exit 2, the lock's code; `tgi` maps them to 1 (SPEC §5.3 step 7)."""
    runner = CliRunner()
    # sync takes negative chat ids, so its unknown options are read as a bad CHAT_ID.
    assert runner.invoke(app, ["sync", "--bogus"]).exit_code == 1
    bogus = runner.invoke(app, ["db", "migrate", "--bogus"])
    assert bogus.exit_code == 1
    assert "No such option" in bogus.stderr
    assert runner.invoke(app, ["chats", "enable"]).exit_code == 1  # missing argument
    assert runner.invoke(app, ["sync", "--max-minutes", "-1"]).exit_code == 1  # bad value


def test_enable_refuses_per_account_chat_of_another_account(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    private = Chat(chat_id=4200, title="Ana", kind="private")
    chats.enable(conn, private, ACCOUNT)
    with pytest.raises(chats.ChatConflict):
        chats.enable(conn, private, "+34600000002")
    assert chats.disable(conn, 4200)
    with pytest.raises(chats.ChatConflict):
        chats.enable(conn, private, "+34600000002")
    chats.enable(conn, private, ACCOUNT)  # same account: fine
    supergroup = Chat(chat_id=A, title="vlc", kind="supergroup")
    chats.enable(conn, supergroup, ACCOUNT, 4)
    holder = "SELECT account_phone, enabled, backfill_years FROM chat WHERE chat_id = %s"
    with pytest.raises(chats.ChatConflict, match="already enabled under another account"):
        chats.enable(conn, supergroup, "+34600000002", 1)  # enabled there: refused
    with pytest.raises(chats.ChatConflict):
        chats.check_account(conn, A, "+34600000002")
    chats.check_account(conn, A, ACCOUNT)
    chats.check_account(conn, -1009999999999, "+34600000002")  # unknown chat: free
    assert conn.execute(holder, (A,)).fetchone() == (ACCOUNT, True, 4)
    assert chats.disable(conn, A)
    chats.check_account(conn, A, "+34600000002")
    chats.enable(conn, supergroup, "+34600000002")  # -100… ids are global: may move
    assert conn.execute(holder, (A,)).fetchone() == ("+34600000002", True, 4)


def test_first_enables_from_two_accounts_keep_the_first_owner(db_url: str) -> None:
    """#43: two first enables of an absent chat, each from another account (CLI and CLI,
    or CLI and MCP: both are `chats.enable`). The second waits for the first, is refused
    and the first owner stays."""
    supergroup = Chat(chat_id=A, title="vlc", kind="supergroup")
    other: list[BaseException | None] = []

    def second() -> None:
        with psycopg.connect(db_url, autocommit=True) as c2:
            try:
                chats.enable(c2, supergroup, "+34600000002")
                other.append(None)
            except chats.ChatConflict as exc:
                other.append(exc)

    with psycopg.connect(db_url, autocommit=True) as c1:
        with c1.transaction():  # the first enable, not yet committed
            chats.enable(c1, supergroup, ACCOUNT)
            thread = threading.Thread(target=second)
            thread.start()
            thread.join(1.0)
            assert thread.is_alive()  # it waits for the first instead of reading "absent"
        thread.join(10)
        assert not thread.is_alive()
        assert isinstance(other[0], chats.ChatConflict)
        row = c1.execute("SELECT account_phone, enabled FROM chat WHERE chat_id = %s", (A,))
        assert row.fetchone() == (ACCOUNT, True)


def test_cli_enable_refuses_a_chat_enabled_under_another_account(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Antony, 2026-10-09: `account_phone` is never overwritten; nothing in the row changes."""
    chats.enable(conn, Chat(chat_id=A, title="vlc", kind="supergroup"), "+34600000002", 5)
    row = "SELECT account_phone, enabled, backfill_years, title FROM chat WHERE chat_id = %s"
    before = conn.execute(row, (A,)).fetchone()
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"chats": [chat(A, messages(1, 10, timedelta(days=1)))]}))
    monkeypatch.setenv("DATABASE_URL", db_url)
    args = ["chats", "enable", str(A), "--account", ACCOUNT, "--years", "2",
            "--scenario", str(path)]  # fmt: skip
    result = CliRunner().invoke(app, args, terminal_width=200)
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert f"chat {A} is already enabled under another account" in result.stderr
    assert conn.execute(row, (A,)).fetchone() == before == ("+34600000002", True, 5, "vlc")


def test_cli_sync_error_hides_the_tdlib_message(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "MARKER-7f3a продление"
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({
        "chats": [chat(A, messages(1, 120, timedelta(minutes=1)))],
        "faults": [{"call": 1, "error": marker}],
    }))  # fmt: skip
    enable(conn, FakeTdClient.from_file(scenario), A)
    result = CliRunner().invoke(app, ["sync", "--scenario", str(scenario)])
    assert result.exit_code == 3
    assert "sync failed: TdError 500" in result.stderr
    assert "MARKER" not in result.stdout + result.stderr
    assert conn.execute("SELECT status, error FROM sync_run").fetchone() == ("error", "TdError 500")


def test_cli_chats_and_sync(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({"chats": [
        chat(A, messages(1, 120, timedelta(minutes=1))),
        {"chat_id": 4200, "title": "Ana", "kind": "private", "messages": []},
    ]}))  # fmt: skip
    cli = CliRunner()
    s = ["--scenario", str(scenario)]

    def ok(*args: str, **kw: Any) -> str:
        result = cli.invoke(app, list(args), **kw)
        assert result.exit_code == 0, result.output
        return result.output

    ok("chats", "enable", str(A), "--account", ACCOUNT, *s)
    listed = ok("chats", "list", "--account", ACCOUNT, *s)
    assert f"* {A}" in listed and "  4200" in listed
    # Another account in the same supergroup does not see A's chat as enabled for it.
    other = ok("chats", "list", "--account", "+34600000002", *s)
    assert f"  {A}" in other and f"* {A}" not in other
    assert "ok" in ok("sync", *s)
    assert stored(conn, A) == list(range(1, 121))
    # No scenario: the real TDLib adapter, which needs DATA_DIR (tests/test_td.py covers it).
    monkeypatch.delenv("DATA_DIR", raising=False)
    unwired = cli.invoke(app, ["sync"])
    assert unwired.exit_code == 1 and "DATA_DIR is not set" in unwired.output

    ok("chats", "disable", str(A))
    assert chats.enabled_ids(conn, ACCOUNT) == set()

    aborted = cli.invoke(app, ["chats", "forget", str(A)], input="n\n")
    assert aborted.exit_code == 1
    assert len(stored(conn, A)) == 120
    ok("chats", "forget", str(A), input="y\n")
    assert stored(conn, A) == []
    assert conn.execute("SELECT count(*) FROM chat").fetchone() == (0,)


# --- deepening a finished backfill (SPEC §5.3 step 2, A2) ------------------------------------

YEAR_1, YEAR_3 = NOW.replace(year=2025), NOW.replace(year=2023)  # the 1- and 3-year cutoffs


class Recorder(embed.FakeEmbedder):
    """The fake provider, recording every text it embeds."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.texts += texts
        return super().embed_documents(texts)


class LaterClock(FakeClock):
    def __init__(self, now: datetime) -> None:
        super().__init__()
        self._now = now

    def now(self) -> datetime:
        return self._now


def deep_run(
    conn: psycopg.Connection[tuple[object, ...]],
    td: FakeTdClient,
    clock: FakeClock | None = None,
    embedder: embed.Embedder | None = None,
) -> ingest.SyncResult:
    td.history_calls.clear()
    settings = replace(SETTINGS, page_delay_ms=0)
    result = ingest.sync(conn, lambda _: td, settings, clock or FakeClock(), embedder=embedder)
    assert result.status == "ok"
    return result


def deepen(
    conn: psycopg.Connection[tuple[object, ...]], td: FakeTdClient, years: int, **kw: Any
) -> chats.Depth:
    return chats.enable(conn, td.get_chat(A), ACCOUNT, years, default_years=3, now=NOW, **kw)


def finished(conn: psycopg.Connection[tuple[object, ...]], chat_id: int) -> tuple[Any, ...]:
    row = conn.execute(
        "SELECT backfill_done, backfill_cutoff FROM chat WHERE chat_id = %s", (chat_id,)
    ).fetchone()
    assert row is not None
    return row


def at(moment: datetime, ids: range, **extra: Any) -> Msgs:
    """Messages `ids`, one minute apart from `moment`; `extra` maps an id to more fields."""
    return [{"id": i, "date": (moment + timedelta(minutes=n)).isoformat(), "text": f"message {i}",
             "sender_name": "Ana", **extra.get(str(i), {})} for n, i in enumerate(ids)]  # fmt: skip


def window_rows(conn: psycopg.Connection[tuple[object, ...]]) -> dict[int, tuple[Any, ...]]:
    """first server id -> (window_id, text_hash, embedded_at)."""
    rows = conn.execute(
        'SELECT w.first_message_id, w.window_id, w.text_hash, e.embedded_at FROM "window" w'
        " LEFT JOIN window_embedding e USING (window_id) WHERE w.chat_id = %s",
        (A,),
    )
    return {r[0] >> 20: tuple(r[1:]) for r in rows}


def froms(td: FakeTdClient) -> list[int]:
    return [f >> 20 for f in before_sweep(td)]


def test_deepened_backfill_continues_from_the_oldest_stored_message(
    conn: psycopg.Connection[tuple[object, ...]], caplog: pytest.LogCaptureFixture
) -> None:
    # (a) Inside 3 years only: 1-5 (2024) and 6-7 (just below the 1-year cutoff, contiguous
    # with 8-10 just above it). Then 11-13; 14-16 where 15 replies to 2; 17-19.
    msgs = (
        at(datetime(2024, 6, 1, tzinfo=UTC), range(1, 6))
        + at(YEAR_1 - timedelta(minutes=2), range(6, 11))
        + at(datetime(2026, 1, 1, tzinfo=UTC), range(11, 14))
        + at(datetime(2026, 3, 1, tzinfo=UTC), range(14, 17), **{"15": {"reply_to": 2}})
        + at(datetime(2026, 5, 1, tzinfo=UTC), range(17, 20))
    )
    td = FakeTdClient({"chats": [chat(A, msgs)]})
    assert deepen(conn, td, 1) == chats.Depth(1, False)
    deep_run(conn, td)
    assert stored(conn, A) == list(range(8, 20))
    assert finished(conn, A) == (True, YEAR_1)
    # The last page's tail below the cutoff was not stored: the cursor sits below the archive.
    assert chat_row(conn, A)[1] == 1
    before = window_rows(conn)
    assert sorted(before) == [8, 11, 14, 17]

    assert deepen(conn, td, 3) == chats.Depth(3, True)
    caplog.set_level(logging.INFO)
    recorder = Recorder()
    result = deep_run(conn, td, embedder=recorder)
    assert froms(td) == [0, 8, 1]  # catch-up, then down from the oldest stored message
    assert stored(conn, A) == list(range(1, 20))
    assert result.stats["upserted"] == 7  # 1-7, each stored once
    assert finished(conn, A) == (True, YEAR_3)
    assert f"backfill reopened chat_id={A}" in caplog.text
    after = window_rows(conn)
    assert sorted(after) == [1, 6, 11, 14, 17]  # 6-10 replaced the window 8-10
    assert after[11] == before[11] and after[17] == before[17]  # kept with their embeddings
    assert after[14][0] == before[14][0] and after[14][1] != before[14][1]  # the quote renders
    assert after[14][2] != before[14][2]
    embedded = sorted(t.split("\n")[1] for t in recorder.texts)
    assert embedded == ["Ana: message 1", "Ana: message 14", "Ana: message 6"]
    assert any("↪ Ana: message 2" in t for t in recorder.texts)


def test_same_depth_later_clock_fetches_nothing(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    # (b) and the "time has passed" row: a deeper value whose cutoff is still later.
    td = FakeTdClient({"chats": [chat(A, messages(1, 400, timedelta(days=3)))]})
    deepen(conn, td, 1)
    deep_run(conn, td)
    ids = stored(conn, A)
    later = NOW + timedelta(days=400)
    deep_run(conn, td, LaterClock(later))
    assert froms(td) == [0]
    assert finished(conn, A) == (True, YEAR_1)
    assert chats.enable(conn, td.get_chat(A), ACCOUNT, 2, now=later).reopens is False
    deep_run(conn, td, LaterClock(later))
    assert froms(td) == [0]
    assert stored(conn, A) == ids


def test_unknown_finish_depth_reopens_once(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    # (c) A chat finished before migration 0003: one page below the archive, then known.
    td = FakeTdClient({"chats": [chat(A, messages(1, 400, timedelta(days=3)))]})
    deepen(conn, td, 1)
    deep_run(conn, td)
    ids = stored(conn, A)
    conn.execute("UPDATE chat SET backfill_cutoff = NULL")
    assert deepen(conn, td, None).reopens is True
    deep_run(conn, td)
    assert froms(td) == [0, ids[0]]
    assert stored(conn, A) == ids
    assert finished(conn, A) == (True, YEAR_1)
    deep_run(conn, td)
    assert froms(td) == [0]


def test_smaller_depth_fetches_and_deletes_nothing(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    # (d)
    td = FakeTdClient({"chats": [chat(A, messages(1, 400, timedelta(days=3)))]})
    deepen(conn, td, 3)
    deep_run(conn, td)
    ids, windows_before = stored(conn, A), window_rows(conn)
    assert deepen(conn, td, 1) == chats.Depth(1, False)
    deep_run(conn, td)
    assert froms(td) == [0]
    assert stored(conn, A) == ids and window_rows(conn) == windows_before
    assert finished(conn, A) == (True, YEAR_3)
    assert deepen(conn, td, 3) == chats.Depth(3, False)  # raising it back refetches nothing


def test_chat_younger_than_the_lookback_costs_one_empty_page(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    # (e)
    td = FakeTdClient({"chats": [chat(A, messages(1, 50, timedelta(hours=1)))]})
    deepen(conn, td, 1)
    deep_run(conn, td)
    assert deepen(conn, td, 3).reopens is True
    deep_run(conn, td)
    assert froms(td) == [0, 1]
    assert stored(conn, A) == list(range(1, 51))
    assert finished(conn, A) == (True, YEAR_3)


def old_first_page() -> Msgs:
    """1-20 older than 3 years, 21-150 between the 1- and 3-year cutoffs: one page of 100."""
    return at(YEAR_3 - timedelta(days=1), range(1, 21)) + at(
        datetime(2024, 6, 1, tzinfo=UTC), range(21, 151)
    )


def test_nothing_stored_restarts_from_the_top(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    # (f) DR-6: the first page was entirely older than the cutoff.
    td = FakeTdClient({"chats": [chat(A, old_first_page())]})
    deepen(conn, td, 1)
    deep_run(conn, td)
    assert stored(conn, A) == [] and finished(conn, A) == (True, YEAR_1)
    assert chat_row(conn, A)[:2] == (150, 1)
    assert deepen(conn, td, 3).reopens is True
    result = deep_run(conn, td)
    assert froms(td) == [0, 0, 51]  # catch-up, then the backfill from the top
    assert stored(conn, A) == list(range(21, 151))
    assert result.stats["upserted"] == 130
    assert finished(conn, A) == (True, YEAR_3)


def test_nothing_stored_then_caught_up_resumes_below_the_new_messages(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    # (g) like (f), but a later catch-up stored newer messages before the depth grew.
    new = [{"chat_id": A, **m} for m in at(NOW - timedelta(hours=1), range(151, 156))]
    td = FakeTdClient({"chats": [chat(A, old_first_page())], "runs": [{"add": new}]})
    deepen(conn, td, 1)
    deep_run(conn, td)
    td.next_run()
    deep_run(conn, td)
    assert stored(conn, A) == list(range(151, 156))
    assert finished(conn, A) == (True, YEAR_1)
    assert deepen(conn, td, 3).reopens is True
    result = deep_run(conn, td)
    assert froms(td) == [0, 151, 51]  # the first page (51-150) is fetched again and stored
    assert stored(conn, A) == list(range(21, 156))
    assert result.stats["upserted"] == 130
    assert finished(conn, A) == (True, YEAR_3)


def test_depth_reports_whether_the_next_sync_continues(
    conn: psycopg.Connection[tuple[object, ...]],
) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 10, timedelta(days=1)))]})
    assert deepen(conn, td, None) == chats.Depth(3, False)  # not finished
    deep_run(conn, td)
    assert deepen(conn, td, None) == chats.Depth(3, False)  # finished, not deeper
    assert deepen(conn, td, 4) == chats.Depth(4, True)  # finished and deeper
    assert chats.enable(conn, td.get_chat(A), ACCOUNT, now=NOW).years == 4  # value is kept


def test_cli_enable_says_when_the_next_sync_continues_the_backfill(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = {"chats": [chat(A, messages(1, 400, timedelta(days=3)))]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(scenario))

    def enable_cli(*years: str) -> str:
        args = ["chats", "enable", str(A), "--account", ACCOUNT, *years, "--scenario", str(path)]
        result = CliRunner().invoke(app, args)
        assert result.exit_code == 0, result.output
        return result.stdout

    monkeypatch.setenv("DATABASE_URL", db_url)
    assert enable_cli("--years", "1") == f"enabled {A}\n"
    assert cli_sync(db_url, tmp_path, monkeypatch, scenario).exit_code == 0
    assert enable_cli() == f"enabled {A}\n"  # finished, not deeper
    assert enable_cli("--years", "3") == (
        f"enabled {A}, backfill continues to 3 years on the next sync\n"
    )
    assert cli_sync(db_url, tmp_path, monkeypatch, scenario).exit_code == 0
    assert enable_cli("--years", "1") == f"enabled {A}\n"  # smaller: stored, nothing else


def test_cli_enable_refuses_out_of_range_years_before_anything(
    conn: psycopg.Connection[tuple[object, ...]],
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#38: a depth `years_before` or the int4 column cannot hold is refused before TDLib
    is opened; the row keeps its value and the next sync still runs."""
    scenario = {"chats": [chat(A, messages(1, 10, timedelta(days=1)))]}
    path = tmp_path / "s.json"
    path.write_text(json.dumps(scenario))
    monkeypatch.setenv("DATABASE_URL", db_url)

    def enable_cli(years: str) -> Any:
        args = ["chats", "enable", str(A), "--account", ACCOUNT, "--years", years,
                "--scenario", str(path)]  # fmt: skip
        return CliRunner().invoke(app, args, terminal_width=200)

    assert enable_cli("2").exit_code == 0
    with monkeypatch.context() as m:
        m.setattr(cli, "_Clients", None)  # opening TDLib now fails with a TypeError
        for years in ("0", "-1", "101", "10000", "9999999999", str(2**64)):
            result = enable_cli(years)
            assert result.exit_code == 1, (years, result.output)
            assert "years must be between 1 and 100" in result.output
    years_row = "SELECT backfill_years FROM chat WHERE chat_id = %s"
    assert conn.execute(years_row, (A,)).fetchone() == (2,)
    assert cli_sync(db_url, tmp_path, monkeypatch, scenario).exit_code == 0
    for years in ("1", "100"):  # the accepted bounds
        assert enable_cli(years).exit_code == 0
        assert conn.execute(years_row, (A,)).fetchone() == (int(years),)
        assert cli_sync(db_url, tmp_path, monkeypatch, scenario).exit_code == 0


def test_backfill_years_is_checked_like_years(conn: psycopg.Connection[tuple[object, ...]]) -> None:
    assert Settings.from_env({"BACKFILL_YEARS": "100"}).backfill_years == 100
    for raw in ("0", "101", "9999999999"):
        with pytest.raises(ValueError, match="BACKFILL_YEARS must be between 1 and 100"):
            Settings.from_env({"BACKFILL_YEARS": raw})
    td = FakeTdClient({"chats": [chat(A, [])]})
    for years, default in ((10000, 3), (None, 10000)):  # chats.enable writes nothing
        with pytest.raises(ValueError, match="must be between 1 and 100"):
            chats.enable(conn, td.get_chat(A), ACCOUNT, years, default_years=default, now=NOW)
    assert conn.execute("SELECT count(*) FROM chat").fetchone() == (0,)


@pytest.mark.parametrize(("raw", "message"), [
    ("0", "BACKFILL_YEARS must be between 1 and 100"),
    ("101", "BACKFILL_YEARS must be between 1 and 100"),
    ("-3", "BACKFILL_YEARS must not be negative"),
    ("x", "BACKFILL_YEARS must be an integer"),
])  # fmt: skip
def test_cli_bad_backfill_years_is_a_usage_error(
    db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: str, message: str
) -> None:
    """`chats enable` and `sync` exit 1 with the message, not a ValueError traceback."""
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"chats": [chat(A, [])]}))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("BACKFILL_YEARS", raw)
    for args in (["chats", "enable", str(A), "--account", ACCOUNT], ["sync"]):
        result = CliRunner().invoke(app, [*args, "--scenario", str(path)], terminal_width=200)
        assert result.exit_code == 1, (args, result.output)
        assert isinstance(result.exception, SystemExit), (args, result.exception)
        assert message in result.output, (args, result.output)
