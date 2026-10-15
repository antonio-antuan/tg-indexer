"""Sync freshness: the backfill leaves part of the budget to the sweep and the embed step,
and TDLib is closed before the sync lock is released."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_ingest import SETTINGS, A, B, C, FakeClock, chat, chat_row, enable, messages
from typer.testing import CliRunner

from tgi import ingest
from tgi.cli import app
from tgi.td_fake import FakeTdClient
from tgi.td_port import TdError

Conn = psycopg.Connection[tuple[object, ...]]
ACCOUNT = "+34600000001"
# 3000 hourly messages: about 30 history pages at 0.5 s each, more than a 12 s budget allows.
LONG = {"chats": [chat(A, messages(1, 3000, timedelta(hours=1)))], "runs": []}


class SlowTd(FakeTdClient):
    """Each history page takes `latency` seconds of the clock, as a real request does (the
    FakeClock alone only moves while the run sleeps)."""

    def __init__(self, scenario: dict[str, Any], clock: FakeClock, latency: float = 0.5) -> None:
        super().__init__(scenario)
        self.clock, self.latency = clock, latency

    def get_chat_history(self, chat_id: int, from_message_id: int, limit: int) -> Any:
        self.clock.mono += self.latency
        return super().get_chat_history(chat_id, from_message_id, limit)


def spy_sweep(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    swept: list[int] = []
    original = ingest._Run.sweep

    def sweep(self: Any, chat: Any, days: int) -> None:
        original(self, chat, days)
        swept.append(chat.chat_id)  # completed, not merely started

    monkeypatch.setattr(ingest._Run, "sweep", sweep)
    return swept


@pytest.mark.parametrize(("reserve", "sweeps"), [(ingest.BACKFILL_RESERVE, True), (0.0, False)])
def test_a_long_backfill_leaves_the_reserve_to_the_sweep_and_the_embed_step(
    conn: Conn, monkeypatch: pytest.MonkeyPatch, reserve: float, sweeps: bool
) -> None:
    monkeypatch.setattr(ingest, "BACKFILL_RESERVE", reserve)
    clock = FakeClock()
    td = SlowTd(LONG, clock)
    enable(conn, td, A)
    swept = spy_sweep(monkeypatch)
    result = ingest.sync(conn, lambda _: td, SETTINGS, clock, max_minutes=0.2)
    assert result.status == "time_limited"
    assert chat_row(conn, A)[2] is False  # the backfill goes on next run
    # with the reserve the sweep runs and the embed step makes the newest windows searchable;
    # without it (the old behaviour) the backfill took the whole budget
    assert (swept == [A]) is sweeps
    assert (result.stats["embedded"] > 0) is sweeps


def test_a_backfill_that_finishes_in_time_is_ok_as_before(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 250, timedelta(hours=1)))], "runs": []})
    enable(conn, td, A)
    result = ingest.sync(conn, lambda _: td, SETTINGS, FakeClock())
    assert result.status == "ok" and chat_row(conn, A)[2] is True


def test_a_flood_limit_in_the_backfill_still_ends_the_run(
    conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    faults = [{"call": n, "flood_wait": 1} for n in (3, 4, 5)]  # three in a row, in backfill
    td = FakeTdClient({**LONG, "faults": faults})
    enable(conn, td, A)
    swept = spy_sweep(monkeypatch)
    result = ingest.sync(conn, lambda _: td, SETTINGS, FakeClock())
    assert result.status == "flood_limited" and swept == []


def test_tdlib_is_closed_before_the_lock_is_released(conn: Conn, db_url: str) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 5, timedelta(hours=1)))], "runs": []})
    enable(conn, td, A)
    held: list[bool] = []

    def close() -> None:
        with psycopg.connect(db_url) as other:
            try:
                with ingest.sync_lock(other):
                    held.append(False)
            except ingest.SyncLocked:
                held.append(True)

    result = ingest.sync(conn, lambda _: td, SETTINGS, FakeClock(), close_clients=close)
    assert result.status == "ok" and held == [True]


def test_tdlib_is_closed_when_the_run_fails(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 5, timedelta(hours=1)))],
                       "faults": [{"call": 1, "error": "boom"}], "runs": []})  # fmt: skip
    enable(conn, td, A)
    closed: list[bool] = []
    with pytest.raises(Exception, match="boom"):
        ingest.sync(
            conn, lambda _: td, SETTINGS, FakeClock(), close_clients=lambda: closed.append(True)
        )
    assert closed == [True]


@pytest.mark.parametrize("command", [["chats", "list"], ["chats", "enable", str(A)]])
def test_chats_list_and_enable_wait_for_a_running_sync(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: list[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    scenario = tmp_path / "s.json"
    scenario.write_text(json.dumps({"chats": [chat(A, [])], "runs": []}))
    args = [*command, "--account", ACCOUNT, "--scenario", str(scenario)]
    with psycopg.connect(db_url) as other, ingest.sync_lock(other):
        busy = CliRunner().invoke(app, args)
    assert busy.exit_code == 2 and "try again" in busy.output
    free = CliRunner().invoke(app, args)
    assert free.exit_code == 0, free.output


def test_the_reserve_sweeps_every_chat(conn: Conn, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    history = messages(1, 3000, timedelta(hours=1))
    td = SlowTd({"chats": [chat(i, history) for i in (A, B, C)], "runs": []}, clock)
    enable(conn, td, A, B, C)
    swept = spy_sweep(monkeypatch)
    result = ingest.sync(conn, lambda _: td, SETTINGS, clock, max_minutes=0.5)
    assert result.status == "time_limited"
    assert sorted(swept) == sorted([A, B, C])  # the backfill stopped, each chat still swept
    assert all(chat_row(conn, i)[2] is False for i in (A, B, C))


def test_a_catch_up_past_the_backfill_cutoff_skips_the_backfill_not_the_sweep(
    conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    td = SlowTd(LONG, clock, latency=11.0)  # the first page alone passes 80% of 12 s
    enable(conn, td, A)
    swept = spy_sweep(monkeypatch)
    pages: list[int] = []
    original = ingest._Run.backfill_page
    monkeypatch.setattr(
        ingest._Run, "backfill_page", lambda self, c: (pages.append(1), original(self, c))[1]
    )
    result = ingest.sync(conn, lambda _: td, SETTINGS, clock, max_minutes=0.2)
    assert result.status == "time_limited"
    assert pages == [1]  # tried once, stopped at the cutoff before paging
    # exactly the catch-up page and one sweep page, both from the top: no backfill request
    assert [cursor for _, cursor, _ in td.history_calls] == [0, 0]
    assert swept == []  # the sweep then met the run's deadline


def test_a_flood_wait_that_fits_the_budget_is_waited_out_before_the_sweep(
    conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock()
    # the wait (500 s) passes the backfill cutoff (480 s of 600) but not the run's deadline
    td = SlowTd({**LONG, "faults": [{"call": 2, "flood_wait": 500}]}, clock, latency=0.1)
    enable(conn, td, A)
    swept = spy_sweep(monkeypatch)
    calls: list[float] = []
    original = SlowTd.get_chat_history

    def timed(self: Any, *args: Any) -> Any:
        calls.append(clock.mono)
        return original(self, *args)

    monkeypatch.setattr(SlowTd, "get_chat_history", timed)
    result = ingest.sync(conn, lambda _: td, SETTINGS, clock, max_minutes=10)
    assert swept == [A]
    flood_answered = calls[1] + 0.1  # the second request answered with the flood wait
    assert all(t >= flood_answered + 500 for t in calls[2:])  # nothing asked before it ended
    assert result.status == "time_limited"


def test_a_failing_close_records_the_run_as_error(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 5, timedelta(hours=1)))], "runs": []})
    enable(conn, td, A)

    def close() -> None:
        raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        ingest.sync(conn, lambda _: td, SETTINGS, FakeClock(), close_clients=close)
    row = conn.execute("SELECT status, stats->>'upserted' FROM sync_run").fetchone()
    assert row == ("error", "5")


def test_a_failing_close_does_not_hide_the_run_s_own_error(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, messages(1, 5, timedelta(hours=1)))],
                       "faults": [{"call": 1, "error": "boom"}], "runs": []})  # fmt: skip
    enable(conn, td, A)
    closed: list[bool] = []

    def close() -> None:
        closed.append(True)
        raise TdError(408, "TDLib did not close")

    with pytest.raises(Exception, match="boom"):
        ingest.sync(conn, lambda _: td, SETTINGS, FakeClock(), close_clients=close)
    assert closed == [True]
    row = conn.execute("SELECT status, error FROM sync_run").fetchone()
    assert row is not None and row[0] == "error" and "408" not in str(row[1])


def test_cli_sync_ends_with_an_error_when_tdlib_does_not_close(
    db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`tgi sync` closes TDLib strictly (codex review): a session that did not close is
    the run's error, not a warning before the lock is released."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "fake")
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps({"chats": [], "runs": []}))

    def stuck(self: FakeTdClient) -> None:
        raise TdError(408, "TDLib did not close")

    def sync(_: Any, clients: Any, __: Any, *, close_clients: Any, **___: Any) -> None:
        clients("+34600000001")
        close_clients()

    monkeypatch.setattr(FakeTdClient, "close", stuck)
    monkeypatch.setattr(ingest, "sync", sync)
    result = CliRunner().invoke(app, ["sync", "--scenario", str(path)])
    assert result.exit_code == 3, result.output
    assert "closing a TDLib client failed: TdError 408" in result.output
    assert "sync failed: TdError 408" in result.output


@pytest.mark.parametrize(
    ("env", "args", "budget"),
    [({}, [], 150), ({"SYNC_MAX_MINUTES": "50"}, [], 50),
     ({"SYNC_MAX_MINUTES": "50"}, ["--max-minutes", "5"], 5)],
)  # fmt: skip
def test_cli_sync_budget_defaults_to_sync_max_minutes(
    db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    env: dict[str, str], args: list[str], budget: float,
) -> None:  # fmt: skip
    """The service runs plain `tgi sync`: SYNC_MAX_MINUTES, which add_chat's estimate also
    reads, is the run's budget unless --max-minutes says otherwise."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "fake")
    monkeypatch.delenv("SYNC_MAX_MINUTES", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    budgets = []

    def sync(*_: Any, max_minutes: float, **__: Any) -> None:
        budgets.append(max_minutes)
        raise ingest.SyncLocked("held")

    monkeypatch.setattr(ingest, "sync", sync)
    assert CliRunner().invoke(app, ["sync", *args]).exit_code == 2
    assert budgets == [budget]
