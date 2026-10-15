"""`tgi.eta`: when a chat added through the MCP `add_chat` tool becomes searchable."""

from datetime import UTC, datetime, timedelta

import psycopg
import pytest
from psycopg.types.json import Jsonb

from tgi import chats, eta, ingest
from tgi.config import MAX_SYNC_EVERY_MINUTES, Settings
from tgi.td_port import Chat, Message

Conn = psycopg.Connection[tuple[object, ...]]
NOW = datetime(2026, 10, 9, 19, 27, tzinfo=UTC)
NEW = -1001000000301


def page(n: int, days: float, newest_server_id: int = 100_000) -> list[Message]:
    """`n` messages, newest first, spread evenly over the last `days` days."""
    step = timedelta(days=days) / max(n - 1, 1)
    return [Message(NEW, (newest_server_id - i) << 20, NOW - i * step, "x") for i in range(n)]


def test_next_run_follows_the_schedule() -> None:
    assert eta.next_run(NOW, 60) == datetime(2026, 10, 9, 20, 0, tzinfo=UTC)
    assert eta.next_run(datetime(2026, 10, 9, 20, 0, tzinfo=UTC), 60) == datetime(
        2026, 10, 9, 21, 0, tzinfo=UTC
    )
    assert eta.next_run(NOW, 0) is None


def test_estimate_messages() -> None:
    assert eta.estimate_messages(page(100, 10), NOW, 1) == 3650  # 10 a day for 365 days
    assert eta.estimate_messages(page(100, 10, newest_server_id=900), NOW, 1) == 900  # capped
    assert eta.estimate_messages(page(100, 800), NOW, 1) == 46  # the page reaches the cutoff
    assert eta.estimate_messages([], NOW, 3) == 0


def runs(conn: Conn, fetched: int, minutes: int) -> None:
    conn.execute(
        "INSERT INTO sync_run (started_at, finished_at, status, stats) VALUES (%s, %s, 'ok', %s)",
        (NOW - timedelta(minutes=minutes), NOW, Jsonb({"fetched": fetched})),
    )


def test_estimate_shares_the_rounds_with_other_backfilling_chats(conn: Conn) -> None:
    runs(conn, 60_000, 20)  # 3000 a minute
    runs(conn, 10, 1)  # too small to measure: ignored
    for i in (1, 2):
        chats.enable(conn, Chat(-1001000000400 - i, "t", "supergroup"), "+34600000001")
    assert eta.throughput(conn) == 3000
    small = eta.estimate(conn, NEW, page(100, 10), NOW, 1, 60, 150)
    assert (small.estimated_messages, small.estimated_minutes) == (3650, 4)  # 1000 a minute
    assert small.next_run_at == datetime(2026, 10, 9, 20, 0, tzinfo=UTC)
    assert small.searchable_from == datetime(2026, 10, 9, 20, 5, tzinfo=UTC)
    assert small.complete_by == datetime(2026, 10, 9, 20, 5, tzinfo=UTC)

    big = eta.estimate(conn, NEW, page(100, 0.05, newest_server_id=10**6), NOW, 3, 60, 150)
    assert (big.estimated_messages, big.estimated_minutes) == (10**6, 1000)
    # 150-minute runs with 120 of backfill (BACKFILL_RESERVE) start 20:00, 23:00, …, 17:00,
    # then 40 minutes left in the run at 20:00
    assert big.complete_by == datetime(2026, 10, 10, 20, 40, tzinfo=UTC)
    assert big.as_dict()["complete_by"] == "2026-10-10T20:40:00+00:00"
    # hourly 50-minute runs (SYNC_MAX_MINUTES=50): 40 of backfill each, 25 runs
    hourly = eta.estimate(conn, NEW, page(100, 0.05, newest_server_id=10**6), NOW, 3, 60, 50)
    assert hourly.complete_by == datetime(2026, 10, 10, 20, 40, tzinfo=UTC)


def test_a_run_count_on_the_boundary_has_no_float_drift(conn: Conn) -> None:
    """84 minutes at 5.6 a run (SYNC_MAX_MINUTES=7) is exactly 15 runs, not 16."""
    runs(conn, 3000, 1)  # 3000 a minute
    sample = page(100, 0.05, newest_server_id=84 * 3000)
    hourly = eta.estimate(conn, NEW, sample, NOW, 3, 60, 7)
    assert hourly.estimated_minutes == 84
    # runs at 20:00 … 09:00 (14 of them) leave 5.6 minutes for the run at 10:00
    assert hourly.complete_by == datetime(2026, 10, 10, 10, 5, 36, tzinfo=UTC)


def test_no_schedule_no_times(conn: Conn) -> None:
    unknown = eta.estimate(conn, NEW, page(10, 1), NOW, 1, 0, 150)
    assert unknown.next_run_at is None and unknown.complete_by is None
    assert unknown.estimated_minutes == 2  # the default rate: 3650 / 3000


def test_unread_history_is_unknown_not_empty(conn: Conn) -> None:
    """#42: no sample is not an empty chat; the schedule still gives the first results."""
    unknown = eta.estimate(conn, NEW, None, NOW, 1, 60, 150)
    assert (unknown.estimated_messages, unknown.estimated_minutes) == (None, None)
    assert unknown.complete_by is None
    assert unknown.searchable_from == datetime(2026, 10, 9, 20, 5, tzinfo=UTC)
    empty = eta.estimate(conn, NEW, [], NOW, 1, 60, 150)
    assert (empty.estimated_messages, empty.complete_by) == (0, datetime(2026, 10, 9, 20, 5,
                                                                         tzinfo=UTC))  # fmt: skip


def test_sync_max_minutes() -> None:
    assert Settings.from_env({}).sync_max_minutes == ingest.DEFAULT_MAX_MINUTES == 150
    assert Settings.from_env({"SYNC_MAX_MINUTES": "50"}).sync_max_minutes == 50
    for bad in ("0", "-1", str(MAX_SYNC_EVERY_MINUTES + 1), "x"):
        with pytest.raises(ValueError, match="SYNC_MAX_MINUTES"):
            Settings.from_env({"SYNC_MAX_MINUTES": bad})


def test_sync_every_minutes_is_capped() -> None:
    """#41: an absurd schedule is a config error, not an overflow inside `add_chat`."""
    assert Settings.from_env({"SYNC_EVERY_MINUTES": "60"}).sync_every_minutes == 60
    assert Settings.from_env({}).sync_every_minutes == 0
    cap = str(MAX_SYNC_EVERY_MINUTES)
    assert Settings.from_env({"SYNC_EVERY_MINUTES": cap}).sync_every_minutes == 10080
    assert eta.next_run(NOW, MAX_SYNC_EVERY_MINUTES) is not None
    for bad in (str(MAX_SYNC_EVERY_MINUTES + 1), "99999999999999", "-1", "x"):
        with pytest.raises(ValueError, match="SYNC_EVERY_MINUTES"):
            Settings.from_env({"SYNC_EVERY_MINUTES": bad})
