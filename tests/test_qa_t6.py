"""QA probes for T6: truncated sweeps delete nothing; deletion scope; marker; resweep.

Contributed by QA on 7aec220 (65 cases, about 100 s); the fault grids are trimmed to keep
`make ci` fast while keeping both fault types, every page shape and the all-or-nothing
deletion check. `touched` assertions became `chat.windows_dirty_from` ones (SPEC v1.6).
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tgi import chats, ingest
from tgi.config import Settings
from tgi.td_fake import FakeTdClient
from tgi.td_port import TdError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACC = "+34600000001"
BASE = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=0)
A, B = -1001000000301, -1001000000302
IN_WINDOW_GONE = [200, 201, 240]  # 50, 49, 10 hours old
OLD_GONE = [10]  # ~10 days old: outside the 7-day sweep


class Clock:
    def __init__(self) -> None:
        self.mono = 0.0

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.mono += seconds


def msgs() -> list[dict[str, Any]]:
    return [{"id": i, "date": (NOW - (250 - i) * timedelta(hours=1)).isoformat(),
             "text": f"SECRET {i}"} for i in range(1, 251)]  # fmt: skip


def scenario(runs: list[dict[str, Any]], sizes: list[int] | None = None) -> FakeTdClient:
    cs = [{"chat_id": c, "title": "t", "kind": "supergroup", "messages": msgs(),
           "page_sizes": sizes or []} for c in (A, B)]  # fmt: skip
    return FakeTdClient({"chats": cs, "runs": runs})


def sync(conn: Any, td: FakeTdClient, on_delete: str = "tombstone", **kw: Any) -> Any:
    s = replace(BASE, on_delete=on_delete)
    return ingest.sync(conn, lambda _: td, s, kw.pop("clock", None) or Clock(), **kw)


def tomb(conn: Any, chat_id: int) -> list[int]:
    return [r[0] >> 20 for r in conn.execute(
        "SELECT message_id FROM message WHERE chat_id=%s AND deleted_at IS NOT NULL ORDER BY 1",
        (chat_id,))]  # fmt: skip


def count(conn: Any, chat_id: int) -> int:
    return conn.execute("SELECT count(*) FROM message WHERE chat_id=%s", (chat_id,)).fetchone()[0]  # type: ignore[no-any-return]


def deletes(chat_id: int) -> list[dict[str, int]]:
    return [{"chat_id": chat_id, "id": i} for i in IN_WINDOW_GONE + OLD_GONE]


UNEVEN, ONES = [37, 1, 64, 100, 9] * 20, [1] * 1000


@pytest.mark.parametrize(
    ("sizes", "fault", "k"),
    [
        (None, "error", 1),
        (None, "flood3", 2),
        (UNEVEN, "error", 3),
        (UNEVEN, "flood3", 5),
        (UNEVEN, "error", 7),
        (ONES, "flood3", 3),
    ],
    ids=["full-error-1", "full-flood-2", "uneven-error-3", "uneven-flood-5",
         "uneven-error-7", "ones-flood-3"],
)  # fmt: skip
def test_any_truncated_sweep_deletes_nothing(conn: Any, sizes: Any, fault: str, k: int) -> None:
    """Fault at each request of run 2 (catch-up, backfill and sweep alike): no deletion."""
    if fault == "error":
        faults = [{"call": k, "error": "x"}]
    else:
        faults = [{"call": n, "flood_wait": 1} for n in (k, k + 1, k + 2)]
    td = scenario([{"delete": deletes(A) + deletes(B), "faults": faults}, {}], sizes)
    for c in (A, B):
        chats.enable(conn, td.get_chat(c), ACC)
    assert sync(conn, td).status == "ok"
    td.next_run()
    try:
        status = sync(conn, td).status
    except TdError:
        status = "error"
    # A chat's sweep is all-or-nothing: never a partial or out-of-range tombstone set.
    for c in (A, B):
        assert tomb(conn, c) in ([], IN_WINDOW_GONE)
    if status != "ok":
        assert tomb(conn, A) == [] or tomb(conn, B) == []  # the cut-off chat deleted nothing
    td.next_run()
    assert sync(conn, td).status == "ok"
    for c in (A, B):
        assert tomb(conn, c) == IN_WINDOW_GONE  # the old deletion is a documented limit
        assert count(conn, c) == 250


def test_time_budget_inside_sweep_deletes_nothing(conn: Any) -> None:
    td = scenario([{"delete": deletes(A)}], [10] * 100)
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td)
    td.next_run()
    # each page is delayed 1 s; a 3-request budget cuts the run inside the sweep
    s = replace(BASE, page_delay_ms=1000)
    clock = Clock()
    res = ingest.sync(conn, lambda _: td, s, clock, max_minutes=3 / 60)
    assert res.status == "time_limited"
    assert tomb(conn, A) == []


def test_purge_scope_and_disabled_chat_untouched(conn: Any) -> None:
    td = scenario([{"delete": deletes(A) + deletes(B)}])
    for c in (A, B):
        chats.enable(conn, td.get_chat(c), ACC)
    sync(conn, td, "purge")
    chats.disable(conn, B)
    td.next_run()
    res = sync(conn, td, "purge")
    assert res.stats["purged"] == 3 and count(conn, A) == 247
    assert count(conn, B) == 250  # disabled chat is not swept
    kept = conn.execute("SELECT message_id >> 20 FROM message WHERE chat_id=%s", (A,))
    assert {r[0] for r in kept} >= set(OLD_GONE)


def dirty_from(conn: Any, chat_id: int) -> Any:
    sql = "SELECT windows_dirty_from >> 20 FROM chat WHERE chat_id=%s"
    return conn.execute(sql, (chat_id,)).fetchone()[0]


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
def test_unchanged_resweep_marks_nothing_and_edit_marks_only_it(conn: Any) -> None:
    edit = {"chat_id": A, "id": 230, "text": "new", "edit_date": NOW.isoformat()}
    td = scenario([{}, {"edit": [edit]}])
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td)
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")  # windows rebuilt (T7)
    td.next_run()
    sync(conn, td)
    assert dirty_from(conn, A) is None
    td.next_run()
    res = sync(conn, td)
    assert dirty_from(conn, A) == 230 and res.stats["edited"] == 1


def test_resweep_only_that_chat_and_unknown_chat(conn: Any) -> None:
    td = scenario([{"delete": deletes(A) + deletes(B)}])
    for c in (A, B):
        chats.enable(conn, td.get_chat(c), ACC)
    sync(conn, td)
    td.next_run()
    td.history_calls.clear()
    res = sync(conn, td, resweep=(A, 30))
    assert res.status == "ok"
    assert {c for c, _, _ in td.history_calls} == {A}
    assert tomb(conn, A) == sorted(IN_WINDOW_GONE + OLD_GONE) and tomb(conn, B) == []
    with pytest.raises(ingest.UnknownChat):
        sync(conn, td, resweep=(-5, 30))
    last = conn.execute("SELECT status FROM sync_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert last == ("error",)  # recorded, not left 'running'


def test_reappearing_message_and_logs_have_no_text(
    conn: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    gone = [{"chat_id": A, "id": 240}]
    back = [{"chat_id": A, "id": 240, "date": (NOW - timedelta(hours=10)).isoformat(),
             "text": "SECRET 240"}]  # fmt: skip
    td = scenario([{"delete": gone}, {"add": back}])
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td)
    td.next_run()
    assert sync(conn, td).stats["tombstoned"] == 1 and tomb(conn, A) == [240]
    td.next_run()
    assert sync(conn, td).stats["untombstoned"] == 1 and tomb(conn, A) == []
    assert "SECRET" not in caplog.text and ACC not in caplog.text


@pytest.mark.parametrize(("fault", "k"), [("error", 2), ("flood3", 4)])
def test_single_chat_truncated_sweep_deletes_nothing(conn: Any, fault: str, k: int) -> None:
    faults = [{"call": k, "error": "x"}] if fault == "error" else [
        {"call": n, "flood_wait": 1} for n in (k, k + 1, k + 2)]  # fmt: skip
    td = scenario([{"delete": deletes(A), "faults": faults}, {}])
    chats.enable(conn, td.get_chat(A), ACC)
    sync(conn, td)
    td.next_run()
    try:
        status = sync(conn, td).status
    except TdError:
        status = "error"
    assert (status == "ok") == (tomb(conn, A) == IN_WINDOW_GONE), (status, tomb(conn, A))
    td.next_run()
    assert sync(conn, td).status == "ok" and tomb(conn, A) == IN_WINDOW_GONE
