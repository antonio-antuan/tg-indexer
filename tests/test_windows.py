"""T7: conversation windows (A5) and the window step of `tgi sync` (A3's window clause)."""

from dataclasses import replace
from datetime import timedelta
from typing import Any, cast

import psycopg
import pytest
from test_ingest import ACCOUNT, NOW, SETTINGS, A, FakeClock, chat, dirty_from, enable
from test_sweep import DELETED_IN, EDITED, a3_changes, history

from tgi import chats, ingest, windows
from tgi.config import OnDelete, Settings
from tgi.td_fake import FakeTdClient
from tgi.windows import Msg

Conn = psycopg.Connection[tuple[object, ...]]
# The window step's `stale` flags are the subject here; test_embed.py consumes them.
pytestmark = pytest.mark.usefixtures("no_embed_step")
T0 = NOW - timedelta(days=2)
SHIFT = 20


def msg(i: int, minute: float, text: str = "hi", reply_to: int | None = None) -> Msg:
    return Msg(i << SHIFT, T0 + timedelta(minutes=minute), f"u{i}", text,
               reply_to << SHIFT if reply_to else None)  # fmt: skip


def ids(groups: list[list[Msg]]) -> list[list[int]]:
    return [[m.message_id >> SHIFT for m in g] for g in groups]


# --- A5: the rules, on the pure functions ------------------------------------------------


def test_gap_over_window_gap_min_starts_a_new_window() -> None:
    msgs = [msg(1, 0), msg(2, 30), msg(3, 60.5), msg(4, 61)]
    assert ids(windows.split(msgs, SETTINGS)) == [[1, 2], [3, 4]]  # 30 min joins, 30.5 splits


def test_char_and_message_caps() -> None:
    # "u<i>: " + padding = 100 chars per line: 15 lines fit 1 500, the 16th starts a window.
    long = [msg(i, i, "x" * (97 - len(str(i)))) for i in range(1, 18)]
    assert ids(windows.split(long, SETTINGS)) == [list(range(1, 16)), [16, 17]]
    many = [msg(i, i / 10) for i in range(1, 46)]
    assert [len(g) for g in windows.split(many, SETTINGS)] == [40, 5]


def test_window_text_and_reply_prefix() -> None:
    parent = msg(1, 0, "p" * 250)
    later = [msg(2, 100, "inside"), msg(3, 101, "re outside", reply_to=1),
             msg(4, 102, "re inside", reply_to=2)]  # fmt: skip
    parents = {m.message_id: m for m in [parent, *later]}
    w = windows.render("Valencia", later, parents, SETTINGS)
    assert w.text == "\n".join([
        f"[Valencia] {T0 + timedelta(minutes=100):%Y-%m-%d %H:%M}",
        "u2: inside",
        "↪ u1: " + "p" * 200,
        "u3: re outside",
        "u4: re inside",
    ])  # fmt: skip
    assert (w.first_message_id, w.message_ids) == (2 << SHIFT, [2 << SHIFT, 3 << SHIFT, 4 << SHIFT])
    assert (w.start_date, w.end_date) == (later[0].date, later[-1].date)


def test_single_over_long_message_is_a_window_truncated_to_the_cap() -> None:
    msgs = [msg(1, 0), msg(2, 1, "y" * 2000), msg(3, 2)]
    groups = windows.split(msgs, SETTINGS)
    assert ids(groups) == [[1], [2], [3]]
    w = windows.render("t", groups[1], {}, SETTINGS)
    assert w.text.split("\n")[1] == ("u2: " + "y" * 2000)[:1500]


# --- the window step against Postgres ----------------------------------------------------


def sync(
    conn: Conn, td: FakeTdClient, on_delete: OnDelete = "tombstone", settings: Settings = SETTINGS
) -> ingest.SyncResult:
    settings = replace(settings, on_delete=on_delete, page_delay_ms=0)
    return ingest.sync(conn, lambda _: td, settings, FakeClock())


def snapshot(conn: Conn, chat_id: int = A) -> dict[int, tuple[Any, ...]]:
    """first server id → (window_id, stale, text_hash, server ids)."""
    rows = conn.execute(
        'SELECT first_message_id, window_id, stale, text_hash, message_ids FROM "window"'
        " WHERE chat_id = %s",
        (chat_id,),
    )
    return {r[0] >> SHIFT: (r[1], r[2], r[3], [i >> SHIFT for i in r[4]]) for r in rows}


def embedded(conn: Conn) -> None:
    """Stand-in for T8: every window is embedded, so none is stale."""
    conn.execute('UPDATE "window" SET stale = false')


def changed(before: dict[int, tuple[Any, ...]], after: dict[int, tuple[Any, ...]]) -> Any:
    kept = {k for k in before.keys() & after.keys() if before[k] == after[k]}
    return {
        "stale": sorted(k for k, v in after.items() if v[1]),
        "gone": sorted(before.keys() - after.keys()),
        "new": sorted(after.keys() - before.keys()),
        "kept": len(kept),
    }


def full_rebuild_agrees(conn: Conn, settings: Settings = SETTINGS) -> None:
    """A from-scratch rebuild of the chat changes nothing: the incremental one was exact."""
    before = snapshot(conn)
    conn.execute("UPDATE chat SET windows_dirty_from = -1 WHERE chat_id = %s", (A,))
    assert windows.rebuild_chat(conn, A, settings) == {"windows_rebuilt": 0, "windows_deleted": 0}
    assert snapshot(conn) == before


def three_windows() -> list[dict[str, Any]]:
    """Ids 1-3, 11-13 and 21-23: three windows an hour apart, one minute inside each."""
    return [
        {"id": i, "date": (NOW - timedelta(hours=3 - i // 10, minutes=-(i % 10))).isoformat(),
         "text": f"m{i}", "sender_name": "S", "reply_to": 1 if i == 22 else None}
        for i in (1, 2, 3, 11, 12, 13, 21, 22, 23)
    ]  # fmt: skip


def test_sync_builds_windows_once_and_a_rebuild_is_identical(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, three_windows() + [
        {"id": 30, "date": NOW.isoformat(), "text": ""}])], "runs": [{}]})  # fmt: skip
    enable(conn, td, A)
    result = sync(conn, td)
    assert (result.stats["windows_rebuilt"], result.stats["windows_deleted"]) == (3, 0)
    assert dirty_from(conn, A) is None
    first = snapshot(conn)
    assert [v[3] for _, v in sorted(first.items())] == [[1, 2, 3], [11, 12, 13], [21, 22, 23]]
    text = conn.execute('SELECT text FROM "window" WHERE first_message_id = %s', (21 << SHIFT,))
    assert text.fetchone() == (
        f"[chat {A}] {NOW - timedelta(minutes=59):%Y-%m-%d %H:%M}\nS: m21\n↪ S: m1\nS: m22\nS: m23",
    )
    full_rebuild_agrees(conn)  # same text_hash and window_id, nothing marked stale again
    conn.execute("SET TIME ZONE 'Pacific/Auckland'")
    full_rebuild_agrees(conn)  # the date in the text is UTC whatever the session zone

    td.next_run()
    assert sync(conn, td).stats["windows_rebuilt"] == 0  # unchanged sweep: no marker, no work


def test_edit_marks_only_its_window_stale_and_a_tombstone_leaves_it(conn: Conn) -> None:
    edit = {"chat_id": A, "id": 12, "text": "m12 edited", "edit_date": NOW.isoformat()}
    td = FakeTdClient({
        "chats": [chat(A, three_windows())],
        "runs": [{"edit": [edit]}, {"delete": [{"chat_id": A, "id": 13}]}],
    })  # fmt: skip
    enable(conn, td, A)
    sync(conn, td)
    embedded(conn)
    before = snapshot(conn)

    td.next_run()
    assert sync(conn, td).stats["windows_rebuilt"] == 1
    after = snapshot(conn)
    assert changed(before, after) == {"stale": [11], "gone": [], "new": [], "kept": 2}
    assert after[11][0] == before[11][0]  # same window_id: the row was updated in place
    embedded(conn)

    td.next_run()
    sync(conn, td)
    assert snapshot(conn)[11][1:] != after[11][1:] and snapshot(conn)[11][3] == [11, 12]
    full_rebuild_agrees(conn)


def test_removed_first_message_lets_the_earlier_window_absorb_the_next(conn: Conn) -> None:
    """The marker is a window's first message: the window before it is rebuilt too."""
    small = replace(SETTINGS, window_max_chars=100)
    texts = {1: "a" * 60, 2: "b" * 50, 3: "c" * 20}  # lines of 63, 53 and 23 chars
    msgs = [{"id": i, "date": (NOW - timedelta(minutes=10 - i)).isoformat(), "text": t,
             "sender_name": "S"} for i, t in texts.items()]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{"delete": [{"chat_id": A, "id": 2}]}]})
    enable(conn, td, A)
    sync(conn, td, settings=small)
    assert [v[3] for _, v in sorted(snapshot(conn).items())] == [[1], [2, 3]]
    td.next_run()
    sync(conn, td, settings=small)
    assert [v[3] for _, v in sorted(snapshot(conn).items())] == [[1, 3]]
    full_rebuild_agrees(conn, small)


def test_title_change_rebuilds_every_window_of_the_chat(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, three_windows())], "runs": [{}, {}]})
    enable(conn, td, A)
    sync(conn, td)
    embedded(conn)
    before = snapshot(conn)
    enable(conn, td, A)  # re-enabled with the same title: nothing to rebuild
    assert dirty_from(conn, A) is None

    chats.enable(conn, replace(td.get_chat(A), title="Renamed"), ACCOUNT)
    assert dirty_from(conn, A) == 1  # the first message: every window
    td.next_run()
    result = sync(conn, td)
    assert (result.stats["windows_rebuilt"], result.stats["windows_deleted"]) == (3, 0)
    after = snapshot(conn)
    assert changed(before, after) == {"stale": [1, 11, 21], "gone": [], "new": [], "kept": 0}
    assert [after[k][0] for k in (1, 11, 21)] == [before[k][0] for k in (1, 11, 21)]
    texts = conn.execute('SELECT text FROM "window" WHERE chat_id = %s', (A,)).fetchall()
    assert all(cast(str, t[0]).startswith("[Renamed] ") for t in texts)
    assert dirty_from(conn, A) is None
    full_rebuild_agrees(conn)


def a3_run(conn: Conn, on_delete: OnDelete, extra_runs: int = 0) -> tuple[FakeTdClient, Any]:
    td = FakeTdClient({"chats": [chat(A, history())], "runs": [a3_changes()] + [{}] * extra_runs})
    enable(conn, td, A)
    assert sync(conn, td, on_delete).stats["windows_rebuilt"] == 640
    embedded(conn)
    td.next_run()
    return td, snapshot(conn)


# A3's window clause: 5 edited windows and 2 new ones (641-680, 681-690) are stale, the 7
# deleted messages' single-message windows are gone, the other 628 keep id and embedding.
A3_WINDOWS = {"stale": EDITED + [641, 681], "gone": DELETED_IN, "new": [641, 681], "kept": 628}


@pytest.mark.parametrize("on_delete", ["tombstone", "purge"])
def test_a3_only_the_affected_windows_are_rebuilt(conn: Conn, on_delete: OnDelete) -> None:
    td, before = a3_run(conn, on_delete)
    result = sync(conn, td, on_delete)
    assert (result.stats["windows_rebuilt"], result.stats["windows_deleted"]) == (7, 7)
    assert changed(before, snapshot(conn)) == A3_WINDOWS
    assert dirty_from(conn, A) is None
    full_rebuild_agrees(conn)


class Killed(BaseException):
    """A kill between the sweep and the window step."""


def test_a3_kill_before_the_window_step_rebuilds_on_the_next_run(
    conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    td, before = a3_run(conn, "tombstone", extra_runs=1)

    def killed(conn: Conn, settings: Settings) -> dict[str, int]:
        raise Killed

    with monkeypatch.context() as m, pytest.raises(Killed):
        m.setattr(windows, "rebuild_dirty", killed)
        sync(conn, td)
    assert dirty_from(conn, A) == EDITED[0]  # the sweep's writes are committed and marked
    assert snapshot(conn) == before  # windows untouched

    td.next_run()  # nothing new in Telegram
    result = sync(conn, td)
    assert (result.status, result.stats["tombstoned"]) == ("ok", 0)
    assert (result.stats["windows_rebuilt"], result.stats["windows_deleted"]) == (7, 7)
    assert changed(before, snapshot(conn)) == A3_WINDOWS
    assert dirty_from(conn, A) is None
    full_rebuild_agrees(conn)
