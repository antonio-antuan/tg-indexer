# ruff: noqa: E501, S311  (seeded fuzz and long synthetic fixture lines)
"""QA probes for T7 (38132da): windows vs an independent reference, markers, crash and run status."""

import hashlib
import random
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tgi import chats, ingest, windows
from tgi.config import Settings
from tgi.td_fake import FakeTdClient
from tgi.td_port import Chat, TdError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACC = "+34600000001"
A = -1001000000701
SET = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=0, window_gap_min=30, window_max_chars=400,
               window_max_msgs=6)  # fmt: skip
TITLE = "Chat T"
S = 20


class Clock:
    def __init__(self) -> None:
        self.mono = 0.0

    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return self.mono

    def sleep(self, seconds: float) -> None:
        self.mono += seconds


def reference(conn: Any, s: Settings = SET) -> list[tuple[int, list[int], str, bytes]]:
    """Windows from SPEC §5.4, computed from live messages in SQL order, not via tgi.windows."""
    rows = conn.execute(
        "SELECT message_id, date, coalesce(sender_name,'?'), text, reply_to_message_id FROM message"
        " WHERE chat_id=%s AND deleted_at IS NULL AND text <> '' ORDER BY date, message_id", (A,)
    ).fetchall()  # fmt: skip
    by_id = {r[0]: r for r in rows}
    groups: list[list[Any]] = []
    chars = 0
    for r in rows:
        size = len(f"{r[2]}: {r[3]}")
        if (not groups or r[1] - groups[-1][-1][1] > timedelta(minutes=s.window_gap_min)
                or chars + size > s.window_max_chars or len(groups[-1]) >= s.window_max_msgs):  # fmt: skip
            groups.append([r])
            chars = size
        else:
            groups[-1].append(r)
            chars += size
    out = []
    for g in groups:
        ids = {r[0] for r in g}
        lines = [f"[{TITLE}] {g[0][1].astimezone(UTC):%Y-%m-%d %H:%M}"]
        for r in g:
            p = by_id.get(r[4]) if r[4] else None
            if p is None and r[4]:  # parent may be absent from live list (tombstoned/purged)
                p = None
            if p is not None and p[0] not in ids:
                lines.append(f"↪ {p[2]}: {p[3][:200]}")
            lines.append(f"{r[2]}: {r[3]}"[: s.window_max_chars] if len(g) == 1 else f"{r[2]}: {r[3]}")  # fmt: skip
        text = "\n".join(lines)
        out.append((g[0][0], [r[0] for r in g], text, hashlib.sha256(text.encode()).digest()))
    return out


def actual(conn: Any) -> list[tuple[int, list[int], str, bytes]]:
    return [(r[0], r[1], r[2], bytes(r[3])) for r in conn.execute(
        'SELECT first_message_id, message_ids, text, text_hash FROM "window" WHERE chat_id=%s'
        " ORDER BY first_message_id", (A,))]  # fmt: skip


def check(conn: Any, s: Settings = SET) -> None:
    got, want = actual(conn), reference(conn, s)
    assert got == want, (
        next((i, g, w) for i, (g, w) in enumerate(zip(got, want, strict=False)) if g != w)
        if len(got) == len(want)
        else (len(got), len(want))
    )
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)


def lens(rng: random.Random) -> str:
    return "x" * rng.choice([0, 1, 5, 20, 60, 120, 250, 390, 500, 900])


def seed_messages(conn: Any, rng: random.Random, n: int) -> None:
    t = NOW - timedelta(days=3)
    for i in range(1, n + 1):
        t += timedelta(minutes=rng.choice([0, 1, 2, 5, 10, 29, 30, 31, 45, 90]))
        reply = rng.randint(1, i - 1) << S if i > 1 and rng.random() < 0.25 else None
        conn.execute(
            "INSERT INTO message (chat_id, message_id, server_message_id, sender_name, date, text,"
            " reply_to_message_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (A, i << S, i, f"u{i % 4}", t, f"m{i} " + lens(rng), reply))  # fmt: skip


def rebuild(conn: Any, s: Settings = SET) -> dict[str, int]:
    return windows.rebuild_chat(conn, A, s)


@pytest.fixture
def chat_conn(conn: Any) -> Any:
    chats.enable(conn, Chat(chat_id=A, title=TITLE, kind="supergroup"), ACC)
    return conn


def mark(conn: Any, *ids: int) -> None:
    ingest.mark_windows_dirty(conn, A, min(ids) << S)


@pytest.mark.parametrize("seed", range(5))
def test_incremental_equals_from_scratch_under_random_edits(chat_conn: Any, seed: int) -> None:
    conn, rng = chat_conn, random.Random(seed)
    seed_messages(conn, rng, 120)
    mark(conn, 1)
    rebuild(conn)
    check(conn)
    top = 120
    for _ in range(12):
        touched: list[int] = []
        for _ in range(rng.randint(1, 4)):
            op = rng.choice(["edit", "tomb", "purge", "restore", "new", "new", "empty"])
            i = rng.randint(1, top)
            mid = i << S
            if op == "edit":
                conn.execute("UPDATE message SET text=%s WHERE chat_id=%s AND message_id=%s",
                             (f"e{i} " + lens(rng), A, mid))  # fmt: skip
            elif op == "empty":
                conn.execute("UPDATE message SET text='' WHERE chat_id=%s AND message_id=%s", (A, mid))  # fmt: skip
            elif op == "tomb":
                conn.execute("UPDATE message SET deleted_at=now() WHERE chat_id=%s AND message_id=%s", (A, mid))  # fmt: skip
            elif op == "purge":
                mark(conn, i)  # a purged id still marks its position
                conn.execute("DELETE FROM message WHERE chat_id=%s AND message_id=%s", (A, mid))
                touched.append(i)
                continue
            elif op == "restore":
                conn.execute("UPDATE message SET deleted_at=NULL WHERE chat_id=%s AND message_id=%s", (A, mid))  # fmt: skip
            else:
                top += 1
                last = conn.execute("SELECT max(date) FROM message").fetchone()[0]
                conn.execute(
                    "INSERT INTO message (chat_id, message_id, server_message_id, sender_name,"
                    " date, text) VALUES (%s,%s,%s,'u9',%s,%s)",
                    (A, top << S, top, last + timedelta(minutes=rng.choice([1, 20, 40])),
                     "n " + lens(rng)))  # fmt: skip
                i = top
            touched.append(i)
        mark(conn, *touched)
        before = {r[0]: (r[1], r[2]) for r in conn.execute(
            'SELECT first_message_id, window_id, text_hash FROM "window"')}  # fmt: skip
        rebuild(conn)
        check(conn)
        after = {r[0]: (r[1], r[2]) for r in conn.execute(
            'SELECT first_message_id, window_id, text_hash FROM "window"')}  # fmt: skip
        for first, (wid, h) in after.items():  # unchanged windows keep window_id and hash
            if first in before and before[first][1] == h:
                assert before[first][0] == wid


def test_marker_on_first_message_of_a_window_matches_full_rebuild(chat_conn: Any) -> None:
    """Item 7: shrinking the first message of window 2 lets window 1 absorb it."""
    conn = chat_conn
    t = NOW - timedelta(days=1)
    for i, size in [(1, 150), (2, 150), (3, 130)]:  # 1+2 fill window 1 (300 + prefixes)
        conn.execute("INSERT INTO message (chat_id, message_id, server_message_id, sender_name,"
                     " date, text) VALUES (%s,%s,%s,'u',%s,%s)",
                     (A, i << S, i, t + timedelta(minutes=i), "x" * size))  # fmt: skip
    mark(conn, 1)
    rebuild(conn)
    assert [w[1] for w in actual(conn)] == [[1 << S, 2 << S], [3 << S]]
    conn.execute("UPDATE message SET text='y' WHERE message_id=%s", (3 << S,))  # now fits window 1
    mark(conn, 3)  # the marker is exactly window 2's first message
    rebuild(conn)
    check(conn)
    assert [w[1] for w in actual(conn)] == [[1 << S, 2 << S, 3 << S]]
    # deleting window 2's first message is the same position
    conn.execute("UPDATE message SET text=%s WHERE message_id=%s", ("z" * 120, 3 << S))
    mark(conn, 3)
    rebuild(conn)
    check(conn)


def test_all_messages_gone_and_marker_below_every_window(chat_conn: Any) -> None:
    conn, rng = chat_conn, random.Random(1)
    seed_messages(conn, rng, 30)
    mark(conn, 1)
    rebuild(conn)
    cur = conn.execute("SELECT * FROM message WHERE message_id=%s", (2 << S,))
    cols = [d.name for d in cur.description if d.name != "tsv"]
    saved = conn.execute(
        f"SELECT {','.join(cols)} FROM message WHERE message_id=%s", (2 << S,)
    ).fetchone()
    conn.execute("DELETE FROM message WHERE message_id <= %s", (3 << S,))
    mark(conn, 1)
    rebuild(conn)
    check(conn)
    assert min(w[0] for w in actual(conn)) > 3 << S
    ph = ",".join(["%s"] * len(saved))
    conn.execute(
        f"INSERT INTO message ({','.join(cols)}) VALUES ({ph})", saved
    )  # lands below every window
    mark(conn, 2)
    rebuild(conn)
    check(conn)
    assert min(w[0] for w in actual(conn)) == 2 << S
    conn.execute("UPDATE message SET deleted_at = now()")
    mark(conn, 1)
    rebuild(conn)
    check(conn)
    assert actual(conn) == []


def test_unchanged_window_keeps_id_hash_stale_flag_and_embedding(chat_conn: Any) -> None:
    conn, rng = chat_conn, random.Random(3)
    seed_messages(conn, rng, 60)
    mark(conn, 1)
    rebuild(conn)
    conn.execute('UPDATE "window" SET stale = false')
    wid = conn.execute('SELECT window_id, first_message_id FROM "window" ORDER BY 2').fetchall()
    vec = "[" + ",".join("0.1" for _ in range(1024)) + "]"
    conn.execute("INSERT INTO window_embedding (window_id, model, dim, embedding)"
                 " SELECT window_id, 'm', 1024, %s::vector FROM \"window\"", (vec,))  # fmt: skip
    mark(conn, 1)  # rebuild everything with no data change
    counts = rebuild(conn)
    assert counts == {"windows_rebuilt": 0, "windows_deleted": 0}
    assert conn.execute('SELECT window_id, first_message_id FROM "window" ORDER BY 2').fetchall() == wid  # fmt: skip
    assert conn.execute('SELECT count(*) FROM "window" WHERE stale').fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM window_embedding").fetchone() == (len(wid),)
    # one edit: only that window turns stale
    row = conn.execute("SELECT message_id FROM message WHERE text <> '' ORDER BY date, message_id"
                       " OFFSET 30 LIMIT 1").fetchone()  # fmt: skip
    conn.execute("UPDATE message SET text = text || ' edited' WHERE message_id=%s", (row[0],))
    ingest.mark_windows_dirty(conn, A, row[0])
    rebuild(conn)
    check(conn)
    assert conn.execute('SELECT count(*) FROM "window" WHERE stale').fetchone() == (1,)


def test_text_does_not_depend_on_session_time_zone(chat_conn: Any, db_url: str) -> None:
    import psycopg

    conn, rng = chat_conn, random.Random(5)
    seed_messages(conn, rng, 20)
    mark(conn, 1)
    rebuild(conn)
    base = actual(conn)
    with psycopg.connect(db_url, autocommit=True, options="-c TimeZone=Pacific/Auckland") as c2:
        mark(c2, 1)
        windows.rebuild_chat(c2, A, SET)
    assert actual(conn) == base


def test_window_rows_are_atomic_with_the_marker(chat_conn: Any) -> None:
    """Item 5: a failure inside the rebuild leaves marker set and windows untouched."""
    conn, rng = chat_conn, random.Random(7)
    seed_messages(conn, rng, 60)
    mark(conn, 1)
    rebuild(conn)
    before = actual(conn)
    conn.execute("UPDATE message SET text = text || ' e' WHERE message_id <= %s", (50 << S,))
    mark(conn, 1)
    conn.execute(
        "CREATE FUNCTION boom() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN"
        " RAISE EXCEPTION 'boom'; END $$"
    )
    conn.execute(
        "CREATE TRIGGER t BEFORE UPDATE OF windows_dirty_from ON chat"
        " FOR EACH ROW WHEN (NEW.windows_dirty_from IS NULL) EXECUTE FUNCTION boom()"
    )
    with pytest.raises(Exception, match="boom"):
        rebuild(conn)
    assert actual(conn) == before  # nothing partially written
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (1 << S,)
    conn.execute("DROP TRIGGER t ON chat")
    rebuild(conn)
    check(conn)


def scen(runs: list[dict[str, Any]], sizes: list[int] | None = None, **kw: Any) -> FakeTdClient:
    rng = random.Random(11)
    t = NOW - timedelta(days=3)
    ms = []
    for i in range(1, 151):
        t += timedelta(minutes=rng.choice([1, 2, 5, 10, 29, 31, 45]))
        ms.append({"id": i, "date": t.isoformat(), "text": f"m{i} " + lens(rng),
                   "sender_name": f"u{i % 4}", "reply_to": rng.randint(1, i - 1) if i > 1 and rng.random() < 0.2 else None})  # fmt: skip
    return FakeTdClient({"chats": [{"chat_id": A, "title": TITLE, "kind": "supergroup",
                                    "messages": ms, "page_sizes": sizes or []}],
                         "runs": runs, **kw})  # fmt: skip


def sync(conn: Any, td: FakeTdClient, mode: str = "tombstone", **kw: Any) -> Any:
    return ingest.sync(conn, lambda _: td, replace(SET, on_delete=mode), Clock(), **kw)  # type: ignore[arg-type]


def changes(
    rng: random.Random, top: int, dels: list[int], back: list[dict[str, Any]]
) -> dict[str, Any]:
    last = NOW - timedelta(hours=1) + timedelta(hours=top - 150)
    ed = [{"chat_id": A, "id": i, "text": f"E{i} " + lens(rng), "edit_date": (NOW - timedelta(minutes=i)).isoformat()}
          for i in rng.sample(range(1, 150), 8)]  # fmt: skip
    new = [{"chat_id": A, "id": top + k, "date": (last + timedelta(minutes=5 * k)).isoformat(),
            "text": "N" + lens(rng), "sender_name": "u9"} for k in range(1, 6)]  # fmt: skip
    return {"edit": ed, "add": new + back, "delete": [{"chat_id": A, "id": i} for i in dels]}


@pytest.mark.parametrize("mode", ["tombstone", "purge"])
@pytest.mark.parametrize("sizes", [None, [37, 1, 64, 100, 9] * 20, [1] * 400])
def test_sync_runs_keep_windows_equal_to_a_full_rebuild(
    chat_conn: Any, mode: str, sizes: Any
) -> None:
    conn, rng = chat_conn, random.Random(21)
    td = scen([{}, {}, {}], sizes)
    # first-of-window ids are the likeliest to expose boundary bugs; delete a spread of ids
    dels1, dels2 = [3, 4, 20, 21, 22, 60, 99, 100], [101, 140]
    orig = {i: td._messages[A][i << 20] for i in dels1}  # noqa: SLF001
    td._runs[0] = changes(rng, 150, dels1, [])  # noqa: SLF001
    gone = td.get_chat(A)  # keep a handle
    assert gone is not None
    sync(conn, td, mode)
    check(conn)
    td.next_run()
    r = sync(conn, td, mode)
    assert r.status == "ok" and r.stats["windows_rebuilt"] + r.stats["windows_deleted"] > 0
    check(conn)
    saved = [{"chat_id": A, "id": i, "date": orig[i].date.isoformat(), "text": f"back {i}",
              "sender_name": "u1"} for i in dels1[:3]]  # fmt: skip
    td._runs[0] = changes(rng, 160, dels2, saved)  # noqa: SLF001
    td.next_run()
    sync(conn, td, mode)
    check(conn)
    td.next_run()
    sync(conn, td, mode)
    check(conn)


def test_crash_between_sweep_and_window_step_then_recovery(
    chat_conn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn, rng = chat_conn, random.Random(2)
    td = scen([{}, {}])
    td._runs[0] = changes(rng, 150, [3, 4, 50], [])  # noqa: SLF001
    sync(conn, td)
    check(conn)
    td.next_run()
    before = actual(conn)
    real = windows.rebuild_dirty

    def boom(*a: Any) -> Any:
        raise TdError(500, "killed")

    monkeypatch.setattr(windows, "rebuild_dirty", boom)
    with pytest.raises(TdError):
        sync(conn, td)
    assert actual(conn) == before  # windows not touched
    marker = conn.execute("SELECT windows_dirty_from >> 20 FROM chat").fetchone()[0]
    assert marker == 3  # lowest of the edits/deletes/new ids written by the killed run
    assert conn.execute("SELECT status FROM sync_run ORDER BY run_id DESC LIMIT 1").fetchone() == ("error",)  # fmt: skip
    monkeypatch.setattr(windows, "rebuild_dirty", real)
    r = sync(conn, td)
    assert r.status == "ok"
    check(conn)  # rebuilt exactly to the full result and marker cleared
    # windows before the marker's window were not rewritten
    assert r.stats["windows_rebuilt"] >= 1


def test_window_step_runs_on_partial_runs_not_on_error(
    chat_conn: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    conn = chat_conn
    # time_limited: cut off in backfill after some pages were stored
    td = scen([{}], [10] * 60)
    s = replace(SET, page_delay_ms=1000)
    res = ingest.sync(conn, lambda _: td, s, Clock(), max_minutes=3 / 60)
    assert res.status == "time_limited"
    assert conn.execute("SELECT count(*) FROM message").fetchone()[0] > 0
    check(conn)
    assert conn.execute('SELECT count(*) FROM "window"').fetchone()[0] > 0
    # flood_limited
    conn.execute("DELETE FROM message")
    conn.execute('DELETE FROM "window"')
    conn.execute("UPDATE chat SET newest_message_id=NULL, oldest_message_id=NULL, backfill_done=false")  # fmt: skip
    td2 = scen([], [10] * 60, faults=[{"call": n, "flood_wait": 1} for n in (3, 4, 5)])
    res = sync(conn, td2)
    assert res.status == "flood_limited"
    assert conn.execute("SELECT count(*) FROM message").fetchone()[0] > 0
    check(conn)
    # error: pages were stored before the failure, window step does not run, marker stays
    conn.execute("DELETE FROM message")
    conn.execute('DELETE FROM "window"')
    conn.execute("UPDATE chat SET newest_message_id=NULL, oldest_message_id=NULL, backfill_done=false, windows_dirty_from=NULL")  # fmt: skip
    td3 = scen([], [10] * 60, faults=[{"call": 4, "error": "SECRET x"}])
    with pytest.raises(TdError):
        sync(conn, td3)
    assert conn.execute("SELECT count(*) FROM message").fetchone()[0] > 0
    assert conn.execute('SELECT count(*) FROM "window"').fetchone() == (0,)
    assert conn.execute("SELECT windows_dirty_from IS NOT NULL FROM chat").fetchone() == (True,)
    r = sync(conn, scen([], [10] * 60))
    assert r.status == "ok"
    check(conn)
    assert "SECRET" not in caplog.text and ACC not in caplog.text
    assert all("Chat T" not in rec.getMessage() for rec in caplog.records)


def test_title_rename_rebuilds_all_windows_stale_and_matches_full_rebuild(chat_conn: Any) -> None:
    """Delta 78ffdbc: a title-only rename marks the chat; same-title re-enable marks nothing."""
    global TITLE
    conn, rng = chat_conn, random.Random(9)
    seed_messages(conn, rng, 80)
    mark(conn, 1)
    rebuild(conn)
    conn.execute('UPDATE "window" SET stale = false')
    ids = conn.execute('SELECT window_id FROM "window" ORDER BY first_message_id').fetchall()
    n = len(ids)
    assert n > 3
    chats.enable(conn, Chat(chat_id=A, title=TITLE, kind="supergroup"), ACC)  # same title
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)
    # a rename with an existing, lower marker keeps the lower marker
    ingest.mark_windows_dirty(conn, A, 1 << S)
    chats.enable(conn, Chat(chat_id=A, title="Renamed", kind="supergroup"), ACC)
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (1 << S,)
    conn.execute("UPDATE chat SET windows_dirty_from = NULL")
    chats.enable(conn, Chat(chat_id=A, title="Renamed", kind="supergroup"), ACC)  # now same again
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)
    chats.enable(conn, Chat(chat_id=A, title="Renamed 2", kind="supergroup"), ACC)
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (1 << S,)
    res = rebuild(conn)
    assert res["windows_rebuilt"] == n and res["windows_deleted"] == 0
    TITLE = "Renamed 2"
    try:
        check(conn)
    finally:
        TITLE = "Chat T"
    assert conn.execute('SELECT count(*) FROM "window" WHERE stale').fetchone() == (n,)
    assert conn.execute('SELECT window_id FROM "window" ORDER BY first_message_id').fetchall() == ids  # fmt: skip
    assert conn.execute('SELECT count(*) FROM "window" WHERE text LIKE %s', ("[Chat T]%",)).fetchone() == (0,)  # fmt: skip


def test_rename_of_chat_without_messages_marks_nothing_and_new_chat_not_marked(
    chat_conn: Any,
) -> None:
    conn = chat_conn
    chats.enable(conn, Chat(chat_id=A, title="Other", kind="supergroup"), ACC)
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)
    assert windows.rebuild_dirty(conn, SET) == {"windows_rebuilt": 0, "windows_deleted": 0}
