"""QA probes for T5a: exit codes via a real process, stale-run marking, marker ordering."""

import contextlib
import json
import os
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest

from tgi import chats, db, ingest
from tgi.config import Settings
from tgi.td_fake import FakeTdClient
from tgi.td_port import TdError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACC = "+34600000001"
A = -1001000000401
SET = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=0)


class Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        pass


def msgs(first: int, last: int, top: int) -> list[dict[str, Any]]:
    return [{"id": i, "date": (NOW - (top - i) * timedelta(minutes=1)).isoformat(),
             "text": f"SECRET {i}"} for i in range(first, last + 1)]  # fmt: skip


def scen(**extra: Any) -> dict[str, Any]:
    return {"chats": [{"chat_id": A, "title": "t", "kind": "supergroup",
                       "messages": msgs(1, 250, 250)}], **extra}  # fmt: skip


TGI = str(Path(sys.executable).parent / "tgi")


def tgi(db_url: str, tmp: Path, *args: str, scenario: dict[str, Any] | None = None):  # type: ignore[no-untyped-def]
    cmd = [TGI, *args]
    if scenario is not None:
        f = tmp / "s.json"
        f.write_text(json.dumps(scenario))
        cmd += ["--scenario", str(f)]
    env = {**os.environ, "DATABASE_URL": db_url, "PAGE_DELAY_MS": "0"}
    return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)  # noqa: S603


def rows(conn: Any) -> list[tuple[Any, ...]]:
    sql = "SELECT run_id, status, finished_at IS NOT NULL FROM sync_run ORDER BY 1"
    return conn.execute(sql).fetchall()  # type: ignore[no-any-return]


def enable(conn: Any) -> None:
    chats.enable(conn, FakeTdClient(scen()).get_chat(A), ACC)


def test_exit_codes_through_the_installed_entrypoint(
    conn: Any, db_url: str, tmp_path: Path
) -> None:
    """The console script, not CliRunner; partial runs and the lock are in test_ingest."""
    enable(conn)
    r = tgi(db_url, tmp_path, "sync", scenario=scen())
    assert r.returncode == 0, r.stderr
    n_before = len(rows(conn))
    err = tgi(db_url, tmp_path, "sync", scenario=scen(faults=[{"call": 1, "error": "SECRET boom"}]))
    assert err.returncode == 3 and "SECRET" not in err.stderr + err.stdout, err.stderr
    assert rows(conn)[-1][1] == "error" and len(rows(conn)) == n_before + 1
    for args in (["sync", "--max-minutes", "x"], [], ["chats"], ["db"], ["nosuch"],
                 ["sync", "--resweep-days", "0", "1"]):  # fmt: skip
        r = tgi(db_url, tmp_path, *args)
        assert r.returncode == 1, (args, r.returncode, r.stderr[-200:])
    assert tgi(db_url, tmp_path, "--help").returncode == 0
    assert tgi(db_url, tmp_path, "sync", "--help").returncode == 0
    assert tgi(db_url, tmp_path, "--version").returncode == 0
    no_db = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    r = subprocess.run([TGI, "sync"], capture_output=True, text=True, env=no_db)  # noqa: S603
    assert r.returncode == 1, (r.returncode, r.stderr[-200:])


def test_stale_running_rows_interrupted_only_under_lock(conn: Any, db_url: str) -> None:
    enable(conn)
    conn.execute("INSERT INTO sync_run (status) VALUES ('running'), ('running'), ('ok')")
    started, release = threading.Event(), threading.Event()
    td = FakeTdClient(scen())

    def slow(_: str) -> FakeTdClient:
        started.set()
        assert release.wait(30)
        return td

    out: dict[str, Any] = {}

    def first() -> None:
        with db.connect(db_url) as c:
            out["res"] = ingest.sync(c, slow, SET, Clock())

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(30)
    live = conn.execute("SELECT run_id FROM sync_run WHERE status='running'").fetchall()
    assert len(live) == 1  # the stale ones were cleaned, the live one is running
    # a concurrent second run must fail on the lock and leave the live row alone
    with db.connect(db_url) as c2, pytest.raises(ingest.SyncLocked):
        ingest.sync(c2, lambda _: td, SET, Clock())
    assert conn.execute("SELECT run_id FROM sync_run WHERE status='running'").fetchall() == live
    release.set()
    t.join(60)
    assert out["res"].status == "ok"
    st = [r[1] for r in rows(conn)]
    assert st == ["interrupted", "interrupted", "ok", "ok"], st
    assert all(r[2] for r in rows(conn) if r[0] != 3)  # row 3 is my seeded ok row
    open_rows = "SELECT count(*) FROM sync_run WHERE status='interrupted' AND finished_at IS NULL"
    assert conn.execute(open_rows).fetchone() == (0,)


def test_migration_0002_on_v1_database(create_database: Any, tmp_path: Path) -> None:
    url = create_database()
    v1 = tmp_path / "m1"
    v1.mkdir()
    for v, p in db.migrations():
        if v == 1:
            (v1 / p.name).write_bytes(p.read_bytes())
    with db.connect(url) as c:
        assert db.migrate(c, v1) == [1]
    with psycopg.connect(url, autocommit=True) as c:
        # A v1 row by hand: chats.enable reads columns added by later migrations.
        c.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind, enabled)"
            " VALUES (%s, %s, 'A', 'supergroup', true)",
            (A, ACC),
        )
        c.execute(
            "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
            " VALUES (%s, %s, 5, now(), 'x')",
            (A, 5 << 20),
        )
    with db.connect(url) as c:
        assert db.migrate(c) == [2, 3]  # every pending migration, 0003 included
        assert db.migrate(c) == []
    with psycopg.connect(url, autocommit=True) as c:
        col = c.execute(
            "SELECT data_type, is_nullable FROM information_schema.columns"
            " WHERE table_name='chat' AND column_name='windows_dirty_from'"
        ).fetchone()
        assert col == ("bigint", "YES")
        assert c.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)
        assert c.execute("SELECT count(*) FROM message").fetchone() == (1,)


TRIGGER = """
CREATE TABLE ins_log (chat_id bigint, message_id bigint, dirty bigint);
CREATE FUNCTION log_ins() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO ins_log SELECT NEW.chat_id, NEW.message_id, windows_dirty_from
    FROM chat WHERE chat_id = NEW.chat_id;
  RETURN NEW;
END $$;
CREATE TRIGGER t_ins BEFORE INSERT OR UPDATE ON message FOR EACH ROW EXECUTE FUNCTION log_ins();
"""


@pytest.mark.usefixtures("no_window_step")  # reads the marker the writers leave
@pytest.mark.parametrize("sizes", [[100], [37, 1, 64, 100, 9] * 20, [1] * 1000])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 5])
def test_marker_precedes_every_write_and_survives_kills(
    conn: Any, sizes: list[int], inclusive: bool, k: int
) -> None:
    conn.execute(TRIGGER)
    s = scen(from_inclusive=inclusive)
    s["chats"][0]["page_sizes"] = sizes
    s["runs"] = [{"add": [{**m, "chat_id": A} for m in msgs(251, 400, 400)]}, {}]
    if k:
        s["faults"] = [{"call": k, "error": "x"}]
    td = FakeTdClient(s)
    chats.enable(conn, td.get_chat(A), ACC)
    for step in range(4):
        with contextlib.suppress(TdError):
            ingest.sync(conn, lambda _: td, SET, Clock())
        dirty = conn.execute("SELECT windows_dirty_from FROM chat").fetchone()[0]
        low = conn.execute("SELECT min(message_id) FROM message").fetchone()[0]
        if low is not None:
            assert dirty is not None and dirty <= low, (step, dirty, low)
        if step == 0 and td._runs:  # noqa: SLF001
            td.next_run()
    late = "SELECT count(*) FROM ins_log WHERE dirty IS NULL OR dirty > message_id"
    bad = conn.execute(late).fetchone()
    assert bad == (0,)
    assert conn.execute("SELECT count(DISTINCT message_id) FROM message").fetchone() == (400,)
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (1 << 20,)
