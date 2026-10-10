"""QA probes for T5: kill at every request, chained kills, forget scope, log hygiene.

Contributed by QA on 7db4e86; the kill parametrization is trimmed to keep `make ci` fast
(the full grid ran in about 65 s) while covering exclusive and inclusive paging, uneven
pages and one-message pages.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from tgi import chats, db, ingest
from tgi.config import Settings
from tgi.td_fake import FakeTdClient
from tgi.td_port import TdError

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
ACC = "+34600000001"
SET = Settings(embed_provider="fake", backfill_years=3, page_delay_ms=0)
A, B = -1001000000201, -1001000000202


class Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        pass


def msgs(first: int, last: int, top: int, text: str = "m") -> list[dict[str, Any]]:
    return [
        {"id": i, "date": (NOW - (top - i) * timedelta(minutes=1)).isoformat(),
         "text": f"{text} {i}", "sender_id": 1, "sender_name": "Ana"}
        for i in range(first, last + 1)
    ]  # fmt: skip


@pytest.fixture
def conn(create_database: Any) -> Any:
    url = create_database()
    with db.connect(url) as c:
        db.migrate(c)
    with psycopg.connect(url, autocommit=True) as c:
        yield c


def ids(conn: Any, chat_id: int) -> list[int]:
    return [r[0] >> 20 for r in conn.execute(
        "SELECT message_id FROM message WHERE chat_id=%s ORDER BY 1", (chat_id,))]  # fmt: skip


def go(conn: Any, td: FakeTdClient) -> str | None:
    try:
        return ingest.sync(conn, lambda _: td, SET, Clock()).status
    except TdError:
        return None


FULL, UNEVEN, ONES = [100], [37, 1, 64, 100, 9] * 50, [1] * 1000


def killed_run_left_no_gap(conn: Any) -> None:
    """The checkpoint must not move on a killed catch-up. Since T6 the run also sweeps, so
    a kill may land after catch-up completed: then the checkpoint moved with no gap."""
    chk = conn.execute("SELECT newest_message_id>>20 FROM chat").fetchone()
    assert chk == (50,) or (chk == (450,) and ids(conn, A) == list(range(1, 451)))


@pytest.mark.parametrize(
    ("inclusive", "sizes", "k1"),
    [
        (False, FULL, 1),
        (True, FULL, 2),
        (False, UNEVEN, 3),
        (True, UNEVEN, 4),
        (False, UNEVEN, 7),
        (True, ONES, 2),
        (False, ONES, 7),
    ],
    ids=["excl-full-1", "incl-full-2", "excl-uneven-3", "incl-uneven-4", "excl-uneven-7",
         "incl-ones-2", "excl-ones-7"],
)  # fmt: skip
def test_kill_at_every_request_leaves_no_gap(
    conn: Any, inclusive: bool, sizes: list[int], k1: int
) -> None:
    """Catch-up of 400 new messages killed at call k1, then k1+1 (chained), then clean."""
    chat = {"chat_id": A, "title": "a", "kind": "supergroup", "messages": msgs(1, 50, 450),
            "page_sizes": sizes}  # fmt: skip
    td = FakeTdClient({
        "from_inclusive": inclusive, "chats": [chat],
        "runs": [{"add": [{**m, "chat_id": A} for m in msgs(51, 450, 450)],
                  "faults": [{"call": k1, "error": "x"}]},
                 {"faults": [{"call": k1 + 1, "error": "x"}]}, {}],
    })  # fmt: skip
    chats.enable(conn, td.get_chat(A), ACC)
    assert go(conn, td) == "ok"
    assert ids(conn, A) == list(range(1, 51))
    td.next_run()
    first = go(conn, td)
    if first is None:
        killed_run_left_no_gap(conn)
    td.next_run()
    second = go(conn, td)  # may finish already: the jump skips stored pages
    if second is None:
        killed_run_left_no_gap(conn)
        td.next_run()
        assert go(conn, td) == "ok"
    assert ids(conn, A) == list(range(1, 451))
    assert conn.execute("SELECT newest_message_id>>20 FROM chat").fetchone() == (450,)


@pytest.mark.parametrize(
    ("inclusive", "k"), [(False, 1), (True, 2), (False, 3), (True, 4), (False, 7), (True, 11)]
)
def test_kill_during_first_run_and_backfill_then_complete(
    conn: Any, inclusive: bool, k: int
) -> None:
    chat = {"chat_id": A, "title": "a", "kind": "supergroup", "messages": msgs(1, 300, 300),
            "page_sizes": [37, 1, 64, 100, 9] * 50}  # fmt: skip
    td = FakeTdClient({"from_inclusive": inclusive, "chats": [chat],
                       "faults": [{"call": k, "error": "x"}], "runs": [{}]})  # fmt: skip
    chats.enable(conn, td.get_chat(A), ACC)
    if go(conn, td) is None:  # k beyond the run's request count means no kill happened
        td.next_run()
        assert go(conn, td) == "ok"
    assert ids(conn, A) == list(range(1, 301))
    counts = conn.execute("SELECT count(*), count(DISTINCT message_id) FROM message")
    assert counts.fetchone() == (300, 300)


def test_forget_deletes_only_that_chat(conn: Any) -> None:
    td = FakeTdClient({"chats": [
        {"chat_id": A, "title": "a", "kind": "supergroup", "messages": msgs(1, 5, 5)},
        {"chat_id": B, "title": "b", "kind": "supergroup", "messages": msgs(1, 5, 5)},
    ]})  # fmt: skip
    for c in (A, B):
        chats.enable(conn, td.get_chat(c), ACC)
    ingest.sync(conn, lambda _: td, SET, Clock())
    conn.execute(
        "INSERT INTO chat_tag (chat_id, tag, status)"
        " VALUES (%s, 't', 'confirmed'), (%s, 't', 'confirmed')",
        (A, B),
    )
    conn.execute(
        'INSERT INTO "window" (chat_id, first_message_id, start_date, end_date, message_ids,'
        " text) VALUES (%s, 1, now(), now(), '{1}', 'x'), (%s, 1, now(), now(), '{1}', 'x')",
        (A, B),
    )
    assert chats.forget(conn, A) is True
    assert chats.forget(conn, A) is False
    for t in ("message", "chat_tag", '"window"'):
        count = f"SELECT count(*) FROM {t} WHERE chat_id = %s"  # noqa: S608 - fixed table names
        assert conn.execute(count, (A,)).fetchone() == (0,)
        assert conn.execute(count, (B,)).fetchone()[0] > 0
    assert conn.execute("SELECT count(*) FROM sync_run").fetchone() == (1,)


def test_logs_have_no_text_or_phone(conn: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    m = msgs(1, 3, 3, text="SECRETWORD")
    m.append({"id": 4, "date": NOW.isoformat(), "text": "SECRETWORD", "sender_name": "Zed"})
    td = FakeTdClient({"chats": [{"chat_id": A, "title": "TITLEWORD", "kind": "supergroup",
                                  "messages": m}],
                       "faults": [{"call": 1, "flood_wait": 1}]})  # fmt: skip
    chats.enable(conn, td.get_chat(A), ACC)
    ingest.sync(conn, lambda _: td, SET, Clock())
    out = "\n".join(r.getMessage() for r in caplog.records)
    assert out and "SECRETWORD" not in out and "TITLEWORD" not in out
    assert ACC not in out and "Ana" not in out
