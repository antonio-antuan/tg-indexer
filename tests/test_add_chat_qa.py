"""QA regression tests for `add_chat`: #38 range edges and #39 through a TdlibClient."""

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, chat
from test_mcp import NEW, Joiner, enabled_chats, with_telegram
from test_td import CH, FakeLib, client

from tgi.td_fake import FakeTdClient
from tgi.td_port import TdClient

Conn = psycopg.Connection[tuple[object, ...]]
SCENARIO: dict[str, Any] = {
    "chats": [
        chat(
            NEW,
            [{"id": 1, "date": (NOW - timedelta(hours=2)).isoformat(), "text": "hello"}],
            username="valencia_new",
        )
    ],
    "runs": [],
}
ACCOUNT = "+34600000001"


@pytest.mark.parametrize("years", [0, -1, 101, 10000, 2**31, 2**63 - 1, 2**63, 2**64])
def test_out_of_range_years_change_nothing(conn: Conn, db_url: str, years: int) -> None:
    joiner = Joiner(SCENARIO)
    with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new", "years": 5})])
    before = enabled_chats(conn)
    (result,) = with_telegram(
        db_url, joiner, [("add_chat", {"link": "@valencia_new", "years": years})]
    )
    assert result["error"]["code"] == "bad_argument"
    assert len(joiner.clients) == 1 and joiner.joined == [NEW]  # no second client, no join
    assert enabled_chats(conn) == before
    assert sync(conn, FakeTdClient(SCENARIO)).status == "ok"


def test_out_of_range_years_on_a_new_chat_write_nothing(conn: Conn, db_url: str) -> None:
    joiner = Joiner(SCENARIO)
    (result,) = with_telegram(
        db_url, joiner, [("add_chat", {"link": "@valencia_new", "years": 101})]
    )
    assert result["error"]["code"] == "bad_argument"
    assert joiner.clients == [] and enabled_chats(conn) == {}


def _opener(lib: FakeLib, tmp_path: Path) -> Callable[[], tuple[str, TdClient]]:
    return lambda: (ACCOUNT, client(tmp_path, lib))


def test_join_request_then_approval_then_retry_enables_the_chat(
    conn: Conn, db_url: str, tmp_path: Path
) -> None:
    """#39 end to end through `add_chat` over a TdlibClient (the library is faked)."""
    lib = FakeLib(logged_in=True)
    link = "https://t.me/+askAdmins12"
    opener = _opener(lib, tmp_path)
    for _ in range(2):  # sent, then still waiting
        (pending,) = with_telegram(db_url, opener, [("add_chat", {"link": link})])
        assert pending["error"]["code"] == "join_pending" and enabled_chats(conn) == {}
    lib.approved.add(link)
    (added, again) = with_telegram(
        db_url, opener, [("add_chat", {"link": link}), ("add_chat", {"link": link})]
    )  # the second call finds the account a member already
    assert added["chat_id"] == again["chat_id"] == CH
    assert list(enabled_chats(conn)) == [CH]


def test_public_chat_join_request_is_pending_not_enabled(
    conn: Conn, db_url: str, tmp_path: Path
) -> None:
    lib = FakeLib(logged_in=True)
    lib.requests_to_join.add(CH)
    (result,) = with_telegram(db_url, _opener(lib, tmp_path), [("add_chat", {"link": "@avisos"})])
    assert result["error"]["code"] == "join_pending" and enabled_chats(conn) == {}


def test_already_a_member_by_invite_counts_as_joined(
    conn: Conn, db_url: str, tmp_path: Path
) -> None:
    lib = FakeLib(logged_in=True)
    link = "https://t.me/+memberAlready"
    (result,) = with_telegram(db_url, _opener(lib, tmp_path), [("add_chat", {"link": link})])
    assert result["kind"] == "supergroup" and len(enabled_chats(conn)) == 1
