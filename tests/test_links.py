"""T9: message links from TDLib-shaped ids (SPEC §5.5 links, A11)."""

from datetime import timedelta
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, chat, enable

from tgi import search
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
TD_ID = 1 << 20  # TDLib message_id 1 048 576 = server message id 1

SUPERGROUP, CHANNEL, BASIC, PRIVATE = -1001234567890, -1009876543210, -4567, 777000


@pytest.mark.parametrize(
    ("kind", "chat_id", "username", "expected"),
    [
        ("supergroup", SUPERGROUP, None, "https://t.me/c/1234567890/1"),
        ("channel", CHANNEL, None, "https://t.me/c/9876543210/1"),
        ("channel", CHANNEL, "somechannel", "https://t.me/somechannel/1"),
        ("supergroup", SUPERGROUP, "somegroup", "https://t.me/somegroup/1"),
        ("basic_group", BASIC, None, None),
        ("private", PRIVATE, None, None),
        ("private", PRIVATE, "someone", None),  # a user's username is not a message link
    ],
)
def test_link_formats(kind: str, chat_id: int, username: str | None, expected: str | None) -> None:
    assert search.link(kind, chat_id, username, TD_ID >> 20) == expected


def test_a11_links_on_search_hits_use_the_server_id(conn: Conn) -> None:
    def msgs(text: str) -> list[dict[str, Any]]:
        return [{"id": 1, "date": (NOW - timedelta(minutes=10)).isoformat(), "text": text},
                {"id": 3, "date": NOW.isoformat(), "text": f"{text} again"}]  # fmt: skip

    td = FakeTdClient({"chats": [
        chat(SUPERGROUP, msgs("alpha")),
        chat(CHANNEL, msgs("bravo"), kind="channel", username="somechannel"),
        chat(BASIC, msgs("charlie"), kind="basic_group"),
        chat(PRIVATE, msgs("delta"), kind="private"),
    ]})  # fmt: skip
    enable(conn, td, SUPERGROUP, CHANNEL, BASIC, PRIVATE)
    sync(conn, td)
    stored = conn.execute("SELECT message_id FROM message WHERE chat_id = %s ORDER BY 1",
                          (SUPERGROUP,)).fetchall()  # fmt: skip
    assert stored == [(TD_ID,), (3 * TD_ID,)]

    def links(query: str) -> list[str | None]:
        return [m.link for h in search.search(conn, [query], mode="keyword") for m in h.messages]

    assert links("alpha") == ["https://t.me/c/1234567890/1", "https://t.me/c/1234567890/3"]
    assert links("bravo") == ["https://t.me/somechannel/1", "https://t.me/somechannel/3"]
    assert links("charlie") == [None, None]
    assert links("delta") == [None, None]


def test_a11_the_schema_refuses_an_id_that_is_not_server_shaped(conn: Conn) -> None:
    """The ingest path skips such ids (`test_ingest::test_skipped_ids_are_logged_by_id_only`);
    the CHECK constraint is the backstop."""
    td = FakeTdClient({"chats": [chat(SUPERGROUP, [])]})
    enable(conn, td, SUPERGROUP)
    insert = (
        "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
        " VALUES (%s, %s, %s, now(), 'x')"
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(insert, (SUPERGROUP, TD_ID + 1, 1))
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute(insert, (SUPERGROUP, TD_ID, 2))  # server id must be message_id >> 20
    conn.execute(insert, (SUPERGROUP, TD_ID, 1))
