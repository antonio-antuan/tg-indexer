"""P2.T3: the MCP server's tools, stdio and bearer-token HTTP (SPEC §5.6, A8 but the bind).

HTTP runs in process (an ASGI client, no socket); stdio runs `tgi mcp --stdio` as a
subprocess through the SDK's stdio client. The embedder is the fake one throughout.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import anyio
import httpx2
import psycopg
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from mcp_types import CallToolResult
from test_embed import sync
from test_ingest import NOW, A, B, C, chat, enable, messages
from typer.testing import CliRunner

from tgi import chats, ingest, mcp, td
from tgi.cli import app
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient
from tgi.td_port import Chat

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
TOKEN = "t" * 43  # a fake MCP_TOKEN, as long as a real one
BASE = "http://127.0.0.1:8765"
LOOPBACK = mcp.transport_security("127.0.0.1", 8765, [])

# Chat A: 1 ← 2 ← 3 ← 6 ← 7 and 1 ← 4 are replies; 5 is not; 6 is tombstoned below.
A_TEXTS = {
    1: "Где продлить ВНЖ в Валенсии?",
    2: "В полиции на Patraix, нужна cita previa",
    3: "Спасибо!",
    4: "Я делал через gestoría",
    5: "paella на выходных?",
    6: "это сообщение удалят",
    7: "ответ на удалённое",
}
A_REPLIES = {2: 1, 3: 2, 4: 1, 6: 3, 7: 6}


def mid(server_id: int) -> int:
    return server_id << SHIFT


@pytest.fixture
def archive(conn: Conn) -> None:
    """A: a supergroup with a username, tagged `spain`; B: a basic group whose `docs` tag
    is only proposed; both synced and embedded with the fake provider."""
    a = [{"id": i, "date": (NOW - timedelta(hours=10 - i)).isoformat(), "text": t,
          "sender_name": "Ana", "reply_to": A_REPLIES.get(i)}
         for i, t in A_TEXTS.items()]  # fmt: skip
    b = [{"id": 1, "date": (NOW - timedelta(days=3)).isoformat(),
          "text": "renewal of the TIE card", "sender_name": "Bo"}]  # fmt: skip
    td = FakeTdClient({
        "chats": [chat(A, a, username="nie_valencia"), chat(B, b, kind="basic_group")],
        "runs": [],
    })  # fmt: skip
    enable(conn, td, A, B)
    sync(conn, td)
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s", (A, mid(6))
    )
    conn.execute(
        "INSERT INTO chat_tag (chat_id, tag, status) VALUES (%s, 'spain', 'confirmed')", (A,)
    )
    conn.execute(
        "INSERT INTO chat_tag (chat_id, tag, status, proposed_by)"
        " VALUES (%s, 'docs', 'proposed', 'owner')",
        (B,),
    )


# --- driving the server -------------------------------------------------------------------

Calls = list[tuple[str, dict[str, Any]]]


def outcome(result: CallToolResult) -> Any:
    """The tool's value, or {"error": {code, message}} for a tool error."""
    if result.is_error:
        text = result.content[0].text  # type: ignore[union-attr]
        prefix, _, error = text.partition(": ")  # the SDK names the tool first
        assert prefix.startswith("Error executing tool")  # no name for an unknown tool
        return {"error": json.loads(error)}
    value = result.structured_content
    if isinstance(value, dict) and set(value) == {"result"}:
        return value["result"]  # a list result, wrapped by the SDK
    return value


async def _run(client: Client, calls: Calls) -> list[Any]:
    async with client as c:
        return [outcome(await c.call_tool(name, args)) for name, args in calls]


def in_process(db_url: str, calls: Calls, embedder: FakeEmbedder | None = None) -> list[Any]:
    return anyio.run(_run, Client(mcp.build_server(db_url, embedder)), calls)


def over_http(db_url: str, calls: Calls, embedder: FakeEmbedder | None = None) -> list[Any]:
    async def go() -> list[Any]:
        server = mcp.build_server(db_url, embedder)
        asgi = mcp.http_app(server, TOKEN, LOOPBACK)
        async with (
            server.session_manager.run(),  # the app's lifespan
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=asgi),
                base_url=BASE,
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as http,
        ):
            return await _run(
                Client(streamable_http_client(f"{BASE}/mcp", http_client=http)), calls
            )

    return anyio.run(go)


def over_stdio(db_url: str, calls: Calls, cwd: Path) -> list[Any]:
    params = StdioServerParameters(
        command=str(Path(sys.executable).parent / "tgi"),
        args=["mcp", "--stdio"],
        env={"DATABASE_URL": db_url, "EMBED_PROVIDER": "fake"},
        cwd=cwd,
    )
    return anyio.run(_run, Client(params), calls)


def server_ids(messages: list[dict[str, Any]]) -> list[int]:
    return [m["message_id"] >> SHIFT for m in messages]


# --- shapes on every transport ------------------------------------------------------------

READS: Calls = [
    ("list_chats", {}),
    ("list_chats", {"tag": "spain"}),
    ("list_tags", {}),
    ("search", {"queries": ["ВНЖ"], "mode": "keyword"}),
    ("search", {"queries": ["cita previa", "TIE"], "since": "2026-09-01"}),
    ("get_thread", {"chat_id": A, "message_id": mid(3)}),
    ("get_context", {"chat_id": A, "message_id": mid(4), "before": 2, "after": 2}),
]


def test_a8_tools_answer_with_the_spec_shapes_over_http_and_stdio(
    archive: None, db_url: str, tmp_path: Path
) -> None:
    chats, tagged, tags, keyword, hybrid, thread, context = results = in_process(db_url, READS)

    assert chats == [
        {"chat_id": A, "title": f"chat {A}", "kind": "supergroup", "tags": ["spain"],
         "message_count": 6, "first_date": (NOW - timedelta(hours=9)).isoformat(),
         "last_date": (NOW - timedelta(hours=3)).isoformat()},
        {"chat_id": B, "title": f"chat {B}", "kind": "basic_group", "tags": [],
         "message_count": 1, "first_date": (NOW - timedelta(days=3)).isoformat(),
         "last_date": (NOW - timedelta(days=3)).isoformat()},
    ]  # fmt: skip
    assert tagged == chats[:1]
    assert tags == [
        {"tag": "docs", "status": "proposed", "chat_count": 1},
        {"tag": "spain", "status": "confirmed", "chat_count": 1},
    ]

    (hit,) = keyword
    assert set(hit) == {"window_id", "chat_id", "chat_title", "tags", "start_date",
                        "end_date", "score", "messages", "highlights"}  # fmt: skip
    assert (hit["chat_id"], hit["tags"], server_ids(hit["messages"])) == (A, ["spain"], [1])
    assert hit["messages"][0] == {
        "message_id": mid(1),
        "sender_name": "Ana",
        "date": (NOW - timedelta(hours=9)).isoformat(),
        "text": A_TEXTS[1],
        "link": "https://t.me/nie_valencia/1",
    }
    found = {(h["chat_id"], m["message_id"] >> SHIFT) for h in hybrid for m in h["messages"]}
    assert {(A, 2), (B, 1)} <= found  # one query each, fused
    assert all(m["link"] is None for h in hybrid if h["chat_id"] == B for m in h["messages"])

    # The chain 3 → 2 → 1, then the root's other replies; 6 is tombstoned, 7 still found.
    assert server_ids(thread) == [1, 2, 3, 4, 7]
    assert thread[2] == {
        "message_id": mid(3),
        "sender_name": "Ana",
        "date": (NOW - timedelta(hours=7)).isoformat(),
        "text": A_TEXTS[3],
        "link": "https://t.me/nie_valencia/3",
    }
    assert server_ids(context) == [2, 3, 4, 5, 7]  # 6 is tombstoned

    assert over_http(db_url, READS) == results
    assert over_stdio(db_url, READS, tmp_path) == results


def test_get_thread_and_get_context_edges(archive: None, db_url: str) -> None:
    results = in_process(db_url, [
        ("get_thread", {"chat_id": A, "message_id": mid(7)}),  # up through the tombstone
        ("get_thread", {"chat_id": A, "message_id": mid(3), "limit": 2}),
        ("get_thread", {"chat_id": A, "message_id": mid(5)}),
        ("get_thread", {"chat_id": A, "message_id": mid(99)}),
        ("get_context", {"chat_id": A, "message_id": mid(1), "before": 5, "after": 0}),
        ("get_context", {"chat_id": A, "message_id": mid(6), "before": 1, "after": 1}),
        ("get_context", {"chat_id": A, "message_id": mid(99)}),
    ])  # fmt: skip
    assert [server_ids(r) for r in results] == [
        [1, 2, 3, 4, 7],
        [2, 3],  # the chain nearest the message
        [5],
        [],
        [1],
        [5, 7],  # around a tombstoned message, which is not returned
        [],
    ]


# --- errors -------------------------------------------------------------------------------


def codes(results: list[Any]) -> list[str]:
    return [r["error"]["code"] for r in results]


def test_tool_errors_carry_a_code(archive: None, db_url: str) -> None:
    calls: Calls = [
        ("list_chats", {"tag": "nope"}),
        ("search", {"queries": ["x"], "tags": ["nope"]}),
        ("search", {"queries": ["x"], "chat_ids": [C]}),
        ("search", {"queries": ["x"], "limit": 0}),
        ("search", {"queries": ["x"], "limit": 51}),
        ("search", {"queries": ["x"], "since": "2026-09-02", "until": "2026-09-01"}),
        ("search", {"queries": [" "]}),
        ("get_thread", {"chat_id": C, "message_id": 1}),
        ("get_thread", {"chat_id": A, "message_id": mid(1), "limit": 51}),
        ("get_context", {"chat_id": C, "message_id": 1}),
        ("get_context", {"chat_id": A, "message_id": mid(1), "before": 60, "after": 41}),
        ("get_context", {"chat_id": A, "message_id": mid(1), "before": -1}),
        ("search", {"queries": ["x"], "mode": "fuzzy"}),  # the SDK's schema check
        ("search", {"queries": ["x"], "limit": True}),  # not a JSON integer
        ("list_chats", {"tag": "a\x00b"}),  # Postgres cannot store it
    ]
    results = in_process(db_url, calls)
    assert codes(results) == [
        "unknown_tag", "unknown_tag", "unknown_chat", "bad_range", "bad_range", "bad_range",
        "bad_argument", "unknown_chat", "bad_range", "unknown_chat", "bad_range", "bad_range",
        "bad_argument", "bad_argument", "bad_argument",
    ]  # fmt: skip
    assert all(r["error"]["message"] for r in results)
    assert over_http(db_url, calls) == results


def test_a10_a_tool_error_never_logs_the_callers_values(
    archive: None, db_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #26: the SDK logs a tool error's text at INFO, so no message may quote an
    argument. A planted marker goes into every argument an error can be about."""
    caplog.set_level(logging.DEBUG)
    marker, chat_marker = "zq7marker", -1009876543219
    calls: Calls = [
        ("list_chats", {"tag": marker}),
        ("search", {"queries": [marker], "tags": ["spain", marker]}),
        ("search", {"queries": [marker], "chat_ids": [A, chat_marker]}),
        ("search", {"queries": [marker], "mode": marker}),
        ("search", {"queries": [marker], "limit": marker}),
        ("get_thread", {"chat_id": chat_marker, "message_id": 1}),
        ("get_context", {"chat_id": chat_marker, "message_id": 1}),
        ("suggest_tags", {"suggestions": [{"chat_id": chat_marker, "tag": marker}]}),
    ]
    results = in_process(db_url, calls) + over_http(db_url, calls)
    assert all("error" in r for r in results)
    assert codes(results[:3]) == ["unknown_tag", "unknown_tag", "unknown_chat"]
    assert "1 of the tags are not known; see list_tags" in results[1]["error"]["message"]
    assert "Tool 'list_chats' failed" in caplog.text  # the SDK does log the error
    assert "rejected arguments: ['mode']" in caplog.text  # field names only
    assert codes(results[3:5]) == ["bad_argument", "bad_argument"]
    assert results[3]["error"]["message"] == "invalid mode"
    # The test's own MCP client logs what it sends at DEBUG; the server's records are the point.
    logged = "\n".join(
        f"{r.name} {r.getMessage()} {r.exc_text or ''}"
        for r in caplog.records
        if not r.name.startswith("mcp.client")
    )
    assert "mcp.server" in logged
    for text in (logged, json.dumps(results)):
        assert marker not in text and str(chat_marker)[1:] not in text


def test_a10_an_unknown_tool_name_is_never_logged(
    archive: None, db_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Issue #26: the SDK's own unknown-tool error quotes the name, and logs it."""
    caplog.set_level(logging.DEBUG)
    marker = "zq7marker"
    calls: Calls = [(marker, {}), (f"search {marker}", {"queries": [marker]})]
    results = in_process(db_url, calls) + over_http(db_url, calls)
    assert codes(results) == ["unknown_tool"] * 4
    assert "call to an unknown tool" in caplog.text
    logged = "\n".join(
        f"{r.name} {r.getMessage()} {r.exc_text or ''}"
        for r in caplog.records
        if not r.name.startswith("mcp.client")  # the test's own client logs what it sends
    )
    assert "mcp.server" in logged
    for text in (logged, json.dumps(results)):
        assert marker not in text


def test_a8_a_proposed_tag_does_not_filter_search(archive: None, db_url: str) -> None:
    """`docs` is only proposed for B, so it matches no chat; the confirmed `spain` does."""
    results = in_process(db_url, [
        ("search", {"queries": ["TIE"], "tags": ["docs"], "mode": "keyword"}),
        ("search", {"queries": ["TIE"], "mode": "keyword"}),
        ("search", {"queries": ["cita", "TIE"], "tags": ["spain"], "mode": "keyword"}),
        ("list_chats", {"tag": "docs"}),
    ])  # fmt: skip
    assert results[0] == []
    assert [h["chat_id"] for h in results[1]] == [B]
    assert [h["chat_id"] for h in results[2]] == [A]
    assert results[3] == []


def test_a8_a_failing_embedder_is_embedder_unavailable_and_keyword_still_answers(
    archive: None, db_url: str
) -> None:
    broken = FakeEmbedder(fail=lambda text: True)
    hybrid, semantic, keyword = over_http(db_url, [
        ("search", {"queries": ["ВНЖ"]}),
        ("search", {"queries": ["ВНЖ"], "mode": "semantic"}),
        ("search", {"queries": ["ВНЖ"], "mode": "keyword"}),
    ], embedder=broken)  # fmt: skip
    assert hybrid["error"]["code"] == semantic["error"]["code"] == "embedder_unavailable"
    assert 'mode="keyword"' in hybrid["error"]["message"]
    assert [h["chat_id"] for h in keyword] == [A]


def test_another_model_is_model_mismatch(archive: None, db_url: str) -> None:
    (result,) = in_process(
        db_url, [("search", {"queries": ["ВНЖ"]})], embedder=FakeEmbedder(model="other")
    )
    assert result["error"]["code"] == "model_mismatch"


def test_get_context_returns_at_most_100_messages(conn: Conn, db_url: str) -> None:
    """SPEC §5.6: context ≤ 100 messages, the message itself included."""
    msgs = [{"id": i, "date": (NOW - timedelta(minutes=200 - i)).isoformat(),
             "text": f"m{i}", "sender_name": "Ana"} for i in range(1, 151)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(C, msgs)], "runs": []})
    enable(conn, td, C)
    sync(conn, td)
    around = {"chat_id": C, "message_id": mid(75)}
    full, early, late, near_start = in_process(db_url, [
        ("get_context", {**around, "before": 50, "after": 50}),
        ("get_context", {**around, "before": 100, "after": 0}),
        ("get_context", {**around, "before": 0, "after": 100}),
        ("get_context", {"chat_id": C, "message_id": mid(3), "before": 60, "after": 40}),
    ])  # fmt: skip
    assert server_ids(full) == list(range(25, 125))  # `after` gives way: 49 later
    assert server_ids(early) == list(range(1, 76))  # only 74 earlier exist
    assert server_ids(late) == list(range(75, 151))  # only 75 later exist
    assert server_ids(near_start) == list(range(1, 3 + 40))  # 2 earlier, 39 later


# --- suggest_tags -------------------------------------------------------------------------


def tag_rows(conn: Conn) -> list[tuple[Any, ...]]:
    return conn.execute(
        "SELECT chat_id, tag, status, proposed_by FROM chat_tag ORDER BY chat_id, tag"
    ).fetchall()


def test_a8_suggest_tags_writes_only_proposed_rows(archive: None, conn: Conn, db_url: str) -> None:
    before = tag_rows(conn)
    results = over_http(db_url, [
        ("suggest_tags", {"suggestions": [
            {"chat_id": A, "tag": "spain", "reason": "already confirmed"},
            {"chat_id": A, "tag": " valencia ", "reason": "city in the title"},
            {"chat_id": B, "tag": "valencia"},
            {"chat_id": B, "tag": "docs"},
            {"chat_id": B, "tag": "admin", "status": "confirmed"},  # not a field: ignored
        ]}),
        ("suggest_tags", {"suggestions": [{"chat_id": A, "tag": "valencia"}]}),
        ("search", {"queries": ["cita"], "tags": ["valencia"], "mode": "keyword"}),
    ])  # fmt: skip
    assert results == [{"proposed": 3}, {"proposed": 0}, []]
    assert tag_rows(conn) == sorted([
        *before,
        (A, "valencia", "proposed", "mcp"),
        (B, "admin", "proposed", "mcp"),
        (B, "valencia", "proposed", "mcp"),
    ])  # fmt: skip
    assert (A, "spain", "confirmed", None) in before


def test_suggest_tags_is_all_or_nothing(archive: None, conn: Conn, db_url: str) -> None:
    before = tag_rows(conn)
    results = in_process(db_url, [
        ("suggest_tags", {"suggestions": [{"chat_id": A, "tag": "new"},
                                          {"chat_id": C, "tag": "new"}]}),
        ("suggest_tags", {"suggestions": [{"chat_id": A, "tag": "new"},
                                          {"chat_id": A, "tag": " "}]}),
        ("suggest_tags", {"suggestions": []}),
    ])  # fmt: skip
    assert codes(results[:2]) == ["unknown_chat", "bad_argument"]
    assert results[2] == {"proposed": 0}
    assert tag_rows(conn) == before


# --- HTTP auth ----------------------------------------------------------------------------

INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def test_a8_http_without_or_with_a_wrong_token_is_401_and_the_log_has_no_token(
    archive: None, db_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    server = mcp.build_server(db_url)
    asgi = mcp.http_app(server, TOKEN, LOOPBACK)
    wrong = [
        {},
        {"Authorization": f"Bearer {TOKEN[:-1]}"},
        {"Authorization": f"Bearer {TOKEN}x"},
        {"Authorization": f"bearer {TOKEN}"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
    ]

    async def go() -> list[httpx2.Response]:
        async with httpx2.AsyncClient(transport=httpx2.ASGITransport(app=asgi), base_url=BASE) as h:
            out = [await h.post("/mcp", json=INITIALIZE, headers=hd) for hd in wrong]
            out.append(await h.get("/anything", headers=wrong[1]))
            out.append(await h.get("/anything", headers={"Authorization": f"Bearer {TOKEN}"}))
            return out

    *rejected, passed = anyio.run(go)
    for r in rejected:
        assert (r.status_code, r.content) == (401, b"")
        assert r.headers["www-authenticate"] == "Bearer"
    assert passed.status_code == 404  # through to the app
    assert asgi.rejected == 7
    assert "rejected an unauthorized request (7 so far)" in caplog.text

    over_http(db_url, READS)  # a whole session with the token, logged too
    assert caplog.text
    assert TOKEN not in caplog.text and TOKEN[:20] not in caplog.text


def test_parse_bind() -> None:
    assert mcp.parse_bind("127.0.0.1:8765") == ("127.0.0.1", 8765)
    assert mcp.parse_bind("[::1]:8765") == ("::1", 8765)
    assert mcp.parse_bind("localhost:1") == ("localhost", 1)
    for bad in ("127.0.0.1", "::1:8765", ":8765", "127.0.0.1:0", "127.0.0.1:65536", "h:x"):
        with pytest.raises(ValueError, match="MCP_BIND"):
            mcp.parse_bind(bad)


# --- CLI ----------------------------------------------------------------------------------


def test_cli_mcp_token_prints_a_fresh_urlsafe_32_byte_token() -> None:
    runner = CliRunner()
    first, second = (runner.invoke(app, ["mcp", "token"]) for _ in range(2))
    assert first.exit_code == second.exit_code == 0
    token = first.stdout.strip()
    assert len(token) == 43 and token != second.stdout.strip()
    assert set(token) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


@pytest.mark.parametrize(
    ("args", "env", "message"),
    [
        (["mcp"], {}, "exactly one of --http and --stdio"),
        (["mcp", "--http", "--stdio"], {}, "exactly one of --http and --stdio"),
        (["mcp", "--http", "token"], {}, "do not go with a subcommand"),
        (["mcp", "--http"], {}, "MCP_TOKEN is not set"),
        (["mcp", "--http"], {"MCP_TOKEN": "short"}, "MCP_TOKEN is too short"),
        (["mcp", "--http"], {"MCP_TOKEN": TOKEN, "MCP_BIND": "127.0.0.1"}, "MCP_BIND must be"),
    ],
)
def test_cli_mcp_refuses_to_start(
    args: list[str],
    env: dict[str, str],
    message: str,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MCP_TOKEN", raising=False)
    monkeypatch.delenv("MCP_BIND", raising=False)
    monkeypatch.setenv("DATABASE_URL", db_url)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    assert message in result.output
    assert TOKEN not in result.output


# --- add_chat ---------------------------------------------------------------------------------

NEW, PRIV, BASIC, PENDING = -1001000000201, 4242, -77, -1001000000202
ASKING = -1001000000203  # a public channel whose join sends a request
JOIN_SCENARIO: dict[str, Any] = {
    "chats": [
        chat(NEW, [], username="valencia_new"),
        chat(PRIV, [], kind="private", username="some_person"),
        chat(BASIC, [], kind="basic_group"),
        chat(PENDING, [], kind="channel"),
        chat(ASKING, [], kind="channel", username="ask_first", join_request=True),
    ],
    "invites": [
        {"link": "https://t.me/+basicGroupInvite", "chat_id": BASIC},
        {"link": "https://t.me/+askTheAdminsFirst", "chat_id": PENDING, "request": True},
    ],
    "runs": [],
}


class Joiner:
    """A `Telegram` opener over fresh FakeTdClients; records the chats joined."""

    def __init__(self, scenario: dict[str, Any] = JOIN_SCENARIO) -> None:
        self.scenario = scenario
        self.clients: list[FakeTdClient] = []

    def __call__(self) -> tuple[str, FakeTdClient]:
        self.clients.append(FakeTdClient(self.scenario))
        return "+34600000001", self.clients[-1]

    @property
    def joined(self) -> list[int]:
        return [i for c in self.clients for i in c.joined]


def with_telegram(db_url: str, telegram: Any, calls: Calls, every: int = 0) -> list[Any]:
    server = mcp.build_server(
        db_url, FakeEmbedder(), telegram=telegram, default_years=2, sync_every_minutes=every
    )
    return anyio.run(_run, Client(server), calls)


def enabled_chats(conn: Conn) -> dict[int, tuple[Any, ...]]:
    rows = conn.execute("SELECT chat_id, kind, enabled, account_phone, backfill_years FROM chat")
    return {int(cast(int, r[0])): r[1:] for r in rows}


def test_add_chat_joins_and_enables_a_public_group(conn: Conn, db_url: str) -> None:
    joiner = Joiner()
    first, again = with_telegram(db_url, joiner, [
        ("add_chat", {"link": "https://t.me/valencia_new"}),
        ("add_chat", {"link": "@valencia_new", "years": 1}),
    ])  # fmt: skip
    sync = first.pop("sync")
    assert first == {"chat_id": NEW, "title": f"chat {NEW}", "kind": "supergroup",
                     "username": "valencia_new", "backfill_years": 2}  # fmt: skip
    assert (
        sync["estimated_messages"] == 0 and sync["next_run_at"] is None
    )  # no history, no schedule
    assert again["backfill_years"] == 1
    assert enabled_chats(conn) == {NEW: ("supergroup", True, "+34600000001", 1)}
    assert joiner.joined == [NEW, NEW]
    assert all(c._closed for c in joiner.clients)  # every TDLib client is closed


def test_add_chat_refuses_private_chats_basic_groups_and_pending_joins(
    conn: Conn, db_url: str
) -> None:
    joiner = Joiner()
    results = with_telegram(db_url, joiner, [
        ("add_chat", {"link": "https://t.me/some_person"}),
        ("add_chat", {"link": "https://t.me/+basicGroupInvite"}),
        ("add_chat", {"link": "https://t.me/+askTheAdminsFirst"}),
        ("add_chat", {"link": "https://t.me/ask_first"}),
        ("add_chat", {"link": "https://t.me/nobody_here"}),
        ("add_chat", {"link": "https://example.com/x"}),
        ("add_chat", {"link": "@valencia_new", "years": 0}),
    ])  # fmt: skip
    assert [r["error"]["code"] for r in results] == [
        "not_allowed", "not_allowed", "join_pending", "join_pending", "not_found",
        "bad_argument", "bad_argument",
    ]  # fmt: skip
    assert "private" in results[0]["error"]["message"]
    assert "basic_group" in results[1]["error"]["message"]
    assert enabled_chats(conn) == {} and joiner.joined == []


def test_add_chat_refuses_out_of_range_years_before_joining(conn: Conn, db_url: str) -> None:
    """#38: a depth that `years_before` or the int4 column cannot hold is refused before
    TDLib is opened, the row keeps its value and the next sync still runs."""
    joiner = Joiner()
    with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new", "years": 2})])
    bad = [0, -1, 101, 10000, 9999999999, 2**63, 2**64]
    results = with_telegram(db_url, joiner, [
        ("add_chat", {"link": "@valencia_new", "years": y}) for y in bad
    ])  # fmt: skip
    assert [r["error"]["code"] for r in results] == ["bad_argument"] * len(bad)
    assert results[3]["error"]["message"] == "years must be between 1 and 100"
    assert len(joiner.clients) == 1 and joiner.joined == [NEW]  # TDLib not opened again
    assert enabled_chats(conn) == {NEW: ("supergroup", True, "+34600000001", 2)}
    assert sync(conn, FakeTdClient(JOIN_SCENARIO)).status == "ok"
    for years in (1, 100):  # the accepted bounds
        (result,) = with_telegram(db_url, joiner, [
            ("add_chat", {"link": "@valencia_new", "years": years})
        ])  # fmt: skip
        assert result["backfill_years"] == years
        assert enabled_chats(conn)[NEW][3] == years
        assert sync(conn, FakeTdClient(JOIN_SCENARIO)).status == "ok"


def test_add_chat_after_an_approved_join_request(conn: Conn, db_url: str) -> None:
    """#39: the retry after the admins approve enables the chat."""
    scenario = json.loads(json.dumps(JOIN_SCENARIO))
    joiner = Joiner(scenario)
    link = "https://t.me/+askTheAdminsFirst"
    (pending,) = with_telegram(db_url, joiner, [("add_chat", {"link": link})])
    assert pending["error"]["code"] == "join_pending" and enabled_chats(conn) == {}
    scenario["invites"][1]["request"] = False  # approved: the same link now joins
    (added,) = with_telegram(db_url, joiner, [("add_chat", {"link": link})])
    assert added["chat_id"] == PENDING and added["kind"] == "channel"
    assert enabled_chats(conn) == {PENDING: ("channel", True, "+34600000001", None)}
    assert joiner.joined == [PENDING]


def test_add_chat_refuses_a_chat_enabled_under_another_account(conn: Conn, db_url: str) -> None:
    """Antony, 2026-10-09: no join from the second account and no change to the row."""
    other = "+34600000002"
    chats.enable(conn, Chat(NEW, "valencia", "supergroup", "valencia_new"), other, 5)
    before = enabled_chats(conn)
    joiner = Joiner()
    results = with_telegram(db_url, joiner, [
        ("add_chat", {"link": "@valencia_new"}),
        ("add_chat", {"link": "https://t.me/valencia_new", "years": 1}),
        ("add_chat", {"link": str(NEW)}),
    ])  # fmt: skip
    assert [r["error"]["code"] for r in results] == ["not_allowed"] * 3
    assert "already enabled under another account" in results[0]["error"]["message"]
    assert other not in json.dumps(results)  # the other account's phone is not shown
    assert joiner.joined == [] and all(c._closed for c in joiner.clients)
    assert enabled_chats(conn) == before == {NEW: ("supergroup", True, other, 5)}
    title = conn.execute("SELECT title FROM chat WHERE chat_id = %s", (NEW,)).fetchone()
    assert title == ("valencia",)
    assert chats.disable(conn, NEW)  # disabled there: this account may take it over
    (added,) = with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new"})])
    assert added["chat_id"] == NEW and joiner.joined == [NEW]
    assert enabled_chats(conn) == {NEW: ("supergroup", True, "+34600000001", 5)}


def history(faults: list[dict[str, Any]], msgs: int = 150) -> dict[str, Any]:
    """JOIN_SCENARIO with `msgs` messages in NEW, a cold first page of 10, and `faults`."""
    scenario = json.loads(json.dumps(JOIN_SCENARIO))
    scenario["chats"][0] = chat(NEW, messages(1, msgs, timedelta(hours=1)),
                                username="valencia_new", page_sizes=[10])  # fmt: skip
    scenario["faults"] = faults
    return scenario


def added(db_url: str, scenario: dict[str, Any]) -> tuple[dict[str, Any], Joiner]:
    joiner = Joiner(scenario)
    (result,) = with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new"})], 60)
    assert result["chat_id"] == NEW, result  # enabled whatever the estimate says
    return result["sync"], joiner


def test_add_chat_estimates_from_two_history_pages(conn: Conn, db_url: str) -> None:
    sync, joiner = added(db_url, history([]))
    assert [c[1] for c in joiner.clients[0].history_calls] == [0, 141 << 20]  # cold, then more
    assert sync["estimated_messages"] > 0 and sync["complete_by"] is not None
    assert sync["next_run_at"] is not None and sync["searchable_from"] is not None


def test_add_chat_empty_chat_is_complete_with_the_first_run(conn: Conn, db_url: str) -> None:
    sync, _ = added(db_url, history([], msgs=0))
    assert (sync["estimated_messages"], sync["estimated_minutes"]) == (0, 0)
    start = datetime.fromisoformat(sync["next_run_at"])
    assert datetime.fromisoformat(sync["complete_by"]) == start + timedelta(minutes=5)


@pytest.mark.parametrize(
    "faults",
    [
        [{"call": 1, "error": "boom"}],  # the probe fails
        [{"call": 1, "flood_wait": 30}],
        [{"call": 2, "error": "boom"}],  # partial: page 1 read, page 2 fails
        [{"call": 2, "flood_wait": 30}],
    ],
)
def test_add_chat_failed_history_probe_is_unknown_not_empty(
    conn: Conn, db_url: str, faults: list[dict[str, Any]]
) -> None:
    """#42: the volume and the end are null; the next run and the newest messages are not."""
    sync, joiner = added(db_url, history(faults))
    assert len(joiner.clients[0].history_calls) == faults[0]["call"]
    assert sync["estimated_messages"] is None and sync["estimated_minutes"] is None
    assert sync["complete_by"] is None
    start = datetime.fromisoformat(sync["next_run_at"])
    assert datetime.fromisoformat(sync["searchable_from"]) == start + timedelta(minutes=5)
    assert enabled_chats(conn) == {NEW: ("supergroup", True, "+34600000001", None)}


def test_add_chat_estimate_failure_after_enable_still_succeeds(conn: Conn, db_url: str) -> None:
    """#41: the chat is enabled before the estimate, so an estimate that raises (here
    `next_run` overflowing on an absurd schedule that skipped `Settings`) gives a null
    `sync` block, not an error that would hide the enabled chat."""
    joiner = Joiner(history([]))
    (result,) = with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new"})], 10**12)
    assert result["chat_id"] == NEW and joiner.joined == [NEW]
    assert result["sync"] == dict.fromkeys(
        ["next_run_at", "estimated_messages", "estimated_minutes", "searchable_from",
         "complete_by"]
    )  # fmt: skip
    assert enabled_chats(conn) == {NEW: ("supergroup", True, "+34600000001", None)}


def test_add_chat_waits_for_a_running_sync(conn: Conn, db_url: str) -> None:
    joiner = Joiner()
    with psycopg.connect(db_url) as other, ingest.sync_lock(other):
        (busy,) = with_telegram(db_url, joiner, [("add_chat", {"link": "@valencia_new"})])
    assert busy["error"]["code"] == "busy"
    assert joiner.clients == [] and enabled_chats(conn) == {}  # TDLib was not even opened


def test_add_chat_without_a_telegram_account(conn: Conn, db_url: str) -> None:
    (result,) = with_telegram(db_url, None, [("add_chat", {"link": "@valencia_new"})])
    assert result["error"]["code"] == "not_configured"

    def unconfigured() -> tuple[str, FakeTdClient]:
        raise td.SetupError("set MCP_ACCOUNT: there is no logged-in account or several")

    (result,) = with_telegram(db_url, unconfigured, [("add_chat", {"link": "@valencia_new"})])
    assert result["error"] == {
        "code": "not_configured",
        "message": "set MCP_ACCOUNT: there is no logged-in account or several",
    }
