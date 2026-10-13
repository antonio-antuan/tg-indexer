"""Tool annotations, `MCP_READ_ONLY` and `tgi chats add` (assistant safety)."""

import json
import sys
from pathlib import Path
from typing import Any

import anyio
import httpx2
import psycopg
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client
from test_mcp import (
    ASKING,
    BASE,
    BASIC,
    JOIN_SCENARIO,
    LOOPBACK,
    NEW,
    TOKEN,
    Joiner,
    _run,
    enabled_chats,
    outcome,
    with_telegram,
)
from typer.testing import CliRunner

from tgi import ingest, mcp
from tgi.cli import app
from tgi.config import Settings
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient
from tgi.td_port import FloodWait, TdError

Conn = psycopg.Connection[tuple[object, ...]]
READS = {"list_chats", "list_tags", "search", "get_thread", "get_context"}
WRITES = {"suggest_tags", "add_chat"}
ACCOUNT = "+34600000001"


def listed(server: Any) -> dict[str, dict[str, Any]]:
    """Tool name -> its annotations as a client receives them (camelCase JSON)."""

    async def go() -> dict[str, dict[str, Any]]:
        async with Client(server) as c:
            tools = (await c.list_tools()).tools
        return {
            t.name: (t.annotations.model_dump(by_alias=True, exclude_none=True)
                     if t.annotations else {})
            for t in tools
        }  # fmt: skip

    return anyio.run(go)


# --- annotations --------------------------------------------------------------------------------


def test_reads_are_marked_read_only_and_writes_are_not(db_url: str) -> None:
    tools = listed(mcp.build_server(db_url, FakeEmbedder()))
    assert set(tools) == READS | WRITES
    for name in READS:
        assert tools[name] == {"readOnlyHint": True, "openWorldHint": False}, name
    assert tools["suggest_tags"]["readOnlyHint"] is False
    assert tools["suggest_tags"]["openWorldHint"] is False
    # add_chat changes the Telegram account: not read-only, reaches outside the archive
    assert tools["add_chat"] == {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }


# --- MCP_READ_ONLY ------------------------------------------------------------------------------


def test_a_read_only_server_does_not_offer_or_run_the_write_tools(conn: Conn, db_url: str) -> None:
    joiner = Joiner()
    server = mcp.build_server(db_url, FakeEmbedder(), telegram=joiner, read_only=True)
    assert set(listed(server)) == READS
    calls = [
        ("add_chat", {"link": "@valencia_new"}),
        ("suggest_tags", {"suggestions": [{"chat_id": NEW, "tag": "x"}]}),
        ("list_tags", {}),
    ]
    joined, suggested, tags = anyio.run(_run, Client(server), calls)
    assert joined["error"]["code"] == "unknown_tool"
    assert suggested["error"]["code"] == "unknown_tool"
    assert tags == []  # reads still work
    assert joiner.clients == [] and enabled_chats(conn) == {}  # Telegram was never opened


def test_the_default_server_keeps_the_write_tools(conn: Conn, db_url: str) -> None:
    (joined,) = with_telegram(db_url, Joiner(), [("add_chat", {"link": "@valencia_new"})])
    assert joined["chat_id"] == NEW


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", False), ("0", False), ("false", False), ("No", False), ("off", False),
     ("1", True), ("true", True), (" YES ", True), ("on", True)],
)  # fmt: skip
def test_mcp_read_only_setting(raw: str, expected: bool) -> None:
    assert Settings.from_env({"MCP_READ_ONLY": raw}).mcp_read_only is expected
    assert Settings.from_env({}).mcp_read_only is False


@pytest.mark.parametrize("raw", ["2", "enabled", "y", "readonly"])
def test_an_unrecognised_mcp_read_only_is_an_error_not_a_guess(raw: str) -> None:
    with pytest.raises(ValueError, match="MCP_READ_ONLY"):
        Settings.from_env({"MCP_READ_ONLY": raw})


def test_cli_mcp_stdio_honours_mcp_read_only(db_url: str, tmp_path: Path) -> None:
    async def names(env: dict[str, str]) -> set[str]:
        params = StdioServerParameters(
            command=str(Path(sys.executable).parent / "tgi"),
            args=["mcp", "--stdio"],
            env={"DATABASE_URL": db_url, "EMBED_PROVIDER": "fake", **env},
            cwd=tmp_path,
        )
        async with Client(params) as c:
            return {t.name for t in (await c.list_tools()).tools}

    assert anyio.run(names, {"MCP_READ_ONLY": "1"}) == READS
    assert anyio.run(names, {}) == READS | WRITES


def test_cli_mcp_refuses_a_bad_mcp_read_only(
    db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("MCP_READ_ONLY", "maybe")
    result = CliRunner().invoke(app, ["mcp", "--stdio"])
    assert result.exit_code == 1
    assert "MCP_READ_ONLY" in result.output


# --- tgi chats add -------------------------------------------------------------------------------


@pytest.fixture
def scenario(tmp_path: Path, db_url: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    path = tmp_path / "join.json"
    path.write_text(json.dumps(JOIN_SCENARIO))
    return path


def chats_add(scenario: Path, *args: str) -> Any:
    return CliRunner().invoke(
        app, ["chats", "add", *args, "--account", ACCOUNT, "--scenario", str(scenario)]
    )


def test_chats_add_joins_and_enables_a_public_group(conn: Conn, scenario: Path) -> None:
    result = chats_add(scenario, "https://t.me/valencia_new", "--years", "2")
    assert result.exit_code == 0, result.output
    assert f"enabled {NEW} (supergroup, 2 years)" in result.output
    assert enabled_chats(conn) == {NEW: ("supergroup", True, ACCOUNT, 2)}


@pytest.mark.parametrize(
    ("link", "message"),
    [
        ("https://t.me/+basicGroupInvite", "not a basic_group chat"),
        ("@some_person", "not a private chat"),
        ("https://t.me/ask_first", "join request"),
        ("@nobody_here", "no chat this account can join"),
    ],
)
def test_chats_add_refusals_exit_1_and_enable_nothing(
    conn: Conn, scenario: Path, link: str, message: str
) -> None:
    result = chats_add(scenario, link)
    assert result.exit_code == 1, result.output
    assert message in result.output
    assert enabled_chats(conn) == {}


@pytest.mark.parametrize(
    ("args", "where"), [(["hello world"], "LINK"), (["@valencia_new", "--years", "0"], "--years")]
)
def test_chats_add_bad_arguments_exit_1_before_tdlib(
    conn: Conn, scenario: Path, args: list[str], where: str
) -> None:
    result = chats_add(scenario, *args)
    assert result.exit_code == 1
    assert where in result.output
    assert enabled_chats(conn) == {}


def test_chats_add_waits_for_a_running_sync(conn: Conn, scenario: Path, db_url: str) -> None:
    with psycopg.connect(db_url) as other, ingest.sync_lock(other):
        result = chats_add(scenario, "@valencia_new")
    assert result.exit_code == 2
    assert "try again" in result.output
    assert enabled_chats(conn) == {}


def test_chats_add_refuses_a_chat_held_by_another_account(conn: Conn, scenario: Path) -> None:
    assert chats_add(scenario, "@valencia_new").exit_code == 0
    other = CliRunner().invoke(
        app, ["chats", "add", "@valencia_new", "--account", "+34600000002",
              "--scenario", str(scenario)],
    )  # fmt: skip
    assert other.exit_code == 1
    assert enabled_chats(conn)[NEW][2] == ACCOUNT


def test_asking_channel_is_not_enabled_by_mcp_either(conn: Conn, db_url: str) -> None:
    # the CLI and the tool share `join_chat`: the same pending join, the same outcome
    (result,) = with_telegram(db_url, Joiner(), [("add_chat", {"link": "@ask_first"})])
    assert result["error"]["code"] == "join_pending"
    assert ASKING not in enabled_chats(conn) and BASIC not in enabled_chats(conn)


def lock_is_free(db_url: str) -> bool:
    with psycopg.connect(db_url) as other:
        try:
            with ingest.sync_lock(other):
                return True
        except ingest.SyncLocked:
            return False


def test_chats_add_takes_a_negative_chat_id(conn: Conn, scenario: Path) -> None:
    result = chats_add(scenario, str(NEW))  # -100…: not an option
    assert result.exit_code == 0, result.output
    assert NEW in enabled_chats(conn)


def test_chats_add_joins_through_an_invite_link(
    conn: Conn, tmp_path: Path, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    invites = [*JOIN_SCENARIO["invites"], {"link": "https://t.me/+okInvite", "chat_id": NEW}]
    path = tmp_path / "invite.json"
    path.write_text(json.dumps({**JOIN_SCENARIO, "invites": invites}))
    result = chats_add(path, "https://t.me/+okInvite")
    assert result.exit_code == 0, result.output
    assert NEW in enabled_chats(conn)


@pytest.mark.parametrize(
    ("error", "message"),
    [(FloodWait(30), "Telegram asks to wait 30s"), (TdError(500, "boom"), "TdError 500")],
)
def test_chats_add_tdlib_failures_exit_3_and_release_the_lock(
    conn: Conn,
    scenario: Path,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(FakeTdClient, "join_chat", fail)
    result = chats_add(scenario, "@valencia_new")
    assert result.exit_code == 3, result.output
    assert message in result.output
    assert "boom" not in result.output  # TDLib text is never printed
    assert enabled_chats(conn) == {} and lock_is_free(db_url)


def test_chats_add_without_a_usable_account_exits_1_and_releases_the_lock(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))  # no account logged in here
    result = CliRunner().invoke(app, ["chats", "add", "@valencia_new", "--account", ACCOUNT])
    assert result.exit_code == 1, result.output
    assert enabled_chats(conn) == {} and lock_is_free(db_url)


def test_a_read_only_server_over_http(conn: Conn, db_url: str) -> None:
    joiner = Joiner()

    async def go() -> tuple[set[str], Any]:
        server = mcp.build_server(db_url, FakeEmbedder(), telegram=joiner, read_only=True)
        asgi = mcp.http_app(server, TOKEN, LOOPBACK)  # creates the session manager
        async with (
            server.session_manager.run(),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=asgi),
                base_url=BASE,
                headers={"Authorization": f"Bearer {TOKEN}"},
            ) as http,
            Client(streamable_http_client(f"{BASE}/mcp", http_client=http)) as c,
        ):
            tools = (await c.list_tools()).tools
            assert all(t.annotations and t.annotations.read_only_hint for t in tools)
            joined = outcome(await c.call_tool("add_chat", {"link": "@valencia_new"}))
            return {t.name for t in tools}, joined

    names, joined = anyio.run(go)
    assert names == READS
    assert joined["error"]["code"] == "unknown_tool"
    assert joiner.clients == [] and enabled_chats(conn) == {}
