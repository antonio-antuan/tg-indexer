"""The reference query (SPEC §5.7, A9), run by a real MCP client over HTTP.

It does what the agent would: `list_tags` → `search` with the reference queries, the
`valencia` tag and the last 30 days → `get_context` on the top hit, then checks the links.
`tgi selftest` runs it against the configured server; `tgi selftest --scenario FILE` loads
the scenario into a throwaway database, serves it on a loopback port and also checks the
scenario's expectations. Reports carry counts and ids only, never message text.
"""

import json
import logging
import re
import secrets
import signal
import socket
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import anyio
import httpx2
import psycopg
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from psycopg import sql

from tgi import chats, db, ingest, mcp
from tgi.config import Settings
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient

QUERIES = ["продление ВНЖ", "ВНЖ renovación", "renovación TIE", "residence permit renewal"]
TAG = "valencia"
DAYS = 30
MIN_SEEDED = 3  # A9: at least this many of the seeded messages among the hits
ACCOUNT = "+34000000000"  # the scenario's account; it never reaches TDLib
LINK = re.compile(r"https://t\.me/(c/\d+|[A-Za-z]\w{3,31})/\d+")


class Failed(Exception):
    """A check did not hold; the message has counts and ids only."""


class Unreachable(Exception):
    """The query could not run (no connection, an HTTP error status); the message names the
    cause, never the token or the URL."""


@dataclass
class Outcome:
    tags: int
    hits: list[dict[str, Any]]
    context: list[dict[str, Any]]

    def links(self) -> list[list[str]]:
        """The message links of each hit, in rank order."""
        return [[m["link"] for m in h["messages"] if m["link"]] for h in self.hits]


def _value(result: Any, tool: str) -> Any:
    if result.is_error:
        text = result.content[0].text
        code = json.loads(text.partition(": ")[2]).get("code") if ": {" in text else None
        raise Failed(f"{tool} returned an error (code {code})")
    value = result.structured_content
    return value["result"] if isinstance(value, dict) and set(value) == {"result"} else value


async def _query(url: str, token: str, today: date, statuses: list[int]) -> Outcome:
    since = (today - timedelta(days=DAYS)).isoformat()
    # The server is on loopback or a Docker bridge: no proxy from the environment.
    headers = {"Authorization": f"Bearer {token}"}

    async def status(response: httpx2.Response) -> None:
        statuses.append(response.status_code)

    hooks = {"response": [status]}
    async with (
        httpx2.AsyncClient(headers=headers, trust_env=False, timeout=60, event_hooks=hooks) as http,
        Client(streamable_http_client(url, http_client=http)) as client,
    ):
        tags = _value(await client.call_tool("list_tags", {}), "list_tags")
        if not any(t["tag"] == TAG and t["status"] == "confirmed" for t in tags):
            raise Failed(f"no chat has the confirmed tag {TAG}")
        args = {"queries": QUERIES, "tags": [TAG], "since": since}
        hits = _value(await client.call_tool("search", args), "search")
        if not hits:
            raise Failed("search returned no hits")
        top = hits[0]
        args = {"chat_id": top["chat_id"], "message_id": top["messages"][0]["message_id"]}
        context = _value(await client.call_tool("get_context", args), "get_context")
    return Outcome(len(tags), hits, context)


def check_links(outcome: Outcome) -> None:
    """Every link is a t.me message link, and get_context gives the top message the link
    search gave it."""
    messages = [m for h in outcome.hits for m in h["messages"]] + outcome.context
    bad = sum(1 for m in messages if m["link"] is not None and not LINK.fullmatch(m["link"]))
    if bad:
        raise Failed(f"{bad} links are not t.me message links")
    top = outcome.hits[0]["messages"][0]
    same = [m for m in outcome.context if m["message_id"] == top["message_id"]]
    if len(same) != 1 or same[0]["link"] != top["link"]:
        raise Failed("get_context does not return the top message with its link")


def check_expected(outcome: Outcome, expect: dict[str, Any]) -> dict[str, int]:
    """A9: enough seeded messages among the hits; no excluded one (another tag, older than
    the date filter); every unrelated-topic one ranks below every seeded hit (the hybrid
    semantic leg has no relevance cut-off, so it may return them, but last)."""
    ranks: dict[str, int] = {}
    for rank, links in enumerate(outcome.links()):
        for link in links:
            ranks.setdefault(link, rank)
    seeded = [ranks[link] for link in expect["seeded"] if link in ranks]
    if len(seeded) < MIN_SEEDED:
        raise Failed(f"{len(seeded)} of {len(expect['seeded'])} seeded messages found")
    excluded = sum(1 for link in expect["excluded"] if link in ranks)
    if excluded:
        raise Failed(f"{excluded} excluded messages returned")
    unrelated = [ranks[link] for link in expect["unrelated"] if link in ranks]
    above = sum(1 for rank in unrelated if rank <= max(seeded))
    if above:
        raise Failed(f"{above} unrelated messages rank with or above a seeded hit")
    return {"seeded": len(seeded), "unrelated_returned": len(unrelated)}


# The MCP SDK's client logs every message it sends at DEBUG, the queries included, and the
# HTTP client logs each request's URL at INFO; this process is the client.
_CLIENT_LEVELS = {
    "mcp.client": logging.INFO,
    "httpx": logging.WARNING,
    "httpx2": logging.WARNING,
    "httpcore": logging.WARNING,
    "httpcore2": logging.WARNING,
}


@contextmanager
def _quiet_client() -> Iterator[None]:
    """The client loggers stay at `_CLIENT_LEVELS` or above meanwhile, then get their own
    levels back, on failure too."""
    loggers = {logging.getLogger(name): level for name, level in _CLIENT_LEVELS.items()}
    saved = {logger: logger.level for logger in loggers}
    try:
        for logger, level in loggers.items():
            if logger.getEffectiveLevel() < level:
                logger.setLevel(level)
        yield
    finally:
        for logger, level in saved.items():
            logger.setLevel(level)


def _leaves(exc: BaseException) -> Iterable[BaseException]:
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def _cause(exc: Exception, statuses: list[int]) -> BaseException:
    """What ended the query: the client runs it in a task group, so a failed check or a
    transport error arrives wrapped in an ExceptionGroup, and the SDK reports an HTTP error
    status as a bare `MCPError`; the statuses seen tell which one it was."""
    leaves = list(_leaves(exc))
    for leaf in leaves:
        if isinstance(leaf, Failed):
            return leaf
    errors = [s for s in statuses if s >= 400]
    if 401 in errors:
        return Unreachable("the server refused the token (HTTP 401); check MCP_TOKEN")
    if errors:
        return Unreachable(f"the server answered HTTP {errors[0]}")
    if any(isinstance(leaf, httpx2.ConnectError) for leaf in leaves):
        return Unreachable("cannot connect to the MCP server")
    if any(isinstance(leaf, httpx2.TimeoutException) for leaf in leaves):
        return Unreachable("the MCP server did not answer in time")
    return leaves[0] if len(leaves) == 1 else exc


def run(url: str, token: str, today: date | None = None) -> Outcome:
    """The reference query against the server at `url`, with the link checks."""
    statuses: list[int] = []
    try:
        with _quiet_client():
            outcome = anyio.run(_query, url, token, today or datetime.now(UTC).date(), statuses)
    except Exception as exc:
        cause = _cause(exc, statuses)
        if cause is exc:
            raise
        raise cause from None
    check_links(outcome)
    return outcome


# --- the scenario: a throwaway database and a loopback server ---------------------------------


def _shifted(scenario: dict[str, Any], now: datetime) -> dict[str, Any]:
    """The scenario with its dates moved so that its `reference.now` is `now`."""
    delta = now - datetime.fromisoformat(scenario["reference"]["now"])

    def move(value: str) -> str:
        return (datetime.fromisoformat(value) + delta).isoformat()

    out: dict[str, Any] = json.loads(json.dumps(scenario))  # a deep copy
    for chat in out["chats"]:
        for m in chat.get("messages", []):
            m["date"] = move(m["date"])
            if m.get("edit_date"):
                m["edit_date"] = move(m["edit_date"])
    return out


PREFIX = "tgi_selftest_"
_created: set[str] = set()  # the databases this process created, the only ones it drops


def _maintenance(url: str) -> str:
    """The server's `postgres` database: CREATE and DROP run there, not in the archive."""
    return psycopg.conninfo.make_conninfo(url, dbname="postgres")


def drop_database(url: str, name: str) -> None:
    """Drop `name`, a database `_database` created (or was creating) in this process;
    anything else (the archive above all) is refused with ValueError before Postgres is
    asked. A database that does not exist is no error."""
    archive = psycopg.conninfo.conninfo_to_dict(url).get("dbname")
    if name not in _created or not name.startswith(PREFIX) or name == archive:
        raise ValueError("refusing to drop a database this selftest did not create")
    with psycopg.connect(_maintenance(url), autocommit=True) as admin:
        drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)")
        admin.execute(drop.format(sql.Identifier(name)))
    _created.discard(name)


@contextmanager
def _database(url: str) -> Iterator[str]:
    """A new empty `tgi_selftest_<random>` database on the server of `url`, dropped
    afterwards, on failure too. The archive database is never opened. Needs CREATEDB.
    The name is registered and the guarded block entered before the CREATE, so a signal
    that lands just after it still drops the database."""
    name = f"{PREFIX}{secrets.token_hex(8)}"
    _created.add(name)
    try:
        with psycopg.connect(_maintenance(url), autocommit=True) as admin:
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        yield psycopg.conninfo.make_conninfo(url, dbname=name)
    finally:
        drop_database(url, name)


def _load(url: str, scenario: dict[str, Any], settings: Settings) -> None:
    """Migrate, enable every chat, sync and embed with the fake provider, confirm the tags."""
    td = FakeTdClient(scenario)
    with db.connect(url) as conn:
        db.migrate(conn)
        for chat in scenario["chats"]:
            chats.enable(conn, td.get_chat(int(chat["chat_id"])), ACCOUNT)
        result = ingest.sync(
            conn, lambda _: td, replace(settings, page_delay_ms=0), embedder=FakeEmbedder()
        )
        if result.status != "ok":
            raise Failed(f"scenario sync ended {result.status}")
        for tag, chat_ids in scenario["reference"]["tags"].items():
            for chat_id in chat_ids:
                known = chats.propose_tag(conn, chat_id, tag, "owner")
                if not known or not chats.confirm_tag(conn, chat_id, tag):
                    raise Failed(f"the scenario tags chat {chat_id}, which it does not have")


@contextmanager
def _serve(url: str) -> Iterator[tuple[str, str]]:
    """`tgi mcp --http` in a thread on 127.0.0.1 and a free port, with a fresh token."""
    import uvicorn

    token = mcp.new_token()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    app = mcp.http_app(
        mcp.build_server(url, FakeEmbedder()), token, mcp.transport_security("127.0.0.1", port, [])
    )
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", access_log=False))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise Failed("the test server did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}/mcp", token
    finally:
        server.should_exit = True
        thread.join(30)
        sock.close()


@contextmanager
def _sigterm_exits() -> Iterator[None]:
    """SIGTERM raises SystemExit, as SIGINT raises KeyboardInterrupt, so the throwaway
    database is dropped on the way out (Python's default dies without unwinding). Only the
    main thread can set a handler; elsewhere the default stays."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def exit_143(signum: int, frame: Any) -> None:
        raise SystemExit(128 + signal.SIGTERM)

    previous = signal.signal(signal.SIGTERM, exit_143)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_scenario(
    database_url: str, scenario: dict[str, Any], settings: Settings
) -> tuple[Outcome, dict[str, int]]:
    """The reference query over the scenario, with its A9 expectations checked."""
    now = datetime.now(UTC)
    with _sigterm_exits(), _database(database_url) as url:
        _load(url, _shifted(scenario, now), settings)
        with _serve(url) as (mcp_url, token):
            outcome = run(mcp_url, token, now.date())
    return outcome, check_expected(outcome, scenario["reference"])
