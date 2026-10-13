"""The MCP server: tools over search, tags and the allowlist, stdio and bearer-token HTTP
(SPEC §5.6).

Built on the official `mcp` SDK; its 2.x `MCPServer` is the 1.x `FastMCP` renamed. Each
tool call opens its own database connection. Tool errors are MCP tool errors (`isError`)
whose text, after the SDK's `Error executing tool <name>: `, is a JSON object
`{"code": ..., "message": ...}`; the SDK logs that text, so a message names the argument
at fault but never quotes the caller's values (A10, issue #26).
"""

import contextlib
import hmac
import ipaddress
import json
import logging
import re
import secrets
import shutil
import subprocess
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Any, Literal, NoReturn, cast

import psycopg
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp_types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import BaseModel, BeforeValidator, Strict, ValidationError

from tgi import chats, db, eta, ingest, search, status, td
from tgi.config import Settings, check_years
from tgi.embed import Embedder
from tgi.ingest import Conn
from tgi.td_port import (
    Chat,
    ChatLink,
    FloodWait,
    JoinPending,
    Message,
    NotJoinable,
    TdClient,
    TdError,
    parse_link,
)

log = logging.getLogger(__name__)

MAX_THREAD = search.MAX_LIMIT  # get_thread messages
MAX_CONTEXT = 100  # get_context: before + after, and the messages returned
MAX_DEPTH = 1000  # reply-chain steps followed, a guard against a cycle in bad data
TOKEN_BYTES = 32

# ASGI, typed loosely: the SDK's Starlette app is the only one wrapped.
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def new_token() -> str:
    """A fresh `MCP_TOKEN`: 32 random bytes, URL-safe base64 (43 characters)."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def _fail(code: str, message: str) -> NoReturn:
    """A tool error. `message` must not quote the caller's arguments: the SDK logs it at
    INFO (A10), so it says which argument is wrong, not what it held."""
    raise ToolError(json.dumps({"code": code, "message": message}))


# A JSON integer: pydantic would otherwise take `true` as 1 and "5" as 5.
Int = Annotated[int, Strict()]
Bool = Annotated[bool, Strict()]  # a JSON boolean, not "yes" or 1


def _when(value: object) -> object:
    """`YYYY-MM-DD` stays a date (a whole UTC day); anything with a time is a date-time
    (`search.parse_when`). Parsed here because pydantic's `date | datetime` turns a midnight
    date-time into a date, and would take a number as a Unix time."""
    if isinstance(value, str):
        return search.parse_when(value)
    if isinstance(value, date):  # Python callers
        return value
    raise ValueError("a date must be a YYYY-MM-DD or ISO date-time string")


# A date or a date-time bound, as `search` and `recent` take them.
When = Annotated[date | datetime, BeforeValidator(_when)]


class Suggestion(BaseModel):
    chat_id: Int
    tag: str
    reason: str = ""


# --- tools, as plain functions over a connection ----------------------------------------------


def _check_chats(conn: Conn, chat_ids: list[int]) -> None:
    rows = conn.execute("SELECT chat_id FROM chat WHERE chat_id = ANY(%s)", (chat_ids,))
    missing = sorted(set(chat_ids) - {cast(int, r[0]) for r in rows})
    if missing:
        _fail(
            "unknown_chat", f"{len(missing)} of the chat ids are not in the archive; see list_chats"
        )


def _check_tags(conn: Conn, tags: list[str]) -> None:
    """A tag is known when any chat has it, proposed or confirmed."""
    rows = conn.execute("SELECT DISTINCT tag FROM chat_tag WHERE tag = ANY(%s)", (tags,))
    missing = sorted(set(tags) - {cast(str, r[0]) for r in rows})
    if missing:
        _fail("unknown_tag", f"{len(missing)} of the tags are not known; see list_tags")


def list_chats(conn: Conn, tag: str | None = None) -> list[dict[str, Any]]:
    """Every chat in the archive with its confirmed tags; `tag` keeps the chats it is
    confirmed for. Counts and dates cover messages not tombstoned."""
    if tag is not None:
        _check_tags(conn, [tag])
    rows = conn.execute(
        "SELECT c.chat_id, c.title, c.kind,"
        " ARRAY(SELECT t.tag FROM chat_tag t WHERE t.chat_id = c.chat_id"
        "       AND t.status = 'confirmed' ORDER BY t.tag),"
        " s.n, s.first_date, s.last_date"
        " FROM chat c, LATERAL (SELECT count(*) AS n, min(date) AS first_date,"
        "  max(date) AS last_date FROM message m"
        "  WHERE m.chat_id = c.chat_id AND m.deleted_at IS NULL) s"
        " WHERE %(tag)s::text IS NULL OR EXISTS (SELECT 1 FROM chat_tag t"
        "  WHERE t.chat_id = c.chat_id AND t.tag = %(tag)s AND t.status = 'confirmed')"
        " ORDER BY c.title, c.chat_id",
        {"tag": tag},
    )
    return [
        {
            "chat_id": r[0],
            "title": r[1],
            "kind": r[2],
            "tags": r[3],
            "message_count": r[4],
            "first_date": _iso(r[5]),
            "last_date": _iso(r[6]),
        }
        for r in rows
    ]


def list_tags(conn: Conn) -> list[dict[str, Any]]:
    """Each tag with each status it has and how many chats have it so."""
    rows = conn.execute(
        "SELECT tag, status, count(*) FROM chat_tag GROUP BY tag, status ORDER BY tag, status"
    )
    return [{"tag": r[0], "status": r[1], "chat_count": r[2]} for r in rows]


def run_search(
    conn: Conn,
    queries: list[str],
    tags: list[str] | None = None,
    chat_ids: list[int] | None = None,
    since: date | None = None,  # a datetime too: a `date` is a whole UTC day
    until: date | None = None,
    mode: search.Mode = "hybrid",
    limit: int = 20,
    embedder: Embedder | None = None,
    compact: bool = False,
) -> list[dict[str, Any]]:
    if not 1 <= limit <= search.MAX_LIMIT:
        _fail("bad_range", f"limit must be between 1 and {search.MAX_LIMIT}")
    if tags:
        _check_tags(conn, tags)
    if chat_ids:
        _check_chats(conn, chat_ids)
    try:
        hits = search.search(
            conn,
            queries,
            tags=tags or None,  # an empty list filters nothing, as when it is left out
            chat_ids=chat_ids or None,
            since=since,
            until=until,
            mode=mode,
            limit=limit,
            embedder=embedder,
            compact=compact,
        )
    except search.BadRange as exc:
        _fail("bad_range", str(exc))
    except search.ModelMismatch as exc:
        _fail("model_mismatch", str(exc))
    except search.EmbedderUnavailable as exc:
        _fail("embedder_unavailable", f'{exc}; mode="keyword" does not need it')
    except ValueError as exc:  # no non-empty query
        _fail("bad_argument", str(exc))
    return [h.as_dict() for h in hits]


def run_recent(
    conn: Conn,
    tags: list[str] | None = None,
    chat_ids: list[int] | None = None,
    hours: int | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = 20,
    cursor: str | None = None,
    now: datetime | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    """The `recent` tool: `hours` back from the server's clock (default 24), or an explicit
    `since` (and optional `until`); never both, and `until` only with `since`. With a
    `cursor`, its range and filters are used and the other arguments but `limit` ignored."""
    if not 1 <= limit <= search.MAX_LIMIT:
        _fail("bad_range", f"limit must be between 1 and {search.MAX_LIMIT}")
    now = now or datetime.now(UTC)
    if cursor is not None:
        hours = since = until = tags = chat_ids = None
    if hours is not None and since is not None:
        _fail("bad_argument", "give hours or since, not both")
    if until is not None and since is None:
        _fail("bad_argument", "until needs since")
    if since is None and cursor is None:
        hours = 24 if hours is None else hours
        if not 1 <= hours <= search.MAX_RECENT_HOURS:
            _fail("bad_range", f"hours must be between 1 and {search.MAX_RECENT_HOURS}")
        since = now - timedelta(hours=hours)
    if tags:
        _check_tags(conn, tags)
    if chat_ids:
        _check_chats(conn, chat_ids)
    try:
        answer = search.recent(
            conn,
            since=since,
            until=until,
            tags=tags or None,
            chat_ids=chat_ids or None,
            limit=limit,
            cursor=cursor,
            now=now,
            compact=compact,
        )
    except search.BadRange as exc:
        _fail("bad_range", str(exc))
    except ValueError as exc:  # a cursor this server did not return
        _fail("bad_argument", str(exc))
    return answer.as_dict()


def _chat(conn: Conn, chat_id: int) -> tuple[str, str | None]:
    row = conn.execute("SELECT kind, username FROM chat WHERE chat_id = %s", (chat_id,)).fetchone()
    if row is None:
        _fail("unknown_chat", "chat_id is not in the archive; see list_chats")
    return cast(str, row[0]), cast(str | None, row[1])


def _messages(conn: Conn, chat_id: int, ids: list[int]) -> list[dict[str, Any]]:
    """The live messages among `ids`, in `Hit.messages` shape, by (date, message_id)."""
    kind, username = _chat(conn, chat_id)
    rows = conn.execute(
        "SELECT message_id, sender_name, date, text, server_message_id FROM message"
        " WHERE chat_id = %s AND message_id = ANY(%s) AND deleted_at IS NULL"
        " ORDER BY date, message_id",
        (chat_id, ids),
    )
    return [
        {
            "message_id": r[0],
            "sender_name": r[1],
            "date": _iso(r[2]),
            "text": r[3],
            "link": search.link(kind, chat_id, username, cast(int, r[4])),
        }
        for r in rows
    ]


def get_thread(conn: Conn, chat_id: int, message_id: int, limit: int = 50) -> list[dict[str, Any]]:
    """The reply chain from the message up to its root, then the root's replies (all
    depths, earliest first) up to `limit` messages, returned by date. When the chain alone
    is longer than `limit`, the part nearest the message is kept. Tombstoned messages are
    followed but not returned; an unknown message in a known chat gives `[]`."""
    if not 1 <= limit <= MAX_THREAD:
        _fail("bad_range", f"limit must be between 1 and {MAX_THREAD}")
    _chat(conn, chat_id)
    chain = [
        cast(int, r[0])
        for r in conn.execute(
            "WITH RECURSIVE up(message_id, parent, depth) AS ("
            " SELECT message_id, reply_to_message_id, 0 FROM message"
            "  WHERE chat_id = %(chat)s AND message_id = %(msg)s"
            " UNION ALL"
            " SELECT m.message_id, m.reply_to_message_id, up.depth + 1 FROM up"
            "  JOIN message m ON m.chat_id = %(chat)s AND m.message_id = up.parent"
            "  WHERE up.depth < %(max)s)"
            " SELECT message_id FROM up ORDER BY depth",
            {"chat": chat_id, "msg": message_id, "max": MAX_DEPTH},
        )
    ]
    if not chain:
        return []
    replies = [
        cast(int, r[0])
        for r in conn.execute(
            "WITH RECURSIVE down(message_id, depth) AS ("
            " SELECT %(root)s::bigint, 0"
            " UNION ALL"
            " SELECT m.message_id, down.depth + 1 FROM down"
            "  JOIN message m ON m.chat_id = %(chat)s AND m.reply_to_message_id = down.message_id"
            "  WHERE down.depth < %(max)s)"
            " SELECT m.message_id FROM down JOIN message m"
            "  ON m.chat_id = %(chat)s AND m.message_id = down.message_id"
            " WHERE m.deleted_at IS NULL AND NOT m.message_id = ANY(%(chain)s)"
            " ORDER BY m.date, m.message_id LIMIT %(limit)s",
            {"chat": chat_id, "root": chain[-1], "chain": chain, "max": MAX_DEPTH, "limit": limit},
        )
    ]
    live = {
        cast(int, r[0])
        for r in conn.execute(
            "SELECT message_id FROM message WHERE chat_id = %s AND message_id = ANY(%s)"
            " AND deleted_at IS NULL",
            (chat_id, chain),
        )
    }
    kept = [m for m in chain if m in live][:limit]
    return _messages(conn, chat_id, kept + replies[: limit - len(kept)])


def get_context(
    conn: Conn, chat_id: int, message_id: int, before: int = 10, after: int = 10
) -> list[dict[str, Any]]:
    """The message with up to `before` earlier and `after` later messages of its chat, by
    (date, message_id), at most `MAX_CONTEXT` messages in all: `after` gives way first when
    `before + after` is 100. Tombstoned messages are left out; an unknown message gives `[]`."""
    if before < 0 or after < 0 or before + after > MAX_CONTEXT:
        _fail(
            "bad_range", f"before and after must be at least 0 and add up to {MAX_CONTEXT} at most"
        )
    before = min(before, MAX_CONTEXT - 1)
    after = min(after, MAX_CONTEXT - 1 - before)
    _chat(conn, chat_id)
    row = conn.execute(
        "SELECT date FROM message WHERE chat_id = %s AND message_id = %s", (chat_id, message_id)
    ).fetchone()
    if row is None:
        return []
    params = {"chat": chat_id, "msg": message_id, "date": row[0], "before": before, "after": after}
    ids = [
        cast(int, r[0])
        for r in conn.execute(
            "(SELECT message_id FROM message WHERE chat_id = %(chat)s AND deleted_at IS NULL"
            "  AND (date, message_id) < (%(date)s, %(msg)s)"
            "  ORDER BY date DESC, message_id DESC LIMIT %(before)s)"
            " UNION ALL"
            " (SELECT message_id FROM message WHERE chat_id = %(chat)s AND deleted_at IS NULL"
            "  AND (date, message_id) > (%(date)s, %(msg)s)"
            "  ORDER BY date, message_id LIMIT %(after)s)",
            params,
        )
    ]
    return _messages(conn, chat_id, [message_id, *ids])


def suggest_tags(conn: Conn, suggestions: list[Suggestion]) -> dict[str, int]:
    """Add each tag as `proposed` by `mcp`; an existing tag, proposed or confirmed, is left
    as is. All or nothing: an unknown chat or an empty tag writes nothing. `reason` is not
    stored (`chat_tag` has no column for it). `proposed` is the number of new rows."""
    rows = [(s.chat_id, s.tag.strip()) for s in suggestions]
    if any(not tag for _, tag in rows):
        _fail("bad_argument", "a tag must not be empty")
    _check_chats(conn, sorted({chat_id for chat_id, _ in rows}))
    proposed = 0
    with conn.transaction():
        for chat_id, tag in rows:
            cur = conn.execute(
                "INSERT INTO chat_tag (chat_id, tag, status, proposed_by)"
                " VALUES (%s, %s, 'proposed', 'mcp') ON CONFLICT (chat_id, tag) DO NOTHING",
                (chat_id, tag),
            )
            proposed += cur.rowcount
    log.info("suggest_tags: %s suggested, %s proposed", len(rows), proposed)
    return {"proposed": proposed}


# Opens the account `add_chat` joins with: its phone and a client the caller closes.
Telegram = Callable[[], tuple[str, TdClient]]

# TDLib answers for a link that leads nowhere the account can go.
_NOT_FOUND = ("USERNAME_NOT_OCCUPIED", "USERNAME_INVALID", "INVITE_HASH_INVALID",
              "INVITE_HASH_EXPIRED", "CHANNEL_PRIVATE", "Chat not found")  # fmt: skip


def _sample(client: TdClient, chat_id: int) -> list[Message] | None:
    """The chat's newest messages for the estimate: two pages at most (a cold TDLib answers
    the first with a few). Best effort: any failure, even on the second page, gives None
    (unknown), never a short or empty sample that would look like a small chat."""
    sample: list[Message] = []
    try:
        for _ in range(2):
            page = client.get_chat_history(
                chat_id, sample[-1].message_id if sample else 0, ingest.PAGE_LIMIT
            )
            sample += [m for m in page if not sample or m.message_id < sample[-1].message_id]
            if not page or len(sample) >= ingest.PAGE_LIMIT:
                break
    except Exception as exc:  # the estimate is optional; the join stands
        log.info("add_chat: no history sample for the estimate (%s)", type(exc).__name__)
        return None
    return sample


@dataclass(frozen=True)
class Joined:
    """A chat `join_chat` joined and enabled, the account that joined it, the backfill depth
    now in force, and the history sample for the estimate (None: unknown)."""

    chat: Chat
    account: str
    depth: chats.Depth
    sample: list[Message] | None


def join_chat(
    conn: Conn, telegram: Telegram, link: ChatLink, years: int | None, default_years: int
) -> Joined:
    """Join the chat behind `link` with the account `telegram` opens and add it to the
    allowlist; the next `tgi sync` fetches it. Only supergroups and channels
    (`chats.JOINABLE_KINDS`): a private chat or a basic group is refused before anything is
    joined, and so is a chat enabled under another account (`chats.check_account`, again in
    `chats.enable`). Runs under the sync lock, since a running `tgi sync` holds the
    account's TDLib session. Shared by the MCP `add_chat` tool and `tgi chats add`; raises
    `ingest.SyncLocked`, `td.SetupError`, `NotJoinable`, `chats.ChatConflict`,
    `JoinPending`, `FloodWait` or `TdError`, and the caller words them."""
    with ingest.sync_lock(conn):
        account, client = telegram()
        try:
            chat = client.join_chat(
                link,
                chats.JOINABLE_KINDS,
                lambda chat_id: chats.check_account(conn, chat_id, account),
            )
            sample = _sample(client, chat.chat_id)
        finally:
            with contextlib.suppress(TdError):
                client.close()
        depth = chats.enable(conn, chat, account, years, default_years=default_years)
    return Joined(chat, account, depth, sample)


def not_found(exc: TdError) -> bool:
    """Whether a TDLib error means the link leads nowhere the account can go."""
    return exc.code == 400 and any(k in str(exc) for k in _NOT_FOUND)


def add_chat(
    conn: Conn,
    telegram: Telegram | None,
    link: str,
    years: int | None,
    default_years: int,
    sync_every_minutes: int = 0,
    now: datetime | None = None,
) -> dict[str, Any]:
    """The MCP `add_chat` tool: `join_chat`, with its failures as tool error codes and the
    result's `sync` block estimating when the chat is searchable (`tgi.eta`)."""
    if telegram is None:
        _fail("not_configured", "the server has no Telegram account to join with")
    try:
        parsed = parse_link(link)
    except ValueError:
        _fail("bad_argument", "link must be a t.me link, an @username or a chat id")
    try:
        if years is not None:
            check_years(years)
    except ValueError as exc:
        _fail("bad_argument", str(exc))
    try:
        joined = join_chat(conn, telegram, parsed, years, default_years)
    except ingest.SyncLocked:
        _fail("busy", "a tgi sync is running; try again when it ends")
    except td.SetupError as exc:
        _fail("not_configured", str(exc))
    except NotJoinable as exc:
        _fail("not_allowed", f"only supergroups and channels can be added, not a {exc.kind} chat")
    except chats.ChatConflict as exc:
        _fail("not_allowed", str(exc))
    except JoinPending:
        _fail("join_pending", "the link sends a join request; add it again once it is approved")
    except FloodWait as exc:
        _fail("flood_wait", f"Telegram asks to wait {exc.retry_after}s")
    except TdError as exc:
        if not_found(exc):
            _fail("not_found", "no chat this account can join behind the link")
        _fail("telegram_error", f"TDLib error {exc.code}")
    chat = joined.chat
    log.info("add_chat: enabled chat %s (%s)", chat.chat_id, chat.kind)
    try:
        when = eta.estimate(conn, chat.chat_id, joined.sample, now or datetime.now(UTC),
                            joined.depth.years, sync_every_minutes,
                            ingest.DEFAULT_MAX_MINUTES)  # fmt: skip
    except Exception as exc:  # the chat is enabled: an estimate failure must not hide that
        log.warning("add_chat: no estimate for chat %s (%s)", chat.chat_id, type(exc).__name__)
        when = eta.UNKNOWN
    return {"chat_id": chat.chat_id, "title": chat.title, "kind": chat.kind,
            "username": chat.username, "backfill_years": joined.depth.years,
            "sync": when.as_dict()}  # fmt: skip


def _iso(value: object) -> str | None:
    """ISO 8601 in UTC, as `Hit.as_dict` gives dates."""
    return None if value is None else cast(datetime, value).astimezone(UTC).isoformat()


# --- the server -------------------------------------------------------------------------------

# MCP tool annotations, so a client can tell reads from writes: Hermes on an `untrusted`
# server asks the user before any tool without `readOnlyHint`, and only retries those with
# it after an expired session. Hints, not enforcement: `MCP_READ_ONLY` is the server's own.
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
SUGGEST = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
JOIN = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
)


class _Server(MCPServer):
    """`MCPServer` whose argument-validation errors are `bad_argument` too. The SDK words
    them as pydantic's report, values included; this keeps the field names only. An unknown
    tool name is answered here, before the SDK would log it (and quote it in the error)."""

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context[Any, Any] | None = None
    ) -> CallToolResult | InputRequiredResult:
        if self._tool_manager.get_tool(name) is None:
            log.info("call to an unknown tool")
            error = {"code": "unknown_tool", "message": "no such tool; see tools/list"}
            text = f"Error executing tool: {json.dumps(error)}"
            return CallToolResult(content=[TextContent(type="text", text=text)], is_error=True)
        try:
            return await super().call_tool(name, arguments, context)
        except UnexpectedToolError:
            raise
        except ToolError as exc:
            cause = exc.__cause__
            if not isinstance(cause, ValidationError):
                raise
            fields = sorted({".".join(map(str, e["loc"])) for e in cause.errors()})
            error = {"code": "bad_argument", "message": f"invalid {', '.join(fields)}"}
            raise ToolError(f"Error executing tool {name}: {json.dumps(error)}") from cause


def build_server(
    database_url: str,
    embedder: Embedder | None = None,
    telegram: Telegram | None = None,
    default_years: int = Settings.backfill_years,
    sync_every_minutes: int = 0,
    read_only: bool = False,
) -> MCPServer:
    """The nine tools, or with `read_only` (`MCP_READ_ONLY`) the seven that only read:
    `suggest_tags` and `add_chat` are then not registered at all, whatever a client asks.
    `embedder` replaces the configured provider (tests); `telegram` opens the account
    `add_chat` joins with (none: the tool answers `not_configured`); `default_years` is
    `BACKFILL_YEARS`; `sync_every_minutes` is `SYNC_EVERY_MINUTES`."""
    server = _Server(
        "tgi",
        instructions="Search a personal Telegram archive (read-only)."
        if read_only
        else "Search a personal Telegram archive; add_chat joins a group or channel"
        " and adds it to the archive.",
    )

    def connected[T](fn: Callable[[Conn], T]) -> T:
        with db.connect(database_url) as conn:
            try:
                return fn(conn)
            except (psycopg.DataError, psycopg.errors.ProgramLimitExceeded) as exc:
                # a NUL byte, a tag too long for the index, ...
                _fail("bad_argument", f"an argument cannot be stored ({type(exc).__name__})")

    @server.tool(name="list_chats", annotations=READ_ONLY)
    def list_chats_tool(tag: str | None = None) -> list[dict[str, Any]]:
        """Chats in the archive with confirmed tags, message counts and date range.
        `tag` keeps the chats that tag is confirmed for."""
        return connected(lambda conn: list_chats(conn, tag))

    @server.tool(name="list_tags", annotations=READ_ONLY)
    def list_tags_tool() -> list[dict[str, Any]]:
        """Tags with their status (`confirmed` filters search, `proposed` does not)."""
        return connected(list_tags)

    @server.tool(name="search", annotations=READ_ONLY)
    def search_tool(
        queries: list[str],
        tags: list[str] | None = None,
        chat_ids: list[Int] | None = None,
        since: When | None = None,
        until: When | None = None,
        mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
        limit: Int = 20,
        compact: Bool = False,
    ) -> list[dict[str, Any]]:
        """Conversation windows matching any of `queries` (pass translations and variants;
        `"a phrase"`, `-word`), fused by rank, best first. `tags` are confirmed tags;
        `since`/`until` are inclusive: a date (YYYY-MM-DD) is a whole UTC day, a date-time
        an exact instant (UTC without a zone); `limit` is at most 50. Needs a topic: for
        "what's new" use `recent`. Each hit says how it was retrieved (not that it is
        relevant: check the text): `matched_by` (keyword, trigram, semantic),
        `matched_message_ids` (the keyword legs' matches), `matched_queries` (0-based
        positions in `queries`) and `semantic_distance` (the smallest cosine distance over
        the queries, lower is closer; null when the semantic leg did not find it).
        `compact=true` keeps the matched messages and 2 neighbours each (a semantic-only hit:
        its first 5, a sample), cuts long texts (`truncated`) and counts the rest in
        `omitted_messages` (`get_context` reads them).
        Retry with mode="keyword" on `embedder_unavailable` or `model_mismatch`."""
        return connected(
            lambda conn: run_search(
                conn, queries, tags, chat_ids, since, until, mode, limit, embedder, compact
            )
        )

    @server.tool(name="recent", annotations=READ_ONLY)
    def recent_tool(
        tags: list[str] | None = None,
        chat_ids: list[Int] | None = None,
        hours: Int | None = None,
        since: When | None = None,
        until: When | None = None,
        limit: Int = 20,
        cursor: str | None = None,
        compact: Bool = False,
    ) -> dict[str, Any]:
        """What's new, without a topic: conversation windows with messages posted in a time
        range, newest activity first, each with only its messages inside the range. Range:
        `hours` back from the server's clock (default 24, at most 744), or `since` and
        optionally `until` (omit `hours` then): `since` ≤ date < `until`, a date meaning
        that UTC day, `until` defaulting to now. The answer has the server's `now`, the
        exact range covered (UTC), `hits` (score has no meaning here) and `next_cursor`.
        For the next page call recent(cursor=<next_cursor>): the cursor keeps the range,
        filters and `compact`, so other arguments but `limit` are ignored. Continue until
        `next_cursor` is null, even after a short or empty page. `compact=true` cuts long texts
        (`truncated`). Use for digests and "what happened in these chats"; use `search` for
        a topic."""
        return connected(
            lambda conn: run_recent(
                conn, tags, chat_ids, hours, since, until, limit, cursor, compact=compact
            )
        )

    @server.tool(name="status", annotations=READ_ONLY)
    def status_tool(chat_ids: list[Int] | None = None) -> dict[str, Any]:
        """How current the archive is, to tell "not fetched yet" from "nothing found":
        `sync_running`, the `schedule` (next run estimated), the last run and the last `ok`
        one (a whole sync or a one-chat resweep), windows waiting for a new embedding
        (meaning-based search has none or an outdated one for them), and per chat
        `last_sync_at` (its last sync turn, complete, cut short or failed),
        `history_complete` (fetched back to its lookback), `oldest_message_at`,
        `newest_message_at`, `windows_need_rebuild` (a window rebuild is pending: `recent`
        and meaning-based results may be incomplete or outdated; keyword search finds the
        stored messages) and `windows_waiting_for_embedding`. Dates UTC."""

        def answer(conn: Conn) -> dict[str, Any]:
            if chat_ids:
                _check_chats(conn, chat_ids)
            return status.status(conn, chat_ids or None, default_years=default_years,
                                 sync_every_minutes=sync_every_minutes)  # fmt: skip

        return connected(answer)

    @server.tool(name="get_thread", annotations=READ_ONLY)
    def get_thread_tool(chat_id: Int, message_id: Int, limit: Int = 50) -> list[dict[str, Any]]:
        """The reply chain up to the root message and the replies to it (at most 50)."""
        return connected(lambda conn: get_thread(conn, chat_id, message_id, limit))

    @server.tool(name="get_context", annotations=READ_ONLY)
    def get_context_tool(
        chat_id: Int, message_id: Int, before: Int = 10, after: Int = 10
    ) -> list[dict[str, Any]]:
        """The message with the messages around it in its chat (before + after ≤ 100; at
        most 100 messages are returned, the message included)."""
        return connected(lambda conn: get_context(conn, chat_id, message_id, before, after))

    if read_only:
        return server

    @server.tool(name="suggest_tags", annotations=SUGGEST)
    def suggest_tags_tool(suggestions: list[Suggestion]) -> dict[str, int]:
        """Propose tags for chats; the owner confirms them before they filter search."""
        return connected(lambda conn: suggest_tags(conn, suggestions))

    @server.tool(name="add_chat", annotations=JOIN)
    def add_chat_tool(link: str, years: Int | None = None) -> dict[str, Any]:
        """Join a Telegram group or channel and add it to the archive; its history arrives
        with the next sync. `link`: https://t.me/name, an invite link https://t.me/+…,
        @name, or the chat id of a chat the account is in. Private chats, basic groups and
        chats already enabled under another account are refused (`not_allowed`). `busy`
        while a sync runs (retry later); `join_pending` when the chat's admins must approve.
        `years`: backfill depth, 1 to 100 (default: the server's BACKFILL_YEARS). The
        result's `sync` estimates when it is searchable: `next_run_at` (the next scheduled
        sync), `searchable_from` (its newest messages), `complete_by` (the whole lookback),
        `estimated_messages`, `estimated_minutes`; UTC, rough, null when no schedule is
        configured. Tell the user these times."""
        return connected(
            lambda conn: add_chat(conn, telegram, link, years, default_years, sync_every_minutes)
        )

    return server


class BearerAuth:
    """ASGI middleware: every HTTP request needs `Authorization: Bearer <token>`.

    Anything else gets `401` with an empty body; the log gets a count, never the token or
    the header it was sent."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.expected = f"Bearer {token}".encode()
        self.rejected = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            sent = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(sent, self.expected):
                self.rejected += 1
                log.warning("mcp: rejected an unauthorized request (%s so far)", self.rejected)
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await send({
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-length", b"0"), (b"www-authenticate", b"Bearer")],
                })  # fmt: skip
                await send({"type": "http.response.body", "body": b""})
                return
        await self.app(scope, receive, send)


def http_app(server: MCPServer, token: str, security: TransportSecuritySettings) -> BearerAuth:
    """The streamable-HTTP app at `/mcp` behind the token. Stateless with JSON responses:
    the tools never call back, so no session or event stream is kept. `security` is always
    explicit: the SDK turns DNS-rebinding protection on by itself only for 127.0.0.1,
    localhost and ::1, and leaves it off for any other bind."""
    app = server.streamable_http_app(
        stateless_http=True, json_response=True, transport_security=security
    )
    return BearerAuth(app, token)


def parse_bind(bind: str) -> tuple[str, int]:
    """`MCP_BIND` as (host, port): `127.0.0.1:8765`, `[::1]:8765`."""
    host, sep, port = bind.rpartition(":")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    elif ":" in host:
        host = ""  # an IPv6 address needs brackets
    if not sep or not host or not port.isdigit() or not 0 < int(port) < 65536:
        raise ValueError("MCP_BIND must be host:port, e.g. 127.0.0.1:8765 or [::1]:8765")
    return host, int(port)


# --- the bind guard and the Host check (SPEC §5.6, A12) ---------------------------------------

Interface = tuple[str, str]  # (interface name, address)
BRIDGES = ("docker", "br-")  # Docker's default bridge and user-defined networks
DOCKER_HOST = "host.docker.internal"  # the name a container reaches the host by (§5.6)


def parse_interfaces(text: str) -> list[Interface]:
    """(name, address) pairs from `ip -j addr` output."""
    return [
        (cast(str, i["ifname"]), cast(str, a["local"]))
        for i in json.loads(text)
        for a in i.get("addr_info", [])
        if "local" in a
    ]


def read_interfaces() -> list[Interface]:
    """The host's interface addresses; [] when `ip -j addr` is missing or fails, so only a
    loopback bind passes the guard then."""
    ip = shutil.which("ip")
    if ip is None:
        return []
    try:
        out = subprocess.run(  # noqa: S603 - fixed arguments, `ip` from PATH
            [ip, "-j", "addr"], capture_output=True, text=True, check=True, timeout=10
        ).stdout
        return parse_interfaces(out)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return []


def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_loopback(host: str) -> bool:
    addr = _ip(host)
    return host == "localhost" or (addr is not None and addr.is_loopback)


def _on_bridge(host: str, interfaces: list[Interface]) -> bool:
    addr = _ip(host)
    return addr is not None and any(
        name.startswith(BRIDGES) and _ip(local) == addr for name, local in interfaces
    )


def check_bind(host: str, interfaces: list[Interface], allow_any: bool = False) -> str | None:
    """None when `tgi mcp --http` may listen on `host`, else the one-line reason it may not:
    a loopback address, or an address of a local `docker*` or `br-*` interface, passes;
    `0.0.0.0`, `::` and anything else need `allow_any`."""
    if allow_any or _is_loopback(host) or _on_bridge(host, interfaces):
        return None
    addr = _ip(host)
    if addr is None:
        why = "not an IP address"
    elif addr.is_unspecified:
        why = "every interface, the LAN included"
    else:
        why = "not a loopback address or an address of a docker*/br-* interface"
    return (
        f"refusing to listen on {host} ({why}); use 127.0.0.1 or the Docker network's"
        " gateway address in MCP_BIND, or pass --allow-any-bind"
    )


_HOST_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?")


def transport_security(
    host: str, port: int, interfaces: list[Interface], extra: tuple[str, ...] = ()
) -> TransportSecuritySettings:
    """DNS-rebinding protection for the bind: the Host header must name the bind address,
    or a name a client on that path uses, at this port, and an Origin header (browsers
    only) must be one of those over http. A loopback bind adds `localhost`; a Docker-network
    bind adds `host.docker.internal`; `0.0.0.0` or `::` (`--allow-any-bind`) adds every
    interface address and both names. `extra` is `MCP_ALLOWED_HOSTS`."""
    names = [host]
    addr = _ip(host)
    if _is_loopback(host):
        names.append("localhost")
    elif addr is not None and addr.is_unspecified:
        names = [local for _, local in interfaces] + ["localhost", DOCKER_HOST]
    else:
        names.append(DOCKER_HOST)
    names += extra
    hosts = list(dict.fromkeys(f"[{n}]:{port}" if ":" in n else f"{n}:{port}" for n in names))
    # A client behind an HTTPS reverse proxy (`tailscale serve`) sends the proxy's name on
    # its default port, so without one: only for MCP_ALLOWED_HOSTS names, never a pattern.
    hosts += [n for n in extra if _HOST_NAME.fullmatch(n) and n not in hosts]
    _quiet_header_values()
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts],
    )


class _HeaderValues(logging.Filter):
    """The SDK logs a rejected Host or Origin header with its value, which the caller
    chose; keep the rejection, drop the value."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for prefix in ("Invalid Host header", "Invalid Origin header"):
            if message.startswith(prefix):
                record.msg, record.args = prefix, None
        return True


def _quiet_header_values() -> None:
    sdk = logging.getLogger("mcp.server.transport_security")
    if not any(isinstance(f, _HeaderValues) for f in sdk.filters):
        sdk.addFilter(_HeaderValues())
