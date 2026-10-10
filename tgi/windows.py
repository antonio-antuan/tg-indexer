"""Conversation windows (SPEC §5.4) and the window step of `tgi sync` (§5.3 step 5).

Within one chat, live messages with text, ordered by `(date, message_id)`, are cut into
windows at a gap over `WINDOW_GAP_MIN` or when the window would exceed `WINDOW_MAX_CHARS`
(counted over its `<sender>: <text>` lines) or `WINDOW_MAX_MSGS`. Windows are deterministic,
so a chat is rebuilt from the start of the window holding `chat.windows_dirty_from` onward
and upserted on `(chat_id, first_message_id)`: an unchanged window keeps its `window_id`,
`text_hash` and embedding. Logs carry ids and counts only.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import psycopg

from tgi.config import Settings

log = logging.getLogger(__name__)

Conn = psycopg.Connection[tuple[object, ...]]

REPLY_QUOTE_CHARS = 200


@dataclass(frozen=True)
class Msg:
    message_id: int
    date: datetime
    sender: str
    text: str
    reply_to: int | None


@dataclass(frozen=True)
class Window:
    first_message_id: int
    start_date: datetime
    end_date: datetime
    message_ids: list[int]
    text: str

    @property
    def text_hash(self) -> bytes:
        return hashlib.sha256(self.text.encode()).digest()


def _line(m: Msg) -> str:
    return f"{m.sender}: {m.text}"


def split(msgs: list[Msg], settings: Settings) -> list[list[Msg]]:
    """Cut `msgs` (sorted by `(date, message_id)`) into windows' message lists."""
    gap = timedelta(minutes=settings.window_gap_min)
    groups: list[list[Msg]] = []
    chars = 0
    for m in msgs:
        size = len(_line(m))
        cur = groups[-1] if groups else None
        if (
            cur is None
            or m.date - cur[-1].date > gap
            or chars + size > settings.window_max_chars
            or len(cur) >= settings.window_max_msgs
        ):
            groups.append([m])
            chars = size
        else:
            cur.append(m)
            chars += size
    return groups


def render(title: str, group: list[Msg], parents: dict[int, Msg], settings: Settings) -> Window:
    """Window text: `[<title>] <UTC start, minutes>` then one `<sender>: <text>` line per message.

    A reply to a message outside the window is preceded by `↪ <sender>: <first 200 chars>`
    of its parent; a single message longer than `WINDOW_MAX_CHARS` is truncated to it.
    """
    ids = {m.message_id for m in group}
    # UTC, so the text (and its hash) does not depend on the session's TimeZone.
    lines = [f"[{title}] {group[0].date.astimezone(UTC):%Y-%m-%d %H:%M}"]
    for m in group:
        parent = parents.get(m.reply_to) if m.reply_to is not None else None
        if parent is not None and parent.message_id not in ids:
            lines.append(f"↪ {parent.sender}: {parent.text[:REPLY_QUOTE_CHARS]}")
        if len(group) == 1:
            lines.append(_line(m)[: settings.window_max_chars])
        else:
            lines.append(_line(m))
    return Window(
        first_message_id=group[0].message_id,
        start_date=group[0].date,
        end_date=group[-1].date,
        message_ids=[m.message_id for m in group],
        text="\n".join(lines),
    )


_MSG_COLUMNS = (
    "message_id, date, coalesce(sender_name, '?'), text, reply_to_message_id FROM message"
    " WHERE chat_id = %s AND deleted_at IS NULL AND text <> ''"
)


def _msg(row: tuple[object, ...]) -> Msg:
    return Msg(
        message_id=cast(int, row[0]),
        date=cast(datetime, row[1]),
        sender=cast(str, row[2]),
        text=cast(str, row[3]),
        reply_to=cast(int | None, row[4]),
    )


def rebuild_chat(conn: Conn, chat_id: int, settings: Settings) -> dict[str, int]:
    """Rebuild the chat's windows from its marker and clear it, in one transaction.

    Returns counts: `windows_rebuilt` (new or changed text, now `stale`) and
    `windows_deleted`. Does nothing if the marker is clear.
    """
    counts = {"windows_rebuilt": 0, "windows_deleted": 0}
    with conn.transaction():
        row = conn.execute(
            "SELECT windows_dirty_from, title FROM chat WHERE chat_id = %s FOR UPDATE",
            (chat_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return counts
        dirty_from, title = cast(int, row[0]), cast(str | None, row[1]) or ""
        # The window holding the marker (greatest first_message_id at or below it). When the
        # marker is a window's first message, the window before it is rebuilt as well: that
        # message may have shrunk or gone, so the earlier window may now absorb what follows.
        start = conn.execute(
            'SELECT start_date, first_message_id FROM "window"'
            " WHERE chat_id = %s AND first_message_id < %s"
            " ORDER BY first_message_id DESC LIMIT 1",
            (chat_id, dirty_from),
        ).fetchone()
        if start is None:
            rows = conn.execute(
                f"SELECT {_MSG_COLUMNS} ORDER BY date, message_id",  # noqa: S608 - fixed SQL
                (chat_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_MSG_COLUMNS} AND (date, message_id) >= (%s, %s)"  # noqa: S608
                " ORDER BY date, message_id",
                (chat_id, start[0], start[1]),
            ).fetchall()
        msgs = [_msg(r) for r in rows]
        loaded = {m.message_id: m for m in msgs}
        wanted = sorted({m.reply_to for m in msgs if m.reply_to is not None} - loaded.keys())
        parents = dict(loaded)
        if wanted:
            for r in conn.execute(
                f"SELECT {_MSG_COLUMNS} AND message_id = ANY(%s)",  # noqa: S608 - fixed SQL
                (chat_id, wanted),
            ):
                parents[cast(int, r[0])] = _msg(r)
        windows = [render(title, g, parents, settings) for g in split(msgs, settings)]

        for w in windows:
            changed = conn.execute(
                'INSERT INTO "window" AS w (chat_id, first_message_id, start_date, end_date,'
                " message_ids, text, text_hash, stale) VALUES (%s, %s, %s, %s, %s, %s, %s, true)"
                " ON CONFLICT (chat_id, first_message_id) DO UPDATE SET"
                " start_date = excluded.start_date, end_date = excluded.end_date,"
                " message_ids = excluded.message_ids, text = excluded.text,"
                " text_hash = excluded.text_hash,"
                " stale = w.stale OR w.text_hash IS DISTINCT FROM excluded.text_hash"
                " WHERE (w.start_date, w.end_date, w.message_ids, w.text, w.text_hash)"
                " IS DISTINCT FROM (excluded.start_date, excluded.end_date,"
                " excluded.message_ids, excluded.text, excluded.text_hash)"
                " RETURNING w.stale",
                (
                    chat_id,
                    w.first_message_id,
                    w.start_date,
                    w.end_date,
                    w.message_ids,
                    w.text,
                    w.text_hash,
                ),
            ).fetchone()
            if changed is not None:
                counts["windows_rebuilt"] += 1

        # Windows from the rebuild start onward that this rebuild no longer produces.
        produced = [w.first_message_id for w in windows]
        if start is None:
            gone = conn.execute(
                'DELETE FROM "window" WHERE chat_id = %s AND first_message_id <> ALL(%s)',
                (chat_id, produced),
            ).rowcount
        else:
            gone = conn.execute(
                'DELETE FROM "window" WHERE chat_id = %s'
                " AND (start_date, first_message_id) >= (%s, %s)"
                " AND first_message_id <> ALL(%s)",
                (chat_id, start[0], start[1], produced),
            ).rowcount
        counts["windows_deleted"] = gone
        conn.execute("UPDATE chat SET windows_dirty_from = NULL WHERE chat_id = %s", (chat_id,))
    log.info(
        "windows chat_id=%s from=%s rebuilt=%s deleted=%s",
        chat_id,
        dirty_from,
        counts["windows_rebuilt"],
        counts["windows_deleted"],
    )
    return counts


def rebuild_dirty(conn: Conn, settings: Settings) -> dict[str, int]:
    """The window step: rebuild every chat whose `windows_dirty_from` is set."""
    totals = {"windows_rebuilt": 0, "windows_deleted": 0}
    ids = [
        cast(int, r[0])
        for r in conn.execute(
            "SELECT chat_id FROM chat WHERE windows_dirty_from IS NOT NULL ORDER BY chat_id"
        )
    ]
    for chat_id in ids:
        for key, n in rebuild_chat(conn, chat_id, settings).items():
            totals[key] += n
    return totals
