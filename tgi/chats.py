"""The chat allowlist and chat tags: enable, disable, forget, tags (SPEC §5.2, §4 rules)."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from tgi.config import Settings, check_years
from tgi.ingest import Conn, deepened, years_before
from tgi.td_port import Chat, ChatKind

# Private and basic-group chat ids are per account and can collide between accounts.
PER_ACCOUNT_KINDS = ("private", "basic_group")
# What the MCP `add_chat` tool may join and enable: never a private chat or a basic group.
JOINABLE_KINDS: tuple[ChatKind, ...] = ("supergroup", "channel")


# Namespace of the per-chat `enable` lock; the two-int advisory keys never meet the
# one-bigint keys of the sync and migration locks.
ENABLE_LOCK_NS = 0x7467_6932


class ChatConflict(Exception):
    """The chat id is already held by another account."""


def check_account(conn: Conn, chat_id: int, account_phone: str) -> None:
    """Raise `ChatConflict` if `enable` would refuse `chat_id` for `account_phone` because
    another account holds it; `add_chat` asks before joining. Same rule as `enable`."""
    row = conn.execute(
        "SELECT account_phone, kind, enabled FROM chat WHERE chat_id = %s", (chat_id,)
    ).fetchone()
    if row is not None:
        _check_holder(chat_id, account_phone, cast(str, row[0]), cast(str, row[1]), row[2])


def _check_holder(
    chat_id: int, account_phone: str, holder: str, kind: str, enabled: object, new_kind: str = ""
) -> None:
    if holder == account_phone:
        return
    if kind in PER_ACCOUNT_KINDS or new_kind in PER_ACCOUNT_KINDS:
        raise ChatConflict(f"chat {chat_id} is a per-account chat already held by another account")
    if enabled:
        raise ChatConflict(
            f"chat {chat_id} is already enabled under another account;"
            " disable it there first (tgi chats disable)"
        )


@dataclass(frozen=True)
class Depth:
    years: int  # effective lookback after this call
    reopens: bool  # True when a finished backfill will continue to `years` on the next sync


def enable(
    conn: Conn,
    chat: Chat,
    account_phone: str,
    years: int | None = None,
    *,
    default_years: int = Settings.backfill_years,
    now: datetime | None = None,
) -> Depth:
    """Add `chat` to the allowlist, fetched by `account_phone`; report its backfill depth.

    A private or basic-group chat id stored under another account is refused: it may be a
    different chat with the same id, and one row cannot hold both (`chats forget` first).
    Any chat enabled under another account is refused too (Antony, 2026-10-09): its
    `account_phone` is never overwritten; a disabled supergroup or channel may move.
    A changed title is part of every window's text, so it marks the whole chat for rebuild.
    The reopen itself is left to the next `tgi sync` (SPEC §5.3 step 2); a smaller depth is
    stored and deletes nothing. `default_years` is `BACKFILL_YEARS`; `now` is for tests.
    Both are checked (`check_years`) before anything is written.
    """
    if years is not None:
        check_years(years)
    check_years(default_years, "BACKFILL_YEARS")
    with conn.transaction():
        # FOR UPDATE locks nothing while the row is absent: two first enables from different
        # accounts would both insert, the second overwriting the owner (#43). Serialize per
        # chat id instead; a hash collision only makes two enables wait for each other.
        conn.execute(
            "SELECT pg_advisory_xact_lock(%s, hashtext(%s::text))", (ENABLE_LOCK_NS, chat.chat_id)
        )
        row = conn.execute(
            "SELECT account_phone, kind, title, enabled FROM chat WHERE chat_id = %s FOR UPDATE",
            (chat.chat_id,),
        ).fetchone()
        if row is not None:
            _check_holder(chat.chat_id, account_phone, cast(str, row[0]), cast(str, row[1]),
                          row[3], chat.kind)  # fmt: skip
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind, username, enabled,"
            " backfill_years) VALUES (%s, %s, %s, %s, %s, true, %s)"
            " ON CONFLICT (chat_id) DO UPDATE SET account_phone = EXCLUDED.account_phone,"
            " title = EXCLUDED.title, kind = EXCLUDED.kind, username = EXCLUDED.username,"
            " enabled = true,"
            " backfill_years = COALESCE(EXCLUDED.backfill_years, chat.backfill_years)",
            (chat.chat_id, account_phone, chat.title, chat.kind, chat.username, years),
        )
        if row is not None and row[2] != chat.title:
            # least() ignores NULLs: the chat's lowest message id, or the marker if lower.
            conn.execute(
                "UPDATE chat SET windows_dirty_from = least(windows_dirty_from,"
                " (SELECT min(message_id) FROM message WHERE chat_id = %s)) WHERE chat_id = %s",
                (chat.chat_id, chat.chat_id),
            )
        row = conn.execute(
            "SELECT backfill_done, backfill_years, backfill_cutoff FROM chat WHERE chat_id = %s",
            (chat.chat_id,),
        ).fetchone()
    assert row is not None  # noqa: S101 - upserted above
    effective = cast(int | None, row[1]) or default_years
    cutoff = years_before(now or datetime.now(UTC), effective)
    done, finished_at = cast(bool, row[0]), cast(datetime | None, row[2])
    return Depth(effective, done and deepened(cutoff, finished_at))


def disable(conn: Conn, chat_id: int) -> bool:
    """Stop syncing the chat and keep its data; False if the chat is unknown."""
    with conn.transaction():
        cur = conn.execute("UPDATE chat SET enabled = false WHERE chat_id = %s", (chat_id,))
        return cur.rowcount == 1


def forget(conn: Conn, chat_id: int) -> bool:
    """Delete the chat with its messages, windows, embeddings and tags (FK cascade)."""
    with conn.transaction():
        cur = conn.execute("DELETE FROM chat WHERE chat_id = %s", (chat_id,))
        return cur.rowcount == 1


def enabled_ids(conn: Conn, account_phone: str) -> set[int]:
    """Chats enabled for (fetched by) `account_phone`."""
    rows = conn.execute(
        "SELECT chat_id FROM chat WHERE enabled AND account_phone = %s", (account_phone,)
    )
    return {cast(int, r[0]) for r in rows}


def _tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        raise ValueError("a tag must not be empty")
    return tag


def propose_tag(conn: Conn, chat_id: int, tag: str, proposed_by: str) -> bool:
    """Add `tag` as `proposed`; an existing tag (confirmed or not) is left as is.

    False if the chat is unknown.
    """
    with conn.transaction():
        if conn.execute("SELECT 1 FROM chat WHERE chat_id = %s", (chat_id,)).fetchone() is None:
            return False
        conn.execute(
            "INSERT INTO chat_tag (chat_id, tag, status, proposed_by)"
            " VALUES (%s, %s, 'proposed', %s) ON CONFLICT (chat_id, tag) DO NOTHING",
            (chat_id, _tag(tag), proposed_by),
        )
    return True


def confirm_tag(conn: Conn, chat_id: int, tag: str) -> bool:
    """Mark `tag` `confirmed`, adding it if it was not proposed; False if the chat is unknown."""
    with conn.transaction():
        if conn.execute("SELECT 1 FROM chat WHERE chat_id = %s", (chat_id,)).fetchone() is None:
            return False
        conn.execute(
            "INSERT INTO chat_tag (chat_id, tag, status) VALUES (%s, %s, 'confirmed')"
            " ON CONFLICT (chat_id, tag) DO UPDATE SET status = 'confirmed'",
            (chat_id, _tag(tag)),
        )
    return True


def remove_tag(conn: Conn, chat_id: int, tag: str) -> bool:
    """Delete `tag` from the chat, proposed or confirmed; False if it was not there."""
    with conn.transaction():
        cur = conn.execute(
            "DELETE FROM chat_tag WHERE chat_id = %s AND tag = %s", (chat_id, tag.strip())
        )
        return cur.rowcount == 1
