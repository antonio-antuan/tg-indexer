"""Ingestion: catch-up, backfill, the sweep, then windows and embeddings (SPEC §5.3 steps 1-6).

Every page is stored and checkpointed in its own transaction, so a crash resumes from the
last committed page and the `(chat_id, message_id)` key makes re-fetched pages idempotent.
Logs carry chat and message ids only, never text or account phones (SPEC §5.1, A10).
"""

import logging
import time
from collections.abc import Callable, Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Protocol, cast

import psycopg
from psycopg.types.json import Jsonb

from tgi import embed, windows
from tgi.config import Settings
from tgi.td_port import FloodWait, Message, TdClient, TdError, server_message_id

log = logging.getLogger(__name__)

Conn = psycopg.Connection[tuple[object, ...]]

PAGE_LIMIT = 100
# `tgi sync --max-minutes` default: a run's time budget.
DEFAULT_MAX_MINUTES = 150
MAX_FLOOD_WAITS = 3
# Arbitrary constant, distinct from the migration lock; one `tgi sync` at a time.
SYNC_LOCK_KEY = 0x7467_6931

STAT_KEYS = (
    "chats",
    "requests",
    "fetched",
    "upserted",
    "skipped",
    "edited",
    "tombstoned",
    "purged",
    "untombstoned",
    "flood_waits",
    "windows_rebuilt",
    "windows_deleted",
    "embedded",
    "embed_failed",
)


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def describe_error(exc: BaseException) -> str:
    """A safe summary of `exc`: its type and any TDLib code, never its message.

    Exception text may quote row data, TDLib responses or connection strings.
    """
    if isinstance(exc, embed.ModelMismatch):
        return "model_mismatch"
    summary = type(exc).__name__
    if isinstance(exc, TdError):
        summary += f" {exc.code}"
    return summary


class SyncLocked(Exception):
    """Another `tgi sync` holds the advisory lock."""


@contextmanager
def sync_lock(conn: Conn) -> Iterator[None]:
    """Hold the `tgi sync` advisory lock (also taken by `tgi embed`) or raise `SyncLocked`."""
    conn.autocommit = True
    row = conn.execute("SELECT pg_try_advisory_lock(%s)", (SYNC_LOCK_KEY,)).fetchone()
    if not row or not row[0]:
        raise SyncLocked("another tgi sync is running")
    try:
        yield
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (SYNC_LOCK_KEY,))


class UnknownChat(Exception):
    """`--resweep-days` named a chat that is not in the archive."""


class _Stop(Exception):
    """Ends the run early with `status` (budget or flood limit); checkpoints stay valid."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


@dataclass
class SyncResult:
    run_id: int
    status: str
    stats: dict[str, int]


@dataclass
class _Chat:
    chat_id: int
    account_phone: str
    newest: int | None
    oldest: int | None
    backfill_done: bool
    cutoff: datetime
    backfill_cutoff: datetime | None = None  # the cutoff when the backfill last finished


def mark_windows_dirty(conn: Conn, chat_id: int, message_id: int) -> None:
    """Lower `chat.windows_dirty_from` to `message_id`; issue before the write it covers."""
    conn.execute(
        "UPDATE chat SET windows_dirty_from = least(coalesce(windows_dirty_from, %s), %s)"
        " WHERE chat_id = %s",
        (message_id, message_id, chat_id),
    )


def deepened(cutoff: datetime, backfill_cutoff: datetime | None) -> bool:
    """A finished backfill continues: its cutoff is unknown or the lookback now reaches further."""
    return backfill_cutoff is None or cutoff < backfill_cutoff


def years_before(moment: datetime, years: int) -> datetime:
    try:
        return moment.replace(year=moment.year - years)
    except ValueError:  # 29 February
        return moment.replace(year=moment.year - years, day=28)


def _oldest_id(page: list[Message]) -> int | None:
    """The paging cursor below `page`: its lowest positive id (None if it has none)."""
    ids = [m.message_id for m in page if m.message_id > 0]
    return min(ids) if ids else None


@dataclass
class _Run:
    conn: Conn
    client_for: Callable[[str], TdClient]
    settings: Settings
    clock: Clock
    deadline: float
    stats: dict[str, int] = field(default_factory=lambda: dict.fromkeys(STAT_KEYS, 0))
    _floods: int = 0
    _last_request: dict[str, float] = field(default_factory=dict)

    # --- TDLib paging -------------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        """Sleep within the budget; a wait that would reach the deadline ends the run now."""
        if self.clock.monotonic() + seconds >= self.deadline:
            raise _Stop("time_limited")
        self.clock.sleep(seconds)

    def _request(self, chat: _Chat, from_message_id: int) -> list[Message]:
        client = self.client_for(chat.account_phone)
        delay = self.settings.page_delay_ms / 1000
        while True:
            if self.clock.monotonic() >= self.deadline:
                raise _Stop("time_limited")
            last = self._last_request.get(chat.account_phone)
            if last is not None and (wait := last + delay - self.clock.monotonic()) > 0:
                self._sleep(wait)
            self._last_request[chat.account_phone] = self.clock.monotonic()
            self.stats["requests"] += 1
            try:
                page = client.get_chat_history(chat.chat_id, from_message_id, PAGE_LIMIT)
            except FloodWait as exc:
                self._floods += 1
                self.stats["flood_waits"] += 1
                log.warning(
                    "flood wait chat_id=%s retry_after=%s consecutive=%s",
                    chat.chat_id,
                    exc.retry_after,
                    self._floods,
                )
                if self._floods >= MAX_FLOOD_WAITS:
                    raise _Stop("flood_limited") from None
                self._sleep(exc.retry_after)
                continue
            self._floods = 0
            self.stats["fetched"] += len(page)
            return page

    def page(self, chat: _Chat, cursor: int) -> list[Message]:
        """Messages strictly older than `cursor` (0: the newest page); [] = end of history.

        TDLib may or may not include `from_message_id` itself in a page; the boundary
        message is dropped here, and a page holding nothing but the boundary steps the
        cursor one below it so paging cannot loop.
        """
        while True:
            raw = self._request(chat, cursor)
            if not raw:
                return []
            page = [m for m in raw if cursor == 0 or m.message_id < cursor]
            if page:
                return page
            cursor -= 1

    # --- storage ------------------------------------------------------------------------

    def _store(self, chat: _Chat, page: list[Message], stored: Collection[int] = ()) -> None:
        """Upsert `page`; ids in `stored` are kept even if older than the chat's cutoff."""
        rows = []
        for m in page:
            try:
                sid = server_message_id(m.message_id)
            except ValueError:
                self.stats["skipped"] += 1
                log.info(
                    "skipped non-server message chat_id=%s message_id=%s",
                    chat.chat_id,
                    m.message_id,
                )
                continue
            if m.date < chat.cutoff and m.message_id not in stored:
                continue
            rows.append(
                (
                    chat.chat_id,
                    m.message_id,
                    sid,
                    m.sender_id,
                    m.sender_name,
                    m.date,
                    m.edit_date,
                    m.reply_to_message_id,
                    m.text,
                    m.has_media,
                )
            )
        if not rows:
            return
        # The window-rebuild marker comes first (SPEC §4 rule); callers hold a transaction.
        mark_windows_dirty(self.conn, chat.chat_id, min(r[1] for r in rows))
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO message (chat_id, message_id, server_message_id, sender_id,"
                " sender_name, date, edit_date, reply_to_message_id, text, has_media)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT (chat_id, message_id) DO UPDATE SET"
                " sender_id = EXCLUDED.sender_id, sender_name = EXCLUDED.sender_name,"
                " date = EXCLUDED.date, edit_date = EXCLUDED.edit_date,"
                " reply_to_message_id = EXCLUDED.reply_to_message_id,"
                " text = EXCLUDED.text, has_media = EXCLUDED.has_media"
                # Re-fetched unchanged messages are not rewritten.
                " WHERE (message.sender_id, message.sender_name, message.date,"
                " message.edit_date, message.reply_to_message_id, message.text,"
                " message.has_media) IS DISTINCT FROM (EXCLUDED.sender_id,"
                " EXCLUDED.sender_name, EXCLUDED.date, EXCLUDED.edit_date,"
                " EXCLUDED.reply_to_message_id, EXCLUDED.text, EXCLUDED.has_media)",
                rows,
            )
        self.stats["upserted"] += len(rows)

    def _all_stored(self, chat: _Chat, page: list[Message]) -> bool:
        """Every server message in `page` is already stored (skipped ids count as stored)."""
        ids = [m.message_id for m in page if m.message_id > 0 and not m.message_id % (1 << 20)]
        if not ids:
            return True
        row = self.conn.execute(
            "SELECT count(*) FROM message WHERE chat_id = %s AND message_id = ANY(%s)",
            (chat.chat_id, ids),
        ).fetchone()
        return row is not None and row[0] == len(ids)

    # --- steps --------------------------------------------------------------------------

    def catch_up(self, chat: _Chat) -> None:
        if chat.newest is None:
            self._first_page(chat)
            return
        checkpoint = chat.newest
        cursor, top = 0, None
        while True:
            page = self.page(chat, cursor)
            if not page:
                break
            oldest = _oldest_id(page)
            if top is None:
                top = max(m.message_id for m in page)
            with self.conn.transaction():
                self._store(chat, [m for m in page if m.message_id > checkpoint])
            if oldest is None or oldest <= checkpoint:
                break
            cursor = oldest
            if self._all_stored(chat, page):
                # The stored range above the checkpoint is contiguous: jump below it.
                row = self.conn.execute(
                    "SELECT min(message_id) FROM message WHERE chat_id = %s AND message_id > %s",
                    (chat.chat_id, checkpoint),
                ).fetchone()
                lowest = cast(int | None, row[0] if row else None)
                if lowest is not None and lowest < cursor:
                    log.info("catch-up jump chat_id=%s from=%s to=%s", chat.chat_id, cursor, lowest)
                    cursor = lowest
        # Only a completed catch-up moves the checkpoint (SPEC §5.3 step 1).
        if top is not None and top > checkpoint:
            self.conn.execute(
                "UPDATE chat SET newest_message_id = %s WHERE chat_id = %s", (top, chat.chat_id)
            )
            chat.newest = top

    def _first_page(self, chat: _Chat) -> None:
        page = self.page(chat, 0)
        oldest = _oldest_id(page)
        if oldest is None:
            return
        newest = max(m.message_id for m in page)
        with self.conn.transaction():
            self._store(chat, page)
            self.conn.execute(
                "UPDATE chat SET newest_message_id = %s, oldest_message_id = %s,"
                " backfill_done = false WHERE chat_id = %s",
                (newest, oldest, chat.chat_id),
            )
        chat.newest, chat.oldest, chat.backfill_done = newest, oldest, False

    def backfill_page(self, chat: _Chat) -> None:
        """Fetch one page below `oldest_message_id`; sets `backfill_done` at the end."""
        assert chat.oldest is not None  # noqa: S101 - only chats with a first page backfill
        page = self.page(chat, chat.oldest)
        oldest = _oldest_id(page)
        done = oldest is None or min(m.date for m in page) < chat.cutoff
        with self.conn.transaction():
            self._store(chat, page)
            self.conn.execute(
                "UPDATE chat SET oldest_message_id = %s, backfill_done = %s,"
                " backfill_cutoff = CASE WHEN %s THEN %s ELSE backfill_cutoff END"
                " WHERE chat_id = %s",
                (oldest or chat.oldest, done, done, chat.cutoff, chat.chat_id),
            )
        chat.oldest, chat.backfill_done = oldest or chat.oldest, done
        if done:
            chat.backfill_cutoff = chat.cutoff
            log.info("backfill done chat_id=%s", chat.chat_id)

    def sweep(self, chat: _Chat, days: int) -> None:
        """Re-page the last `days` days: apply edits, then deletions if the sweep completed.

        The sweep is complete once it receives a message older than the cutoff or reaches
        the chat start; only then are stored messages dated inside the paged range and not
        seen treated as deleted (SPEC §5.3 step 3). Any earlier stop deletes nothing.
        """
        cutoff = self.clock.now() - timedelta(days=days)
        seen: list[int] = []
        lowest: int | None = None  # set when the sweep ended on a message older than cutoff
        cursor = 0
        while True:
            page = self.page(chat, cursor)
            if not page:
                if self._archive_disputes_empty_page(chat, cutoff, cursor):
                    log.info("sweep chat_id=%s incomplete: untrusted empty page", chat.chat_id)
                    return
                break
            with self.conn.transaction():
                self._sweep_page(chat, [m for m in page if m.date >= cutoff])
            seen += [m.message_id for m in page]
            oldest = _oldest_id(page)
            if oldest is None:
                # Nothing to page below: the range is not known to be complete.
                log.info("sweep chat_id=%s incomplete: no server ids", chat.chat_id)
                return
            if min(m.date for m in page) < cutoff:
                lowest = oldest
                break
            cursor = oldest
        params = (chat.chat_id, cutoff, seen, lowest or 0)
        absent = "chat_id = %s AND date >= %s AND message_id <> ALL(%s) AND message_id >= %s" + (
            " AND deleted_at IS NULL" if self.settings.on_delete == "tombstone" else ""
        )
        with self.conn.transaction():
            row = self.conn.execute(
                f"SELECT min(message_id) FROM message WHERE {absent}",  # noqa: S608 - fixed SQL
                params,
            ).fetchone()
            if row is None or row[0] is None:
                return
            mark_windows_dirty(self.conn, chat.chat_id, cast(int, row[0]))
            if self.settings.on_delete == "purge":
                gone = self.conn.execute(
                    f"DELETE FROM message WHERE {absent} RETURNING message_id",  # noqa: S608
                    params,
                ).fetchall()
                self.stats["purged"] += len(gone)
            else:
                gone = self.conn.execute(
                    f"UPDATE message SET deleted_at = now() WHERE {absent}"  # noqa: S608
                    " RETURNING message_id",
                    params,
                ).fetchall()
                self.stats["tombstoned"] += len(gone)
        if gone:
            log.info("sweep chat_id=%s deleted=%s", chat.chat_id, len(gone))

    def _archive_disputes_empty_page(self, chat: _Chat, cutoff: datetime, cursor: int) -> bool:
        """Whether a live stored window message lies where an empty page found nothing.

        Such a page is not trusted as the chat start (SPEC §5.3 step 3): a spurious empty
        response would otherwise delete every stored message in the window.
        """
        row = self.conn.execute(
            "SELECT EXISTS (SELECT 1 FROM message WHERE chat_id = %s AND date >= %s"
            " AND deleted_at IS NULL AND (%s = 0 OR message_id < %s))",
            (chat.chat_id, cutoff, cursor, cursor),
        ).fetchone()
        return bool(row and row[0])

    def _sweep_page(self, chat: _Chat, page: list[Message]) -> None:
        """Store messages new to the archive or edited, and un-tombstone reappearing ones."""
        ids = [m.message_id for m in page]
        stored = {
            cast(int, r[0]): (r[1], r[2])
            for r in self.conn.execute(
                "SELECT message_id, edit_date, deleted_at IS NOT NULL FROM message"
                " WHERE chat_id = %s AND message_id = ANY(%s)",
                (chat.chat_id, ids),
            )
        }
        changed = [
            m for m in page if m.message_id not in stored or stored[m.message_id][0] != m.edit_date
        ]
        self.stats["edited"] += sum(m.message_id in stored for m in changed)
        # A deep --resweep-days may reach stored messages older than the backfill cutoff.
        self._store(chat, changed, stored=stored.keys())
        back = [i for i, (_, deleted) in stored.items() if deleted]
        if back:
            mark_windows_dirty(self.conn, chat.chat_id, min(back))
            self.conn.execute(
                "UPDATE message SET deleted_at = NULL WHERE chat_id = %s AND message_id = ANY(%s)",
                (chat.chat_id, back),
            )
            self.stats["untombstoned"] += len(back)

    def _load_chats(self, only: int | None = None) -> list[_Chat]:
        """Enabled chats in sync order, or the one chat `only` whether enabled or not."""
        rows = self.conn.execute(
            "SELECT chat_id, account_phone, newest_message_id, oldest_message_id,"
            " backfill_done, backfill_years, backfill_cutoff FROM chat"
            " WHERE CASE WHEN %(only)s::bigint IS NULL THEN enabled ELSE chat_id = %(only)s END"
            " ORDER BY last_sync_at ASC NULLS FIRST, chat_id",
            {"only": only},
        ).fetchall()
        now = self.clock.now()
        chats = [
            _Chat(
                chat_id=cast(int, r[0]),
                account_phone=cast(str, r[1]),
                newest=cast(int | None, r[2]),
                oldest=cast(int | None, r[3]),
                backfill_done=cast(bool, r[4]),
                cutoff=years_before(now, cast(int | None, r[5]) or self.settings.backfill_years),
                backfill_cutoff=cast(datetime | None, r[6]),
            )
            for r in rows
        ]
        if only is None:
            for chat in chats:
                if chat.backfill_done and deepened(chat.cutoff, chat.backfill_cutoff):
                    self._reopen(chat)
        self.stats["chats"] = len(chats)
        return chats

    def _reopen(self, chat: _Chat) -> None:
        """Continue a finished backfill downward from the oldest stored message (SPEC §5.3).

        `oldest_message_id` of a finished chat may sit below the oldest stored message (the
        last page's older-than-cutoff tail is not stored), so the cursor is taken from the
        stored rows; with none stored it is 0, the top, which re-pages the whole chat.
        """
        row = self.conn.execute(
            "UPDATE chat SET backfill_done = false, oldest_message_id = coalesce("
            "(SELECT min(message_id) FROM message WHERE chat_id = %s), 0)"
            " WHERE chat_id = %s RETURNING oldest_message_id",
            (chat.chat_id, chat.chat_id),
        ).fetchone()
        assert row is not None  # noqa: S101 - the chat row was just read
        chat.backfill_done, chat.oldest = False, cast(int, row[0])
        log.info("backfill reopened chat_id=%s", chat.chat_id)

    def resweep(self, chat_id: int, days: int) -> None:
        """`tgi sync --resweep-days N <chat_id>`: the sweep alone, over a deeper range."""
        chats = self._load_chats(chat_id)
        if not chats:
            raise UnknownChat(f"chat {chat_id} is not in the archive")
        self.sweep(chats[0], days)

    def run(self) -> None:
        chats = self._load_chats()
        for chat in chats:
            try:
                self.catch_up(chat)
            finally:
                # The chat's turn ends here, completed or cut off (SPEC §5.3 ordering).
                self.conn.execute(
                    "UPDATE chat SET last_sync_at = clock_timestamp() WHERE chat_id = %s",
                    (chat.chat_id,),
                )
        pending = [c for c in chats if not c.backfill_done and c.oldest is not None]
        while pending:
            for chat in pending:
                self.backfill_page(chat)
            pending = [c for c in pending if not c.backfill_done]
        for chat in chats:
            if chat.newest is not None:
                self.sweep(chat, self.settings.sweep_days)


def sync(
    conn: Conn,
    client_for: Callable[[str], TdClient],
    settings: Settings,
    clock: Clock | None = None,
    max_minutes: float = DEFAULT_MAX_MINUTES,
    resweep: tuple[int, int] | None = None,
    embedder: embed.Embedder | None = None,
) -> SyncResult:
    """One `tgi sync` run over all enabled chats; `client_for(account_phone)` gives clients.

    `resweep=(chat_id, days)` runs only that chat's sweep over the last `days` days.
    `embedder` defaults to the configured provider (`embed.provider(settings)`).

    Raises `SyncLocked` if another run holds the lock. Errors other than flood limits and
    the time budget mark the run `error` and propagate.
    """
    clock = clock or SystemClock()
    with sync_lock(conn):
        # Under the lock no live run can own a `running` row: those died without `finally`.
        stale = conn.execute(
            "UPDATE sync_run SET status = 'interrupted', finished_at = now()"
            " WHERE status = 'running'"
        ).rowcount
        if stale:
            log.info("marked %s stale running sync_run rows interrupted", stale)
        row = conn.execute(
            "INSERT INTO sync_run (status) VALUES ('running') RETURNING run_id"
        ).fetchone()
        assert row is not None  # noqa: S101 - RETURNING always yields a row
        run_id = cast(int, row[0])
        log.info("sync run %s started", run_id)
        run = _Run(conn, client_for, settings, clock, clock.monotonic() + max_minutes * 60)
        status, error = "ok", None
        try:
            try:
                if resweep is None:
                    run.run()
                else:
                    run.resweep(*resweep)
            except _Stop as stop:
                status = stop.status
            # Window step (§5.3 step 5): local work, so partial runs do it too.
            for key, n in windows.rebuild_dirty(conn, settings).items():
                run.stats[key] += n
            # Embed step (§5.3 step 6), within the time budget; failed batches stay stale.
            done = embed.embed_stale(
                conn,
                embedder or embed.provider(settings),
                lambda: clock.monotonic() >= run.deadline,
            )
            run.stats["embedded"] += done["embedded"]
            run.stats["embed_failed"] += done["embed_failed"]
            if done["embed_stopped"] and status == "ok":
                status = "time_limited"
        except BaseException as exc:
            status, error = "error", describe_error(exc)
            raise
        finally:
            conn.execute(
                "UPDATE sync_run SET finished_at = now(), status = %s, stats = %s, error = %s"
                " WHERE run_id = %s",
                (status, Jsonb(run.stats), error, run_id),
            )
            log.info("sync run %s finished status=%s stats=%s", run_id, status, run.stats)
        return SyncResult(run_id, status, run.stats)


def purge_cutoff(now: datetime, days: int) -> datetime:
    """`now` less `days` days; a cutoff before year 1 is year 1, which nothing precedes."""
    try:
        return now - timedelta(days=days)
    except OverflowError:
        return datetime.min.replace(tzinfo=UTC)


def tombstones_before(conn: Conn, before: datetime) -> dict[int, int]:
    """Per chat, the tombstoned messages whose `deleted_at` is before `before` (D-TGI-3)."""
    rows = conn.execute(
        "SELECT chat_id, count(*) FROM message WHERE deleted_at < %s"
        " GROUP BY chat_id ORDER BY chat_id",
        (before,),
    )
    return {cast(int, r[0]): cast(int, r[1]) for r in rows}


def purge_tombstones(conn: Conn, before: datetime) -> dict[int, int]:
    """Hard-delete the tombstones `tombstones_before` counts; the deleted count per chat.

    Each chat's window marker is lowered to its lowest purged id before its delete (§4 rule),
    all in one transaction. Live messages are never touched. Call it under `sync_lock`.
    """
    purged: dict[int, int] = {}
    with conn.transaction():
        firsts = conn.execute(
            "SELECT chat_id, min(message_id) FROM message WHERE deleted_at < %s"
            " GROUP BY chat_id ORDER BY chat_id",
            (before,),
        ).fetchall()
        for chat_id, first in firsts:
            mark_windows_dirty(conn, cast(int, chat_id), cast(int, first))
            cur = conn.execute(
                "DELETE FROM message WHERE chat_id = %s AND deleted_at < %s", (chat_id, before)
            )
            purged[cast(int, chat_id)] = cur.rowcount
    log.info("purged tombstones: chats=%d messages=%d", len(purged), sum(purged.values()))
    return purged
