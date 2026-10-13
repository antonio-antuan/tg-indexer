"""How current the archive is (`tgi status`, the MCP `status` tool).

Per chat: when its sync turn last ended, whether its history is complete back to its
lookback, the newest stored message, whether stored messages still wait for the window step,
and the conversation windows still waiting for an embedding. Overall: whether a sync is
running now, the last run and the last `ok` one, the schedule, and the embedding backlog. It
lets an answer say "not fetched yet" instead of "nothing found". No account phones: the MCP
tool returns this to an agent.
"""

from datetime import UTC, datetime
from typing import Any, cast

from tgi.eta import next_run
from tgi.ingest import SYNC_LOCK_KEY, Conn, deepened, years_before

# The run stats worth showing; the rest (requests, skipped, …) are for debugging.
RUN_STATS = ("fetched", "upserted", "tombstoned", "windows_rebuilt", "embedded", "embed_failed")


def _iso(value: object) -> str | None:
    return None if value is None else cast(datetime, value).astimezone(UTC).isoformat()


def sync_lock_held(conn: Conn) -> bool:
    """Whether the sync lock is held now: by `tgi sync`, or by `embed`, `purge`, `forget`,
    `chats add` / `add_chat`, which share it.

    Read from `pg_locks` rather than by trying the lock, which would block it for a moment.
    A one-key advisory lock below 2^32 has `classid` 0, `objid` the key, `objsubid` 1; the
    lock belongs to this database (`pg_locks` lists every database on the server)."""
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND granted"
        " AND classid = 0 AND objid = %s AND objsubid = 1"
        " AND database = (SELECT oid FROM pg_database WHERE datname = current_database()))",
        (SYNC_LOCK_KEY,),
    ).fetchone()
    return bool(row and row[0])


def _run(row: tuple[object, ...] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    stats = cast(dict[str, int], row[4] or {})
    return {
        "run_id": row[0],
        "started_at": _iso(row[1]),
        "finished_at": _iso(row[2]),
        "status": row[3],
        "stats": {k: stats[k] for k in RUN_STATS if k in stats},
        "error": row[5],
    }


def status(
    conn: Conn,
    chat_ids: list[int] | None = None,
    now: datetime | None = None,
    default_years: int | None = None,
    sync_every_minutes: int = 0,
) -> dict[str, Any]:
    """The archive's freshness; `chat_ids` limits the per-chat part (None: every chat);
    `default_years` (`BACKFILL_YEARS`) is the lookback of a chat without its own;
    `sync_every_minutes` (`SYNC_EVERY_MINUTES`) the host's schedule, 0 when unknown.

    `last_sync_attempt_at` is when the chat's turn in a run last ended: completed, cut
    short by the budget or a flood wait, or failed. `history_complete` is whether the
    backfill reached the chat's current lookback, `history_cutoff` (a deeper lookback set
    since makes it false at once, before the next sync reopens it). `windows_need_rebuild`
    means a window rebuild is pending (new, edited or deleted messages, a renamed chat):
    `recent` and the semantic leg may be incomplete or outdated for that chat; keyword
    search finds the stored messages.
    `last_ok_run` is the latest run that ended `ok`: a whole sync or a single-chat resweep,
    so not proof that every chat is current."""
    now = (now or datetime.now(UTC)).astimezone(UTC)
    runs = "SELECT run_id, started_at, finished_at, status, stats, error FROM sync_run"
    last = conn.execute(f"{runs} ORDER BY run_id DESC LIMIT 1").fetchone()  # noqa: S608
    ok = conn.execute(
        f"{runs} WHERE status = 'ok' ORDER BY run_id DESC LIMIT 1"  # noqa: S608 - fixed SQL
    ).fetchone()
    totals = conn.execute('SELECT count(*), count(*) FILTER (WHERE stale) FROM "window"').fetchone()
    windows, pending = (cast(int, totals[0]), cast(int, totals[1])) if totals else (0, 0)
    rows = conn.execute(
        "SELECT c.chat_id, c.title, c.kind, c.enabled,"
        " ARRAY(SELECT t.tag FROM chat_tag t WHERE t.chat_id = c.chat_id"
        "       AND t.status = 'confirmed' ORDER BY t.tag),"
        " c.last_sync_at, c.backfill_done, c.backfill_years, c.backfill_cutoff,"
        " c.windows_dirty_from IS NOT NULL,"
        " (SELECT min(date) FROM message m WHERE m.chat_id = c.chat_id AND m.deleted_at IS NULL),"
        " (SELECT max(date) FROM message m WHERE m.chat_id = c.chat_id AND m.deleted_at IS NULL),"
        ' (SELECT count(*) FROM "window" w WHERE w.chat_id = c.chat_id AND w.stale)'
        " FROM chat c WHERE %(ids)s::bigint[] IS NULL OR c.chat_id = ANY(%(ids)s)"
        " ORDER BY c.title, c.chat_id",
        {"ids": chat_ids},
    ).fetchall()
    chats = []
    for r in rows:
        years = cast(int | None, r[7]) if r[7] is not None else default_years
        complete = bool(r[6])
        if complete and years is not None:  # a deeper lookback reopens it at the next sync
            cutoff = cast(datetime | None, r[8])
            complete = not deepened(years_before(now, years), cutoff)
        chats.append({
            "chat_id": r[0],
            "title": r[1],
            "kind": r[2],
            "enabled": r[3],
            "tags": r[4],
            "last_sync_attempt_at": _iso(r[5]),
            "history_complete": complete,
            "history_cutoff": None if years is None else years_before(now, years).isoformat(),
            "backfill_years": years,
            "oldest_message_at": _iso(r[10]),
            "newest_message_at": _iso(r[11]),
            "windows_need_rebuild": r[9],
            "windows_waiting_for_embedding": r[12],
        })  # fmt: skip
    upcoming = next_run(now, sync_every_minutes)
    return {
        "now": now.isoformat(),
        "sync_lock_held": sync_lock_held(conn),
        "schedule": {
            "every_minutes": sync_every_minutes or None,
            "next_run_at": None if upcoming is None else upcoming.isoformat(),  # an estimate
        },
        "last_run": _run(last),
        "last_ok_run": _run(ok),
        "windows": {"total": windows, "waiting_for_embedding": pending},
        "chats": chats,
    }
