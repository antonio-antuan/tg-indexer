"""When a chat added by the MCP `add_chat` tool becomes searchable: a rough estimate.

The next run comes from the sync schedule (`SYNC_EVERY_MINUTES`, which must match the
host's timer); the volume from a sample of the chat's newest messages; the duration from
the throughput of past runs. Backfill is interleaved one page per chat per round (§5.3),
so a chat still backfilling shares a run with every other chat that is.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from tgi.ingest import Conn, years_before
from tgi.td_port import SERVER_ID_SHIFT, Message

# Messages per minute when no past run says otherwise (the first full run did ~4000).
DEFAULT_RATE = 3000.0
# Runs that fetched fewer messages say little about throughput.
MIN_SAMPLE_FETCHED = 1000
# Catch-up of the newest messages comes first in a run: they are searchable this soon.
FIRST_RESULTS_MIN = 5


@dataclass(frozen=True)
class Eta:
    next_run_at: datetime | None  # None: no schedule configured
    estimated_messages: int | None  # None: the history could not be sampled
    estimated_minutes: int | None
    searchable_from: datetime | None
    complete_by: datetime | None  # None: no schedule, or the volume is unknown

    def as_dict(self) -> dict[str, object]:
        def iso(t: datetime | None) -> str | None:
            return None if t is None else t.astimezone(UTC).isoformat()

        return {
            "next_run_at": iso(self.next_run_at),
            "estimated_messages": self.estimated_messages,
            "estimated_minutes": self.estimated_minutes,
            "searchable_from": iso(self.searchable_from),
            "complete_by": iso(self.complete_by),
        }


# Nothing is known: the estimate itself failed.
UNKNOWN = Eta(None, None, None, None, None)


def next_run(now: datetime, every_minutes: int) -> datetime | None:
    """The next multiple of `every_minutes` since the epoch (an hourly timer fires on the
    hour); None when no schedule is configured."""
    if every_minutes <= 0:
        return None
    step = every_minutes * 60
    seconds = math.floor(now.timestamp() / step + 1) * step
    return datetime.fromtimestamp(seconds, UTC)


def throughput(conn: Conn) -> float:
    """Messages per minute over recent finished runs that fetched enough to measure."""
    row = conn.execute(
        "SELECT sum((stats->>'fetched')::bigint),"
        " sum(extract(epoch FROM finished_at - started_at)) / 60"
        " FROM (SELECT * FROM sync_run WHERE finished_at IS NOT NULL"
        " AND (stats->>'fetched')::bigint >= %s ORDER BY run_id DESC LIMIT 10) r",
        (MIN_SAMPLE_FETCHED,),
    ).fetchone()
    fetched, minutes = row or (None, None)
    if not fetched or not minutes:
        return DEFAULT_RATE
    return max(float(cast(int, fetched)) / float(cast(float, minutes)), 1.0)


def backfilling_chats(conn: Conn, chat_id: int) -> int:
    """Other enabled chats whose backfill is not done: they share each run's rounds."""
    row = conn.execute(
        "SELECT count(*) FROM chat WHERE enabled AND NOT backfill_done AND chat_id <> %s",
        (chat_id,),
    ).fetchone()
    return int(cast(int, row[0])) if row else 0


def estimate_messages(sample: Sequence[Message], now: datetime, years: int) -> int:
    """Messages inside the lookback, from the newest page: its daily rate times the lookback
    in days, capped by the newest server message id (about all messages the chat ever had).
    A page that reaches past the cutoff is the whole lookback."""
    if not sample:
        return 0
    cutoff = years_before(now, years)
    newest, oldest = sample[0], sample[-1]
    ceiling = max(newest.message_id >> SERVER_ID_SHIFT, len(sample))
    if oldest.date <= cutoff:
        return sum(1 for m in sample if m.date > cutoff)
    span_days = (newest.date - oldest.date).total_seconds() / 86400
    if span_days <= 0:
        return min(ceiling, len(sample))
    per_day = len(sample) / span_days
    return min(ceiling, round(per_day * (now - cutoff).days))


def estimate(
    conn: Conn,
    chat_id: int,
    sample: Sequence[Message] | None,
    now: datetime,
    years: int,
    every_minutes: int,
    run_minutes: float,
) -> Eta:
    """When the chat's messages are searchable: first the newest (catch-up comes first in a
    run), all of them once its share of the runs' backfill rounds has fetched the lookback.
    A run stops at `run_minutes`; the next one starts at the following schedule tick.
    `sample` None (the history could not be read) leaves the volume and the end unknown:
    an unread chat is not an empty one."""
    start = next_run(now, every_minutes)
    first = None if start is None else start + timedelta(minutes=FIRST_RESULTS_MIN)
    if sample is None:
        return Eta(start, None, None, first, None)
    messages = estimate_messages(sample, now, years)
    rate = throughput(conn) / (1 + backfilling_chats(conn, chat_id))
    minutes = math.ceil(messages / rate) if messages else 0
    if start is None or first is None:
        return Eta(None, messages, minutes, None, None)
    end, left = start, float(minutes)
    while left > run_minutes:  # a run uses its whole budget, then waits for the next tick
        left -= run_minutes
        end = cast(datetime, next_run(end + timedelta(minutes=run_minutes), every_minutes))
    end += timedelta(minutes=max(left, FIRST_RESULTS_MIN))
    return Eta(start, messages, minutes, first, end)
