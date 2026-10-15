"""Search core (SPEC §5.5): filters, keyword and semantic legs, fusion, windows, links.

Each query runs the full-text match (`websearch_to_tsquery` in russian, spanish and english,
OR-ed) against `message.tsv`, plus a trigram leg for its short tokens (`token <<% text`
under `pg_trgm.strict_word_similarity_threshold = 0.5`, served by the `gin_trgm_ops` index).
Negated terms (`-word`, `-"a phrase"`) are taken out of both legs and exclude every message
they match in any of the three configs.
The top `K` messages per query are mapped to their windows.

The semantic leg (`hybrid` and `semantic` modes) embeds each query's positive part with
`embed_queries` and takes the top `K` windows by cosine distance under the same filters
(dates on `window.start_date/end_date`). It ignores negation: in `hybrid` a window can come
back through it although one of its messages holds a negated term. A negation-only query
gets no semantic leg. HNSW with a selective filter can miss rows, so a pass that returns
fewer than `limit` rows is rerun as an exact scan (SPEC §7, filtered recall).

Windows are fused across legs and queries by reciprocal rank (`k = 60`). Queries are used
as given: expansion is the agent's job. Logs carry counts only, never queries or text.
"""

import base64
import binascii
import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from itertools import groupby
from typing import Any, Literal, cast

from psycopg import sql

from tgi import embed
from tgi.config import Settings
from tgi.embed import Embedder
from tgi.ingest import Conn

log = logging.getLogger(__name__)

Mode = Literal["hybrid", "keyword", "semantic"]

K = 100  # messages per query from the keyword leg, windows from the semantic leg
EF_SEARCH = 200  # hnsw.ef_search for the semantic leg (SPEC §4)
RRF_K = 60
MAX_LIMIT = 50  # SPEC §5.6 result bound
MAX_RECENT_HOURS = 24 * 31  # `recent(hours=…)`: a month; longer is what search is for
SHORT_TOKEN = 6  # tokens up to this many characters also go to the trigram leg
TRGM_THRESHOLD = 0.5
HIGHLIGHT_CHARS = 200
MAX_HIGHLIGHTS = 3
# compact hits: the matched messages and this many neighbours each side, or the first few
# messages when only the semantic leg found the window; texts cut to COMPACT_CHARS
COMPACT_NEIGHBOURS = 2
COMPACT_MESSAGES = 5
COMPACT_CHARS = 400
CONFIGS = ("russian", "spanish", "english")


class BadRange(ValueError):
    """`since` is after `until`."""


class ModelMismatch(embed.ModelMismatch):
    """Stored embeddings come from another model or dimension than the query provider's."""


class EmbedderUnavailable(Exception):
    """The embedding provider could not embed the queries (not configured, or failed)."""


@dataclass(frozen=True)
class HitMessage:
    message_id: int
    sender_name: str | None
    date: datetime
    text: str
    link: str | None
    truncated: bool = False  # compact: `text` was cut to COMPACT_CHARS


@dataclass(frozen=True)
class Hit:
    """SPEC §5.5 `Hit`; `window_id` is None for a single-message hit."""

    window_id: int | None
    chat_id: int
    chat_title: str
    tags: list[str]
    start_date: datetime
    end_date: datetime
    score: float
    messages: list[HitMessage]
    highlights: list[str]
    # why `search` returned it: the legs (`keyword`, `trigram`, `semantic`), the positions
    # in `queries` that found it, and its best cosine distance on the semantic leg
    matched_by: tuple[str, ...] = ()
    matched_message_ids: tuple[int, ...] = ()  # the messages the keyword legs matched
    matched_queries: tuple[int, ...] = ()
    semantic_distance: float | None = None
    omitted_messages: int = 0  # compact: the window's messages left out (`get_context`)

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready: dates as ISO 8601 strings, in UTC."""
        return {
            "window_id": self.window_id,
            "chat_id": self.chat_id,
            "chat_title": self.chat_title,
            "tags": self.tags,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "score": self.score,
            "messages": [
                {
                    "message_id": m.message_id,
                    "sender_name": m.sender_name,
                    "date": m.date.isoformat(),
                    "text": m.text,
                    "link": m.link,
                    **({"truncated": True} if m.truncated else {}),
                }
                for m in self.messages
            ],  # fmt: skip
            "highlights": self.highlights,
            "matched_by": list(self.matched_by),
            "matched_message_ids": list(self.matched_message_ids),
            "matched_queries": list(self.matched_queries),
            "semantic_distance": self.semantic_distance,
            "omitted_messages": self.omitted_messages,
        }


def compacted(hit: Hit, keep: set[int] | None) -> Hit:
    """`hit` for a chat assistant: the messages in `keep` and COMPACT_NEIGHBOURS around each
    (None or none of them present: the first COMPACT_MESSAGES, a sample rather than the
    relevant part), texts over COMPACT_CHARS cut to at most that many characters plus "…"
    and marked `truncated`, and `omitted_messages` counting what was left out."""
    positions = [i for i, m in enumerate(hit.messages) if keep and m.message_id in keep]
    if positions:
        wanted = {j for i in positions
                  for j in range(i - COMPACT_NEIGHBOURS, i + COMPACT_NEIGHBOURS + 1)}  # fmt: skip
        kept = [m for i, m in enumerate(hit.messages) if i in wanted]
    else:
        kept = hit.messages[:COMPACT_MESSAGES]
    cut = [
        replace(m, text=m.text[:COMPACT_CHARS].rstrip() + "…", truncated=True)
        if len(m.text) > COMPACT_CHARS else m
        for m in kept
    ]  # fmt: skip
    return replace(hit, messages=cut, omitted_messages=len(hit.messages) - len(kept))


def link(kind: str, chat_id: int, username: str | None, server_message_id: int) -> str | None:
    """A t.me link to a message (SPEC §5.5), from its server id, never the TDLib id.

    Supergroups and channels link by username when they have one, else by internal id
    (`chat_id` without `-100`). Basic groups and private chats have no deep link: None.
    """
    if kind not in ("supergroup", "channel"):
        return None
    if username:
        return f"https://t.me/{username}/{server_message_id}"
    internal = -chat_id - 10**12  # -100XXXXXXXXXX -> XXXXXXXXXX
    if internal <= 0:
        return None
    return f"https://t.me/c/{internal}/{server_message_id}"


@dataclass(frozen=True)
class MessageHit:
    """One keyword-leg result; `fts` and `trgm` say which leg matched (None = did not)."""

    chat_id: int
    message_id: int
    window_id: int | None
    deleted: bool
    fts: float | None
    trgm: float | None


def _bound(value: date | None, end: bool) -> datetime | None:
    """The filter bound on `message.date`, `>=` for `since` and `<` for `until` (`end`).

    A `date` covers its whole UTC day; a `datetime` is inclusive (timestamps are whole
    microseconds, so the exclusive bound is one microsecond later) and UTC when it has no
    zone, never the session's TimeZone."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value + timedelta(microseconds=1) if end else value
    day = datetime.combine(value, time(), tzinfo=UTC)
    return day + timedelta(days=1) if end else day


def resolve_chats(
    conn: Conn, tags: list[str] | None, chat_ids: list[int] | None
) -> list[int] | None:
    """Chat filter: chats with any of `tags` **confirmed**, intersected with `chat_ids`."""
    allowed = None if chat_ids is None else set(chat_ids)
    if tags is not None:
        rows = conn.execute(
            "SELECT DISTINCT chat_id FROM chat_tag WHERE status = 'confirmed' AND tag = ANY(%s)",
            (tags,),
        )
        tagged = {cast(int, r[0]) for r in rows}
        allowed = tagged if allowed is None else allowed & tagged
    return None if allowed is None else sorted(allowed)


# websearch_to_tsquery syntax, read left to right: a quoted phrase (an unclosed quote runs to
# the end), and a `-` at the start of a word or phrase negates it (`a-b` is one hyphenated
# word, so `-a-b` negates the compound; a `-` inside a phrase negates nothing).
PHRASE = r'"[^"]*(?:"|$)'
TERM = re.compile(rf'(?<!\S)-(?P<neg>{PHRASE}|[^\s"]+)|(?P<phrase>{PHRASE})|(?P<word>[^\s"]+)')


def negated(query: str) -> str | None:
    """The query's negated words and phrases as one websearch OR query, or None."""
    terms = [m["neg"] for m in TERM.finditer(query) if m["neg"]]
    return " OR ".join(terms) if terms else None


def positive(query: str) -> str:
    """The query without its negated words and phrases."""
    return " ".join(m[0] for m in TERM.finditer(query) if not m["neg"])


def short_tokens(conn: Conn, query: str) -> list[str]:
    """The query's tokens of up to `SHORT_TOKEN` characters that no config treats as a stop
    word (`de`, `и`, `the` would otherwise match nearly every message). Negated words and
    quoted phrases are left to the FTS leg."""
    plain = " ".join(m["word"] for m in TERM.finditer(query) if m["word"])
    tokens = list(dict.fromkeys(t for t in re.findall(r"\w+", plain) if len(t) <= SHORT_TOKEN))
    if not tokens:
        return []
    kept = conn.execute(
        "SELECT t FROM unnest(%s::text[]) WITH ORDINALITY AS u(t, n)"
        " WHERE length(to_tsvector('russian', t)) > 0 AND length(to_tsvector('spanish', t)) > 0"
        " AND length(to_tsvector('english', t)) > 0 ORDER BY n",
        (tokens,),
    )
    return [cast(str, r[0]) for r in kept]


def leg_query(
    conn: Conn,
    query: str,
    chats: list[int] | None,
    since: datetime | None,
    until: datetime | None,
    include_deleted: bool,
) -> tuple[sql.Composed, dict[str, Any]]:
    """The keyword leg's statement and parameters for one query (see `keyword_leg`)."""
    tokens = short_tokens(conn, query)

    def tsquery(param: str) -> sql.Composable:
        return sql.SQL(" || ").join(
            sql.SQL("websearch_to_tsquery({}, %({})s)").format(sql.Literal(c), sql.SQL(param))
            for c in CONFIGS
        )

    params: dict[str, Any] = {"q": positive(query), "k": K}
    # A negated term excludes a message when it matches in any config: OR-ing whole per-config
    # queries instead would exclude it only where all three configs stem it the same way.
    neg = negated(query)
    if neg is not None:
        params["neg"] = neg
    kept = sql.SQL("NOT m.tsv @@ q.neg") if neg is not None else sql.SQL("TRUE")
    if TERM.search(params["q"]):
        fts_match, fts_score = sql.SQL("m.tsv @@ q.tsq"), sql.SQL("ts_rank_cd(m.tsv, q.tsq)")
    else:  # only negations: every message they do not match, unranked
        fts_match, fts_score = sql.SQL("TRUE"), sql.SQL("0::real")
    trgm_match = sql.SQL(" OR ").join(
        sql.SQL("{} <<%% m.text").format(sql.Literal(t)) for t in tokens
    ) if tokens else sql.SQL("false")  # fmt: skip
    trgm_score = sql.SQL("greatest({})").format(
        sql.SQL(", ").join(
            sql.SQL("strict_word_similarity({}, m.text)").format(sql.Literal(t)) for t in tokens
        )
    ) if tokens else sql.SQL("NULL::real")  # fmt: skip
    filters = [sql.SQL("TRUE")]
    if chats is not None:
        filters.append(sql.SQL("m.chat_id = ANY(%(chats)s)"))
        params["chats"] = chats
    if since is not None:
        filters.append(sql.SQL("m.date >= %(since)s"))
        params["since"] = since
    if until is not None:
        filters.append(sql.SQL("m.date < %(until)s"))
        params["until"] = until
    if not include_deleted:
        filters.append(sql.SQL("m.deleted_at IS NULL"))
    stmt = sql.SQL(
        "SELECT m.chat_id, m.message_id, w.window_id, m.deleted_at IS NOT NULL, h.fts, h.trgm"
        " FROM message m, (SELECT {tsq} AS tsq, {neg_tsq} AS neg) q,"
        " LATERAL (SELECT CASE WHEN {fts_match} THEN {fts_score} END AS fts,"
        "   CASE WHEN {trgm_match} THEN {trgm_score} END AS trgm) h"
        ' LEFT JOIN LATERAL (SELECT window_id FROM "window" w'
        "   WHERE m.deleted_at IS NULL AND w.chat_id = m.chat_id"
        "   AND w.message_ids @> ARRAY[m.message_id] LIMIT 1) w ON true"
        " WHERE ({fts_match} OR {trgm_match}) AND {kept} AND {filters}"
        " ORDER BY h.fts DESC NULLS LAST, h.trgm DESC NULLS LAST, m.date DESC, m.message_id DESC"
        " LIMIT %(k)s"
    ).format(
        tsq=tsquery("q"),
        neg_tsq=tsquery("neg") if neg is not None else sql.SQL("NULL::tsquery"),
        fts_match=fts_match,
        fts_score=fts_score,
        kept=kept,
        trgm_match=trgm_match,
        trgm_score=trgm_score,
        filters=sql.SQL(" AND ").join(filters),
    )
    return stmt, params


def keyword_leg(
    conn: Conn,
    query: str,
    chats: list[int] | None,
    since: datetime | None,
    until: datetime | None,
    include_deleted: bool,
) -> list[MessageHit]:
    """Top `K` messages for one query: FTS matches by `ts_rank_cd`, then trigram-only
    matches by `strict_word_similarity`. Live messages carry the window that holds them."""
    stmt, params = leg_query(conn, query, chats, since, until, include_deleted)
    with conn.transaction():
        conn.execute(f"SET LOCAL pg_trgm.strict_word_similarity_threshold = {TRGM_THRESHOLD}")
        rows = conn.execute(stmt, params).fetchall()
    return [
        MessageHit(
            chat_id=cast(int, r[0]),
            message_id=cast(int, r[1]),
            window_id=cast(int | None, r[2]),
            deleted=cast(bool, r[3]),
            fts=cast(float | None, r[4]),
            trgm=cast(float | None, r[5]),
        )
        for r in rows
    ]


def semantic_query(
    vec: list[float], chats: list[int] | None, since: datetime | None, until: datetime | None
) -> tuple[sql.Composed, dict[str, Any]]:
    """The semantic leg's statement and parameters (see `semantic_leg`)."""
    params: dict[str, Any] = {"vec": embed._literal(vec), "k": K}
    filters = [sql.SQL("TRUE")]
    if chats is not None:
        filters.append(sql.SQL("w.chat_id = ANY(%(chats)s)"))
        params["chats"] = chats
    if since is not None:  # windows overlapping [since, until)
        filters.append(sql.SQL("w.end_date >= %(since)s"))
        params["since"] = since
    if until is not None:
        filters.append(sql.SQL("w.start_date < %(until)s"))
        params["until"] = until
    stmt = sql.SQL(
        "SELECT e.window_id, e.embedding <=> %(vec)s::vector AS distance"
        ' FROM window_embedding e JOIN "window" w USING (window_id)'
        " WHERE {filters} ORDER BY e.embedding <=> %(vec)s::vector LIMIT %(k)s"
    ).format(filters=sql.SQL(" AND ").join(filters))
    return stmt, params


def semantic_leg(
    conn: Conn,
    vec: list[float],
    chats: list[int] | None,
    since: datetime | None,
    until: datetime | None,
    *,
    limit: int,
    ef_search: int = EF_SEARCH,
) -> tuple[list[tuple[int, float]], bool]:
    """Top `K` `(window_id, distance)` by cosine distance, and whether the exact pass ran.

    The HNSW pass can return fewer rows than exist under a selective filter; when it returns
    fewer than `limit`, the same statement runs again with index scans off, so Postgres
    scans the filtered set exactly whatever the session's planner settings are.
    """
    stmt, params = semantic_query(vec, chats, since, until)
    # Read-only, so rolled back: inside the caller's transaction this is a savepoint, and
    # only rolling it back (not releasing it) undoes its SET LOCALs for the next leg.
    with conn.transaction(force_rollback=True):
        conn.execute(sql.SQL("SET LOCAL hnsw.ef_search = {}").format(sql.Literal(ef_search)))
        rows = conn.execute(stmt, params).fetchall()
        exact = len(rows) < limit
        if exact:
            conn.execute("SET LOCAL enable_indexscan = off")
            conn.execute("SET LOCAL enable_bitmapscan = off")
            conn.execute("SET LOCAL enable_seqscan = on")
            rows = conn.execute(stmt, params).fetchall()
    # ties broken here: a second ORDER BY key in SQL would rule out the HNSW index scan
    hits = sorted((cast(float, r[1]), cast(int, r[0])) for r in rows)
    return [(window_id, distance) for distance, window_id in hits], exact


def _query_vectors(
    conn: Conn, positives: list[str], embedder: Embedder | None
) -> list[list[float]]:
    """One vector per positive query, or none at all when nothing is embedded yet.

    Raises `ModelMismatch` when the stored embeddings are not the provider's, and
    `EmbedderUnavailable` when the provider cannot be built, fails, or answers with another
    number of vectors, another dimension or a non-finite value.
    """
    stored = conn.execute("SELECT model, dim FROM window_embedding LIMIT 1").fetchone()
    if stored is None:
        return []  # the semantic leg would find nothing: embed (and pay for) nothing
    if embedder is None:
        try:
            embedder = embed.provider(Settings.from_env())
        except ValueError as exc:
            raise EmbedderUnavailable(str(exc)) from None
    if (stored[0], stored[1]) != (embedder.model, embedder.dim):
        raise ModelMismatch(
            f"the archive is embedded with {stored[0]} ({stored[1]} dims) but the provider is"
            f" {embedder.model} ({embedder.dim} dims): run `tgi embed --reindex` or search"
            ' with mode="keyword"'
        )
    try:
        vectors = embedder.embed_queries(positives)
    except Exception as exc:
        log.warning("search embedder failed: %s", type(exc).__name__)
        raise EmbedderUnavailable(f"the embedding provider failed ({type(exc).__name__})") from None
    if len(vectors) != len(positives) or any(
        len(v) != stored[1] or not all(math.isfinite(x) for x in v) for v in vectors
    ):
        log.warning("search embedder returned malformed vectors")
        raise EmbedderUnavailable("the embedding provider returned malformed vectors")
    return vectors


def _highlight(text: str, query_tokens: list[str]) -> str:
    """Up to `HIGHLIGHT_CHARS` of `text` around the first query token found in it."""
    if len(text) <= HIGHLIGHT_CHARS:
        return text
    lower = text.lower()
    found = [i for t in query_tokens if (i := lower.find(t.lower())) >= 0]
    start = max(0, min(found, default=0) - HIGHLIGHT_CHARS // 4)
    start = min(start, len(text) - HIGHLIGHT_CHARS)
    snippet = text[start : start + HIGHLIGHT_CHARS]
    return ("…" if start else "") + snippet + ("…" if start + HIGHLIGHT_CHARS < len(text) else "")


# A hit's key: ("w", window_id) for a window, ("m", chat_id, message_id) for a single message.
Key = tuple[Any, ...]


def search(
    conn: Conn,
    queries: list[str],
    *,
    tags: list[str] | None = None,
    chat_ids: list[int] | None = None,
    since: date | None = None,
    until: date | None = None,
    mode: Mode = "hybrid",
    limit: int = 20,
    include_deleted: bool = False,
    embedder: Embedder | None = None,
    compact: bool = False,
) -> list[Hit]:
    """SPEC §5.5 `search`: `limit` windows by reciprocal-rank fusion over legs and queries.

    Each hit says why it was found (`matched_by`, `matched_queries`, `semantic_distance`);
    with `compact` it is cut down for a chat assistant (`compacted`).

    `since`/`until` bound `message.date` inclusively (a `date` covers its whole UTC day),
    and the semantic leg's windows by overlap. Tombstoned messages are left out unless
    `include_deleted`, and then come back as single-message keyword hits, as does a matching
    message that no window holds yet. `embedder` defaults to `embed.provider(settings)`,
    built only when a query needs embedding (never in keyword mode).
    """
    if mode not in ("hybrid", "keyword", "semantic"):
        raise ValueError("mode must be hybrid, keyword or semantic")
    numbered = [(i, q.strip()) for i, q in enumerate(queries) if q.strip()]  # caller's positions
    queries = [q for _, q in numbered]
    if not queries:
        raise ValueError("at least one non-empty query is needed")
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    lo, hi = _bound(since, end=False), _bound(until, end=True)
    if lo is not None and hi is not None and lo >= hi:
        raise BadRange("since is after until")
    chats = resolve_chats(conn, tags, chat_ids)
    if chats == []:
        return []

    # Each leg of each query is one ranked list of (key, matched message, legs, distance),
    # tagged with the query's position; keyword lists come first, so on a tie in score the
    # keyword hit seen first wins, as in keyword mode.
    vectors: list[tuple[int, list[float]]] = []
    if mode != "keyword":
        positives = [(i, p) for i, p in ((i, positive(q).strip()) for i, q in numbered) if p]
        if positives:  # a negation-only query gets no semantic leg and embeds nothing
            embedded = _query_vectors(conn, [p for _, p in positives], embedder)
            vectors = [(i, v) for (i, _), v in zip(positives, embedded, strict=False)]
    Entry = tuple[Key, tuple[int, int] | None, tuple[str, ...], float | None]
    lists: list[tuple[int, list[Entry]]] = []
    if mode != "semantic":
        for i, q in numbered:
            lists.append((i, [
                (("w", m.window_id) if m.window_id is not None else ("m", m.chat_id, m.message_id),
                 (m.chat_id, m.message_id),
                 ("keyword",) * (m.fts is not None) + ("trigram",) * (m.trgm is not None),
                 None)
                for m in keyword_leg(conn, q, chats, lo, hi, include_deleted)
            ]))  # fmt: skip
    for i, vec in vectors:
        rows, _ = semantic_leg(conn, vec, chats, lo, hi, limit=limit)
        lists.append((i, [(("w", w), None, ("semantic",), d) for w, d in rows]))

    scores: dict[Key, float] = {}
    matched: dict[Key, list[tuple[int, int]]] = {}
    legs: dict[Key, set[str]] = {}
    found_by: dict[Key, set[int]] = {}
    distance: dict[Key, float] = {}
    for query_index, ranked_list in lists:
        seen: set[Key] = set()
        for key, msg, via, dist in ranked_list:
            matched.setdefault(key, [])
            if msg is not None and msg not in matched[key]:
                matched[key].append(msg)
            legs.setdefault(key, set()).update(via)
            found_by.setdefault(key, set()).add(query_index)
            if dist is not None:
                distance[key] = min(dist, distance.get(key, dist))
            if key not in seen:  # a window's rank in this list is its best message's
                seen.add(key)
                scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + len(seen))
    ranked = sorted(scores, key=lambda k: -scores[k])[:limit]  # stable: first seen wins ties
    tokens = [t for q in queries for t in re.findall(r"\w+", q)]
    order = ("keyword", "trigram", "semantic")
    hits = []
    for key in ranked:
        hit = _hit(conn, key, scores[key], matched[key], tokens, include_deleted)
        if hit is None:
            continue
        hit = replace(
            hit,
            matched_by=tuple(leg for leg in order if leg in legs[key]),
            matched_message_ids=tuple(
                m.message_id for m in hit.messages if (hit.chat_id, m.message_id) in matched[key]
            ),
            matched_queries=tuple(sorted(found_by[key])),
            semantic_distance=distance.get(key),
        )
        if compact:
            hit = compacted(hit, {mid for _, mid in matched[key]})
        hits.append(hit)
    log.info("search mode=%s queries=%s legs=%s hits=%s", mode, len(queries), len(lists), len(hits))
    return hits


@dataclass(frozen=True)
class Recent:
    """`recent`'s answer: the range it covered, `since` ≤ message date < `until`, the
    server's `now`, the hits (their `score` carries no meaning here), and `next_cursor` for
    the next page (None: no more)."""

    now: datetime
    since: datetime
    until: datetime
    hits: list[Hit]
    next_cursor: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "now": self.now.astimezone(UTC).isoformat(),
            "since": self.since.astimezone(UTC).isoformat(),
            "until": self.until.astimezone(UTC).isoformat(),
            "hits": [h.as_dict() for h in self.hits],
            "next_cursor": self.next_cursor,
        }


def parse_when(text: str) -> date:
    """A bound as a caller writes it: `YYYY-MM-DD` is a date (a whole UTC day), anything
    longer an ISO date-time (`T` or a space before the time; UTC without a zone).
    ValueError otherwise."""
    text = text.strip()
    return datetime.fromisoformat(text) if len(text) > 10 else date.fromisoformat(text)


def _utc(value: date) -> datetime:
    """A `recent` range end: a date-time is an exact instant (UTC without a zone), a date
    the start of that UTC day."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.combine(value, time(), tzinfo=UTC)


def _cursor(
    lo: datetime, hi: datetime, tags: list[str] | None, chat_ids: list[int] | None,
    compact: bool, end_date: datetime, window_id: int,
) -> str:  # fmt: skip
    """The next page: the frozen range, filters and `compact`, and the last window's
    position. Opaque to callers, who pass it back as it is."""
    state = {"since": lo.isoformat(), "until": hi.isoformat(), "tags": tags,
             "chat_ids": chat_ids, "compact": compact,
             "end": end_date.astimezone(UTC).isoformat(), "window": window_id}  # fmt: skip
    return base64.urlsafe_b64encode(json.dumps(state, separators=(",", ":")).encode()).decode()


def _parse_cursor(
    cursor: str,
) -> tuple[datetime, datetime, list[str] | None, list[int] | None, bool, datetime, int]:
    try:
        state = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        if not isinstance(state, dict):
            raise ValueError
        moments = [datetime.fromisoformat(state[k]) for k in ("since", "until", "end")]
        tags, chat_ids, window_id = state["tags"], state["chat_ids"], state["window"]
        compact = state.get("compact", False)

        def whole(value: object) -> bool:  # an int, not a bool
            return isinstance(value, int) and not isinstance(value, bool)

        def listed(value: object, ok: Callable[[object], bool]) -> bool:
            return value is None or isinstance(value, list) and all(map(ok, value))

        if (
            any(m.tzinfo is None for m in moments)
            or not whole(window_id)
            or not listed(tags, lambda t: isinstance(t, str))
            or not listed(chat_ids, whole)
            or not isinstance(compact, bool)
        ):
            raise ValueError
        return moments[0], moments[1], tags, chat_ids, compact, moments[2], window_id
    except (ValueError, TypeError, KeyError, binascii.Error, UnicodeDecodeError):
        raise ValueError("cursor is not one this server returned") from None


def recent(
    conn: Conn,
    *,
    since: date | None = None,
    until: date | None = None,
    tags: list[str] | None = None,
    chat_ids: list[int] | None = None,
    limit: int = 20,
    cursor: str | None = None,
    now: datetime | None = None,
    compact: bool = False,
) -> Recent:
    """Conversation windows with messages posted in `since` ≤ date < `until`, newest
    activity first: no query, so "what's new" needs no topic. Unlike `search`, `until` is
    exclusive (a date: up to the end of that UTC day) and defaults to `now`, so consecutive
    ranges meet without gap or overlap and nothing posted after `now` is listed.

    A hit keeps only its messages inside the range (`get_context` widens it); windows are
    ordered by their last message, then id. `next_cursor` (set when the page was full)
    carries the range, the filters and `compact`: with `cursor`, they come from it and
    `since`, `until`, `tags`, `chat_ids` and `compact` are ignored. A page can hold fewer
    than `limit` hits while more follow (a window overlapping the range with no message inside it is
    skipped); only `next_cursor` None means the end. Paging runs over live data: a sync
    that moves a window's last message can make it repeat or drop out between pages.
    Messages a sync has stored but not yet put in a window, and tombstoned ones, are not
    listed."""
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}")
    now = now or datetime.now(UTC)
    after: tuple[datetime, int] | None = None
    if cursor is not None:
        lo, hi, tags, chat_ids, compact, *position = _parse_cursor(cursor)
        after = (cast(datetime, position[0]), cast(int, position[1]))
    else:
        if since is None:
            raise ValueError("since is needed without a cursor")
        lo = _utc(since)
        hi = now if until is None else (
            _utc(until) if isinstance(until, datetime) else _utc(until) + timedelta(days=1)
        )  # fmt: skip
    if lo >= hi:
        raise BadRange("since is not before until")
    chats = resolve_chats(conn, tags, chat_ids)
    if chats == []:
        return Recent(now, lo, hi, [], None)
    filters = [sql.SQL("w.end_date >= %(lo)s"), sql.SQL("w.start_date < %(hi)s")]
    params: dict[str, Any] = {"lo": lo, "hi": hi, "n": limit}
    if chats is not None:
        filters.append(sql.SQL("w.chat_id = ANY(%(chats)s)"))
        params["chats"] = chats
    if after is not None:
        filters.append(sql.SQL("(w.end_date, w.window_id) < (%(c_date)s, %(c_id)s)"))
        params["c_date"], params["c_id"] = after
    # One query: the page of windows with each one's chat and live in-range messages, one
    # row per message. The LEFT JOIN keeps a window with none of them as one row of nulls:
    # no hit, but it counts towards the page and may be the cursor's position.
    rows = conn.execute(
        sql.SQL(
            "WITH page AS (SELECT w.window_id, w.chat_id, w.end_date, w.message_ids"
            ' FROM "window" w WHERE {} ORDER BY w.end_date DESC, w.window_id DESC'
            " LIMIT %(n)s)"
            " SELECT p.window_id, p.end_date, p.chat_id, c.title, c.kind, c.username,"
            " coalesce((SELECT array_agg(t.tag ORDER BY t.tag) FROM chat_tag t"
            " WHERE t.chat_id = c.chat_id AND t.status = 'confirmed'), '{{}}'),"
            " m.message_id, m.server_message_id, m.sender_name, m.date, m.text"
            " FROM page p JOIN chat c ON c.chat_id = p.chat_id"
            " LEFT JOIN message m ON m.chat_id = p.chat_id"
            " AND m.message_id = ANY(p.message_ids) AND m.deleted_at IS NULL"
            " AND m.date >= %(lo)s AND m.date < %(hi)s"
            " ORDER BY p.end_date DESC, p.window_id DESC, m.date, m.message_id"
        ).format(sql.SQL(" AND ").join(filters)),
        params,
    ).fetchall()
    hits, windows, last = [], 0, None
    for _, group in groupby(rows, key=lambda r: r[0]):
        window = list(group)
        windows += 1
        first = window[0]
        last = (cast(datetime, first[1]), cast(int, first[0]))
        if first[7] is None:  # no live message in the range
            continue
        chat_id, kind = cast(int, first[2]), cast(str, first[4])
        username = cast(str | None, first[5])
        messages = [
            HitMessage(
                message_id=cast(int, r[7]),
                sender_name=cast(str | None, r[9]),
                date=cast(datetime, r[10]).astimezone(UTC),
                text=cast(str, r[11]),
                link=link(kind, chat_id, username, cast(int, r[8])),
            )
            for r in window
        ]
        hit = Hit(
            window_id=cast(int, first[0]),
            chat_id=chat_id,
            chat_title=cast(str, first[3]),
            tags=cast(list[str], first[6]),
            start_date=messages[0].date,
            end_date=messages[-1].date,
            score=0.0,
            messages=messages,
            highlights=[],
        )
        # compact: every in-range message, texts cut (a digest needs them all)
        hits.append(compacted(hit, {m.message_id for m in messages}) if compact else hit)
    next_cursor = None
    if windows == limit and last is not None:
        next_cursor = _cursor(lo, hi, tags, chat_ids, compact, *last)
    log.info("recent windows=%s hits=%s more=%s", windows, len(hits), next_cursor is not None)
    return Recent(now, lo, hi, hits, next_cursor)


def _hit(
    conn: Conn,
    key: Key,
    score: float,
    matched: list[tuple[int, int]],
    tokens: list[str],
    include_deleted: bool,
) -> Hit | None:
    """The hit for `key`, or None if a concurrent sync removed it since the leg ran.

    Unless `include_deleted`, tombstoned messages are left out here too: a window not yet
    rebuilt after a sweep (a run that failed before its window step) still lists them.
    """
    if key[0] == "w":
        row = conn.execute(
            'SELECT chat_id, message_ids FROM "window" WHERE window_id = %s', (key[1],)
        ).fetchone()
        if row is None:
            return None
        chat_id, ids, window_id = cast(int, row[0]), cast(list[int], row[1]), cast(int, key[1])
    else:
        chat_id, ids, window_id = cast(int, key[1]), [cast(int, key[2])], None
    chat = conn.execute(
        "SELECT title, kind, username, coalesce((SELECT array_agg(tag ORDER BY tag)"
        " FROM chat_tag t WHERE t.chat_id = c.chat_id AND status = 'confirmed'), '{}')"
        " FROM chat c WHERE chat_id = %s",
        (chat_id,),
    ).fetchone()
    if chat is None:
        return None
    title, kind, username = cast(str, chat[0]), cast(str, chat[1]), cast(str | None, chat[2])
    rows = conn.execute(
        "SELECT message_id, server_message_id, sender_name, date, text FROM message"
        " WHERE chat_id = %s AND message_id = ANY(%s) AND (%s OR deleted_at IS NULL)"
        " ORDER BY date, message_id",
        (chat_id, ids, include_deleted),
    ).fetchall()
    messages = [
        HitMessage(
            message_id=cast(int, r[0]),
            sender_name=cast(str | None, r[2]),
            date=cast(datetime, r[3]).astimezone(UTC),
            text=cast(str, r[4]),
            link=link(kind, chat_id, username, cast(int, r[1])),
        )
        for r in rows
    ]
    if not messages:
        return None
    start, end = messages[0].date, messages[-1].date  # the shown messages' span, in UTC
    texts = {m.message_id: m.text for m in messages}
    highlights = [_highlight(texts[mid], tokens) for _, mid in matched if mid in texts]
    return Hit(
        window_id=window_id,
        chat_id=chat_id,
        chat_title=title,
        tags=cast(list[str], chat[3]),
        start_date=start,
        end_date=end,
        score=score,
        messages=messages,
        highlights=highlights[:MAX_HIGHLIGHTS],
    )
