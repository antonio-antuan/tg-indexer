"""P2.T2: the semantic leg, hybrid fusion and filtered recall (SPEC §5.5 steps 3-4, §7, A7).

`FakeEmbedder` vectors stand in for a provider: texts sharing words are close. The big
fixture inserts chats, messages, windows and `window_embedding` rows directly, with the HNSW
index in place, so the planner and the index behave as they would on a real archive.
"""

import logging
import math
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, A, B, chat, enable
from typer.testing import CliRunner

from tgi import embed, search
from tgi.cli import app
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
WORDS = [
    *("paella", "valencia", "beach", "train", "ticket", "permit", "renewal", "residence"),
    *("card", "office", "cita", "appointment", "visa", "bank", "account", "flat", "rent"),
    *("school", "doctor", "bus", "metro", "market"),
]


class Recording(FakeEmbedder):
    """The fake provider, recording what `embed_queries` is asked to embed."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.queries: list[list[str]] = []

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.queries.append(list(texts))
        return super().embed_queries(texts)


def hourly(texts: list[str], hours: int = 1) -> list[dict[str, Any]]:
    """One message per text, `hours` apart: one window each."""
    n = len(texts)
    return [{"id": i, "date": (NOW - timedelta(hours=hours * (n - i))).isoformat(),
             "text": t} for i, t in enumerate(texts, 1)]  # fmt: skip


SMALL = [
    "best paella recipe in valencia",  # 1: shares paella/valencia
    "train ticket to madrid",
    "residence permit renewal at the office",  # 3: permit + renewal
    "passport renewal appointment",  # 4: renewal, no permit
    "school starts in september",
    "flat for rent near the beach",
    "permit office closed today",  # 7: permit, no renewal
    "doctor appointment on monday",
]


@pytest.fixture
def small(conn: Conn) -> FakeTdClient:
    """Chat A with the eight SMALL windows, chat B with two, all embedded by the fake."""
    td = FakeTdClient({"chats": [
        chat(A, hourly(SMALL)),
        chat(B, hourly(["renewal of the bank card", "metro strike tomorrow"])),
    ], "runs": [{}]})  # fmt: skip
    enable(conn, td, A, B)
    assert sync(conn, td).stats["embedded"] == 10
    return td


def window_of(conn: Conn, chat_id: int, server_id: int) -> int:
    row = conn.execute(
        'SELECT window_id FROM "window" WHERE chat_id = %s AND first_message_id = %s',
        (chat_id, server_id << SHIFT),
    ).fetchone()
    assert row is not None
    return row[0]  # type: ignore[return-value]


def query_vector(text: str) -> list[float]:
    return FakeEmbedder().embed_queries([text])[0]


# --- A7 --------------------------------------------------------------------------------------


def test_a7_a_window_sharing_tokens_is_in_the_semantic_top_5(
    small: FakeTdClient, conn: Conn
) -> None:
    rows, exact = search.semantic_leg(
        conn, query_vector("paella valencia"), None, None, None, limit=5
    )
    assert window_of(conn, A, 1) in [w for w, _ in rows[:5]]
    assert rows[0][0] == window_of(conn, A, 1)
    assert [d for _, d in rows] == sorted(d for _, d in rows)
    assert len(rows) == 10 and not exact  # 10 windows, all found by the first pass


def test_a7_rrf_merges_both_legs(small: FakeTdClient, conn: Conn) -> None:
    hits = search.search(conn, ["renewal"], limit=20)
    keyword = {h.window_id for h in search.search(conn, ["renewal"], mode="keyword")}
    semantic = {h.window_id for h in search.search(conn, ["renewal"], mode="semantic")}
    assert keyword == {window_of(conn, A, 3), window_of(conn, A, 4), window_of(conn, B, 1)}
    assert len(semantic) == 10  # the semantic leg ranks every embedded window
    got = [h.window_id for h in hits]
    assert set(got) == keyword | semantic
    # found by both legs outranks found by one
    assert set(got[:3]) == keyword
    assert min(h.score for h in hits if h.window_id in keyword) > max(
        h.score for h in hits if h.window_id not in keyword)  # fmt: skip
    # one code path: a window's score is the sum of its reciprocal ranks in each leg
    rows, _ = search.semantic_leg(conn, query_vector("renewal"), None, None, None, limit=20)
    sem_rank = {w: i for i, (w, _) in enumerate(rows, 1)}
    kw_rank = {
        h.window_id: i for i, h in enumerate(search.search(conn, ["renewal"], mode="keyword"), 1)
    }
    for h in hits:
        expected = 1 / (60 + sem_rank[h.window_id]) + (
            1 / (60 + kw_rank[h.window_id]) if h.window_id in kw_rank else 0)  # fmt: skip
        assert h.score == pytest.approx(expected)


def test_a7_limit_and_bounds_hold(small: FakeTdClient, conn: Conn) -> None:
    assert len(search.search(conn, ["renewal", "paella"], limit=4)) == 4
    assert len(search.search(conn, ["renewal"], mode="semantic", limit=1)) == 1
    # A's messages are hourly up to NOW (2026-10-01 12:00); B's are the last two hours too.
    day = NOW.date()
    for mode in ("hybrid", "semantic"):
        hits = search.search(conn, ["renewal"], mode=mode, chat_ids=[B])  # type: ignore[arg-type]
        assert {h.chat_id for h in hits} == {B}
        assert search.search(conn, ["renewal"], mode=mode, since=day + timedelta(days=1)) == []  # type: ignore[arg-type]
        early = search.search(conn, ["renewal"], mode=mode, until=day - timedelta(days=1))  # type: ignore[arg-type]
        assert early == []
    # a window is kept when it overlaps the range: A's first window starts at 04:00
    conn.execute(
        'UPDATE "window" SET start_date = %s WHERE window_id = %s',
        (NOW - timedelta(days=3), window_of(conn, A, 1)),
    )
    old = search.search(conn, ["paella"], mode="semantic", until=day - timedelta(days=2))
    assert [h.window_id for h in old] == [window_of(conn, A, 1)]


# --- the big fixture: 2 000 windows in 20 chats, plus one small chat --------------------------

BIG_CHATS = [-1002000000000 - c for c in range(20)]
TINY = -1003000000000


def mix(seed: int) -> str:
    """Four of WORDS, picked deterministically from `seed`."""
    return " ".join(WORDS[(seed * p) % len(WORDS)] for p in (7, 11, 13, 17))


def big(conn: Conn, per_chat: int = 100, tiny: int = 15) -> None:
    """20 chats × `per_chat` windows, and chat TINY with `tiny`, each window one message,
    embedded with `FakeEmbedder` and inserted directly."""
    fake = FakeEmbedder()
    chats = [(c, per_chat) for c in BIG_CHATS] + [(TINY, tiny)]
    for chat_id, n in chats:
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind, enabled)"
            " VALUES (%s, '+0', %s, 'supergroup', true)",
            (chat_id, f"chat {chat_id}"),
        )
        texts = [f"{mix(chat_id * 1000 + i)} w{chat_id}x{i}" for i in range(1, n + 1)]
        vectors = fake.embed_documents(texts)
        with conn.transaction(), conn.cursor() as cur:
            for i, (text, vec) in enumerate(zip(texts, vectors, strict=True), 1):
                when = NOW - timedelta(hours=n - i)
                cur.execute(
                    "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
                    " VALUES (%s, %s, %s, %s, %s)",
                    (chat_id, i << SHIFT, i, when, text),
                )
                cur.execute(
                    'INSERT INTO "window" (chat_id, first_message_id, start_date, end_date,'
                    " message_ids, text, stale) VALUES (%s, %s, %s, %s, %s, %s, false)"
                    " RETURNING window_id",
                    (chat_id, i << SHIFT, when, when, [i << SHIFT], text),
                )
                window_id = cur.fetchone()[0]  # type: ignore[index]
                cur.execute(
                    "INSERT INTO window_embedding (window_id, model, dim, embedding)"
                    " VALUES (%s, 'fake', 1024, %s::vector)",
                    (window_id, embed._literal(vec)),
                )
    conn.execute("ANALYZE")


def true_distances(conn: Conn, vec: list[float], chat_id: int) -> dict[int, float]:
    """Cosine distance from `vec` to every window of `chat_id`, computed in Python."""
    rows = conn.execute(
        'SELECT w.window_id, e.embedding::text FROM "window" w JOIN window_embedding e'
        " USING (window_id) WHERE w.chat_id = %s",
        (chat_id,),
    )
    out = {}
    for window_id, text in rows:
        v = [float(x) for x in text[1:-1].split(",")]  # type: ignore[index]
        dot = sum(a * b for a, b in zip(vec, v, strict=True))
        norm = math.sqrt(sum(a * a for a in vec)) * math.sqrt(sum(b * b for b in v))
        out[window_id] = 1 - dot / norm
    return out  # type: ignore[return-value]


# At 2 000 rows Postgres rightly costs an exact sort below the HNSW scan, so the tests that
# need the index turn off sequential scans and sorts: the index path is then the only cheap
# one, and a statement the index cannot serve (a second ORDER BY key, say) still sorts.
FORCE_HNSW = ("SET LOCAL enable_seqscan = off", "SET LOCAL enable_sort = off")


def plan_of(conn: Conn, stmt: Any, params: dict[str, Any]) -> str:
    return "\n".join(r[0] for r in conn.execute(sql_explain(stmt), params))  # type: ignore[misc]


def test_adversarial_shortfall_reruns_exactly(conn: Conn) -> None:
    big(conn)
    vec = query_vector("paella beach permit renewal")
    chat_id, limit = BIG_CHATS[3], 20
    stmt, params = search.semantic_query(vec, [chat_id], None, None)
    with conn.transaction():
        for setting in FORCE_HNSW:
            conn.execute(setting)
        conn.execute("SET LOCAL hnsw.ef_search = 1")  # what semantic_leg sets for ef_search=1
        assert "Index Scan using window_embedding_hnsw_idx" in plan_of(conn, stmt, params)
        assert len(conn.execute(stmt, params).fetchall()) < limit  # the first pass falls short
        rows, exact = search.semantic_leg(
            conn, vec, [chat_id], None, None, limit=limit, ef_search=1
        )
    assert exact
    assert len(rows) == 100  # the whole chat, >= limit (K = 100)
    expected = true_distances(conn, vec, chat_id)
    assert {w for w, _ in rows} == set(expected)
    for w, d in rows:
        assert d == pytest.approx(expected[w], abs=1e-6)
    by_python = sorted(expected, key=lambda w: (expected[w], w))
    assert [w for w, _ in rows] == by_python


def test_a_partial_first_pass_reruns_exactly(conn: Conn) -> None:
    """A first pass with some rows but fewer than `limit` reruns too, not only an empty
    one: unfiltered, `ef_search` 5 caps the HNSW scan at 5 of the 2 100 windows."""
    big(conn)
    vec = query_vector("paella beach permit renewal")
    limit = 20
    stmt, params = search.semantic_query(vec, None, None, None)
    with conn.transaction():
        for setting in FORCE_HNSW:
            conn.execute(setting)
        conn.execute("SET LOCAL hnsw.ef_search = 5")
        assert "Index Scan using window_embedding_hnsw_idx" in plan_of(conn, stmt, params)
        assert 1 <= len(conn.execute(stmt, params).fetchall()) < limit
        rows, exact = search.semantic_leg(conn, vec, None, None, None, limit=limit, ef_search=5)
    assert exact
    assert len(rows) == search.K


def sql_explain(stmt: Any) -> Any:
    from psycopg import sql

    return sql.SQL("EXPLAIN ") + stmt


def test_explain_unfiltered_semantic_leg_uses_hnsw(conn: Conn) -> None:
    big(conn)
    stmt, params = search.semantic_query(query_vector("paella"), None, None, None)
    with conn.transaction():
        for setting in FORCE_HNSW:
            conn.execute(setting)
        assert "Index Scan using window_embedding_hnsw_idx" in plan_of(conn, stmt, params)
        rows, exact = search.semantic_leg(conn, query_vector("paella"), None, None, None, limit=20)
    assert len(rows) == search.K and not exact  # ef_search 200 finds K windows in one pass


def test_selective_chat_filter_returns_limit_rows(conn: Conn) -> None:
    """Chat TINY holds 15 of 2 015 windows (< 1 %): the filtered search still returns `limit`."""
    big(conn)
    for mode in ("semantic", "hybrid"):
        hits = search.search(conn, ["paella beach"], mode=mode, chat_ids=[TINY], limit=10,  # type: ignore[arg-type]
                             embedder=FakeEmbedder())  # fmt: skip
        assert len(hits) == 10 and {h.chat_id for h in hits} == {TINY}


def test_exact_rerun_restores_the_planner_settings(small: FakeTdClient, db_url: str) -> None:
    """On a non-autocommit connection (as `db.connect()` gives) the leg is a savepoint inside
    the caller's transaction: the exact rerun's settings must not outlive it (CR on #23)."""
    settings = ("hnsw.ef_search", "enable_indexscan", "enable_bitmapscan", "enable_seqscan")

    def current(c: Conn) -> list[object]:
        return [c.execute("SELECT current_setting(%s, true)", (s,)).fetchone()[0]  # type: ignore[index]
                for s in settings]  # fmt: skip

    with psycopg.connect(db_url) as c:
        c.execute("SET enable_seqscan = off")  # a session value, not the default
        before = current(c)
        assert c.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
        rows, exact = search.semantic_leg(c, query_vector("permit"), None, None, None, limit=50)
        assert exact and len(rows) == 10  # 10 windows < limit: the exact pass ran
        assert c.info.transaction_status == psycopg.pq.TransactionStatus.INTRANS
        # hnsw.ef_search exists once pgvector is loaded; 40 is its default
        assert current(c) == ["40", *before[1:]] and before[1:] == ["on", "on", "off"]
        hits = search.search(c, ["permit", "renewal"], mode="semantic", embedder=FakeEmbedder())
        assert hits and current(c) == ["40", "on", "on", "off"]


# --- provider switch, availability -----------------------------------------------------------


def test_provider_switch_is_a_model_mismatch(small: FakeTdClient, conn: Conn) -> None:
    other = Recording(model="other")
    for mode in ("hybrid", "semantic"):
        with pytest.raises(search.ModelMismatch) as err:
            search.search(conn, ["renewal"], mode=mode, embedder=other)  # type: ignore[arg-type]
        assert "fake" in str(err.value) and "other" in str(err.value)
        assert "tgi embed --reindex" in str(err.value) and 'mode="keyword"' in str(err.value)
    assert other.queries == []  # refused before embedding
    assert search.search(conn, ["renewal"], mode="keyword", embedder=other)
    with pytest.raises(search.ModelMismatch):
        search.search(conn, ["renewal"], embedder=FakeEmbedder(dim=512))


def test_keyword_mode_never_builds_a_provider(
    small: FakeTdClient, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_: Any) -> Any:
        raise AssertionError("provider built")

    monkeypatch.setattr(embed, "provider", boom)
    assert search.search(conn, ["renewal"], mode="keyword")


def test_unavailable_provider(
    small: FakeTdClient, conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = FakeEmbedder(fail=lambda t: True)
    for mode in ("hybrid", "semantic"):
        with pytest.raises(search.EmbedderUnavailable):
            search.search(conn, ["renewal"], mode=mode, embedder=failing)  # type: ignore[arg-type]
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")  # and no VOYAGE_API_KEY
    with pytest.raises(search.EmbedderUnavailable, match="VOYAGE_API_KEY"):
        search.search(conn, ["renewal"])
    assert search.search(conn, ["renewal"], mode="keyword")


def test_nothing_embedded_yet_embeds_nothing(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, hourly(SMALL))], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)
    conn.execute("DELETE FROM window_embedding")
    rec = Recording(model="other")  # would mismatch, but there is nothing to compare with
    hybrid = search.search(conn, ["renewal"], embedder=rec)
    assert rec.queries == []
    assert hybrid == search.search(conn, ["renewal"], mode="keyword")
    assert search.search(conn, ["renewal"], mode="semantic", embedder=rec) == []


# --- negation --------------------------------------------------------------------------------


def test_negation_is_stripped_from_the_embedded_text(small: FakeTdClient, conn: Conn) -> None:
    rec = Recording()
    hits = search.search(conn, ["-permit renewal"], embedder=rec)
    assert rec.queries == [["renewal"]]
    permit = {window_of(conn, A, 3), window_of(conn, A, 7)}
    # the keyword leg still excludes the permit messages: none is a matched highlight
    assert not any("permit" in t for h in hits for t in h.highlights)
    keyword = search.search(conn, ["-permit renewal"], mode="keyword")
    assert {h.window_id for h in keyword} == {window_of(conn, A, 4), window_of(conn, B, 1)}
    # documented: the semantic leg ignores negation, so a permit window can still come back
    assert permit & {h.window_id for h in hits}


def test_negation_only_queries(small: FakeTdClient, conn: Conn) -> None:
    rec = Recording()
    only = search.search(conn, ["-permit"], embedder=rec)
    assert rec.queries == []
    assert only == search.search(conn, ["-permit"], mode="keyword") and only
    mixed = search.search(conn, ["-permit", "renewal"], embedder=rec)
    assert rec.queries == [["renewal"]] and mixed
    assert search.search(conn, ["-permit"], mode="semantic", embedder=rec) == []
    assert rec.queries == [["renewal"]]


def test_logs_carry_no_query_or_text(
    small: FakeTdClient, conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    search.search(conn, ["paella renewal"])
    with pytest.raises(search.EmbedderUnavailable):
        search.search(conn, ["paella"], embedder=FakeEmbedder(fail=lambda t: True))
    assert "search mode=hybrid" in caplog.text
    assert "paella" not in caplog.text and "renewal" not in caplog.text


# --- CLI -------------------------------------------------------------------------------------


def test_cli_search_modes(
    small: FakeTdClient, conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    cli = CliRunner()
    default = cli.invoke(app, ["search", "renewal", "--json"])
    hybrid = cli.invoke(app, ["search", "renewal", "--json", "--mode", "hybrid"])
    assert default.exit_code == 0 and default.stdout == hybrid.stdout
    assert len(search.search(conn, ["renewal"])) == default.stdout.count('"window_id"')
    semantic = cli.invoke(app, ["search", "paella", "--mode", "semantic"])
    assert semantic.exit_code == 0 and "best paella recipe" in semantic.stdout
    assert cli.invoke(app, ["search", "x", "--mode", "fuzzy"]).exit_code == 1
    monkeypatch.setenv("EMBED_MODEL", "fake-2")
    mismatch = cli.invoke(app, ["search", "renewal"])
    assert mismatch.exit_code == 3 and "tgi embed --reindex" in mismatch.stderr
    assert cli.invoke(app, ["search", "renewal", "--mode", "keyword"]).exit_code == 0
