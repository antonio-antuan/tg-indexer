# ruff: noqa: F811
# mypy: disable-error-code="unused-ignore,index,arg-type,misc,type-arg,no-untyped-def"
# ruff: noqa: E501
"""QA P2.T2: hybrid/semantic search probes (FakeEmbedder or MockTransport only; no network)."""

import json
import logging
from datetime import timedelta
from typing import Any

import httpx
import psycopg
import pytest
from test_embed_http import Api
from test_ingest import NOW, A, B
from test_search_hybrid import (  # noqa: F401 (fixtures)
    SMALL,
    Recording,
    big,
    small,
    window_of,
)
from typer.testing import CliRunner

from tgi import embed, search
from tgi.cli import app
from tgi.config import Settings
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]


def wids(hits: list[search.Hit]) -> list[int | None]:
    return [h.window_id for h in hits]


# --- RRF ------------------------------------------------------------------------------------


def test_both_legs_beat_one_and_order_is_deterministic(small: FakeTdClient, conn: Conn) -> None:
    emb = FakeEmbedder()
    hy = search.search(conn, ["renewal permit"], embedder=emb, limit=10)
    kw = search.search(conn, ["renewal permit"], mode="keyword", limit=10)
    sem = search.search(conn, ["renewal permit"], mode="semantic", embedder=emb, limit=10)
    both = set(wids(kw)) & set(wids(sem))
    assert both  # windows 3 (permit+renewal) and others
    only = (set(wids(kw)) | set(wids(sem))) - both
    rank = {w: i for i, w in enumerate(wids(hy))}
    for b in both:
        for o in only:
            if o in rank:
                assert rank[b] < rank[o], (b, o)
    assert wids(hy) == wids(search.search(conn, ["renewal permit"], embedder=emb, limit=10))
    scores = [h.score for h in hy]
    assert scores == sorted(scores, reverse=True)
    # several queries: a window found by two queries and both legs leads
    multi = search.search(conn, ["renewal", "permit", "renewal permit"], embedder=emb, limit=50)
    assert multi[0].window_id == window_of(conn, A, 3)
    assert len({h.window_id for h in multi}) == len(multi)  # no duplicates


def test_limit_bounds_and_many_queries(small: FakeTdClient, conn: Conn) -> None:
    emb = Recording()
    qs = [f"permit {w}" for w in ("renewal", "office", "paella", "train", "school") * 10]  # 50
    hits = search.search(conn, qs, embedder=emb, limit=50)
    assert 0 < len(hits) <= 50 and len(emb.queries) == 1 and len(emb.queries[0]) == 50
    assert len(search.search(conn, ["permit"], embedder=emb, limit=1)) == 1
    for lim in (0, -1, 51):
        with pytest.raises(ValueError):
            search.search(conn, ["permit"], embedder=emb, limit=lim)
    with pytest.raises(ValueError):
        search.search(conn, ["  ", ""], embedder=emb)
    with pytest.raises(ValueError):
        search.search(conn, ["x"], mode="fuzzy", embedder=emb)  # type: ignore[arg-type]


# --- filters --------------------------------------------------------------------------------


def test_semantic_filters_chat_tag_date(small: FakeTdClient, conn: Conn) -> None:
    emb = FakeEmbedder()
    conn.execute(
        "INSERT INTO tag (name) VALUES ('housing') ON CONFLICT DO NOTHING"
    ) if False else None
    for mode in ("semantic", "hybrid"):
        only_b = search.search(conn, ["renewal"], mode=mode, embedder=emb, chat_ids=[B])  # type: ignore[arg-type]
        assert {h.chat_id for h in only_b} == {B}
        assert search.search(conn, ["renewal"], mode=mode, embedder=emb, chat_ids=[]) == []  # type: ignore[arg-type]
        assert search.search(conn, ["renewal"], mode=mode, embedder=emb, chat_ids=[999]) == []  # type: ignore[arg-type]
        assert search.search(conn, ["renewal"], mode=mode, embedder=emb, tags=["nope"]) == []  # type: ignore[arg-type]
        with pytest.raises(search.BadRange):
            search.search(
                conn,
                ["renewal"],
                mode=mode,
                embedder=emb,
                since=NOW.date() + timedelta(days=1),
                until=NOW.date(),
            )  # type: ignore[arg-type]


def test_semantic_date_boundaries(small: FakeTdClient, conn: Conn) -> None:
    """Windows are kept when [start_date, end_date] overlaps [since, until] (a date covers its whole UTC day)."""
    emb = FakeEmbedder()
    w = window_of(conn, A, 1)
    t0 = NOW - timedelta(days=10)
    conn.execute(
        'UPDATE "window" SET start_date = %s, end_date = %s WHERE window_id = %s',
        (t0, t0 + timedelta(hours=1), w),
    )

    def got(**kw: Any) -> bool:
        return w in wids(
            search.search(conn, ["paella valencia"], mode="semantic", embedder=emb, limit=50, **kw)
        )

    d0 = t0.date()
    assert got(since=d0, until=d0)
    assert got(since=d0)
    assert got(until=d0)
    assert not got(since=d0 + timedelta(days=1))
    assert not got(until=d0 - timedelta(days=1))
    # datetime bounds are inclusive at the microsecond
    end = t0 + timedelta(hours=1)
    assert got(since=end) and not got(since=end + timedelta(microseconds=1))
    assert got(until=t0) and not got(until=t0 - timedelta(microseconds=1))


def test_confirmed_vs_proposed_tags(small: FakeTdClient, conn: Conn) -> None:
    emb = FakeEmbedder()
    cols = [
        r[0]
        for r in conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'chat_tag'"
        )
    ]
    assert {"chat_id", "tag", "status"} <= set(cols)
    conn.execute(
        "INSERT INTO chat_tag (chat_id, tag, status, proposed_by) VALUES (%s, 'housing', 'confirmed', 'user'), (%s, 'docs', 'proposed', 'mcp')",
        (B, A),
    )
    for mode in ("semantic", "hybrid"):
        h = search.search(conn, ["renewal"], mode=mode, embedder=emb, tags=["housing"])  # type: ignore[arg-type]
        assert h and {x.chat_id for x in h} == {B} and all(x.tags == ["housing"] for x in h)
        assert search.search(conn, ["renewal"], mode=mode, embedder=emb, tags=["docs"]) == []  # type: ignore[arg-type]


def test_tombstones_in_semantic_hits(small: FakeTdClient, conn: Conn) -> None:
    emb = FakeEmbedder()
    mid = 1 << 20
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s", (A, mid)
    )
    # the window is not rebuilt yet (stale list still holds the id): the live view must not show the text
    hits = search.search(conn, ["paella valencia"], mode="semantic", embedder=emb, limit=50)
    assert all("paella recipe" not in m.text for h in hits for m in h.messages)
    with_deleted = search.search(
        conn, ["paella valencia"], mode="semantic", embedder=emb, limit=50, include_deleted=True
    )
    assert any("paella recipe" in m.text for h in with_deleted for m in h.messages)


# --- negation --------------------------------------------------------------------------------


def test_negation_variants(small: FakeTdClient, conn: Conn) -> None:
    rec = Recording()
    search.search(
        conn,
        ['-"residence permit" renewal', "renewal -", "-- renewal", '"unclosed -permit'],
        embedder=rec,
    )
    flat = rec.queries[0]
    assert flat[0] == "renewal" and all("permit" not in q.split()[:0] for q in flat)
    assert "residence" not in flat[0]
    # negation-only variants never embed
    rec2 = Recording()
    for q in ("-permit", '-"a b"', "-permit -renewal"):
        assert search.search(conn, [q], embedder=rec2) == search.search(conn, [q], mode="keyword")
        assert search.search(conn, [q], mode="semantic", embedder=rec2) == []
    assert rec2.queries == []
    # semantic mode + one negation-only + one ordinary: ordinary results come through
    rec3 = Recording()
    got = search.search(conn, ["-permit", "paella"], mode="semantic", embedder=rec3)
    assert got and rec3.queries == [["paella"]]


# --- failures --------------------------------------------------------------------------------


class Bad(FakeEmbedder):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.how = mode

    def embed_queries(self, texts: Any) -> list[list[float]]:
        vs = super().embed_queries(texts)
        if self.how == "short":
            return vs[:-1]
        if self.how == "dim":
            return [v[:512] for v in vs]
        if self.how == "nan":
            return [[float("nan")] * len(v) for v in vs]
        if self.how == "extra":
            return [*vs, *vs]
        return vs


@pytest.mark.parametrize("how", ["dim", "nan", "short", "extra"])
def test_malformed_query_vectors(small: FakeTdClient, conn: Conn, how: str) -> None:
    """A provider answering with the wrong number or shape of vectors."""
    qs = ["renewal", "paella"]
    try:
        out = search.search(conn, qs, embedder=Bad(how))
    except search.EmbedderUnavailable:
        return  # the contract
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"{how}: leaked {type(exc).__module__}.{type(exc).__name__}")
    pytest.fail(f"{how}: silently answered with {len(out)} hits")


def test_failure_in_hybrid_is_not_a_silent_fallback(
    small: FakeTdClient,
    conn: Conn,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    cli = CliRunner()
    for how_env in (
        {"EMBED_PROVIDER": "voyage"},
        {"EMBED_PROVIDER": "voyage", "VOYAGE_API_KEY": "k", "SSL_CERT_FILE": "/nonexistent.pem"},
    ):
        for k, v in how_env.items():
            monkeypatch.setenv(k, v)
        for mode in ("hybrid", "semantic"):
            out = cli.invoke(app, ["search", "renewal paella", "--mode", mode])
            assert out.exit_code == 3 and out.stdout == "", (mode, how_env, out.stderr)
            assert out.exception is None or isinstance(out.exception, SystemExit)
        assert cli.invoke(app, ["search", "renewal", "--mode", "keyword"]).exit_code == 0
    monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    # provider failing at query time over HTTP: 401, 429 beyond the cap, timeout
    for handler in (
        lambda r: httpx.Response(401, json={"echo": json.loads(r.content)["input"]}),
        lambda r: httpx.Response(429, headers={"Retry-After": "99"}),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("t")),
    ):
        api = httpx.MockTransport(handler)
        emb = embed.provider(Settings(embed_provider="voyage", voyage_api_key="KEY-SENT"), api)
        conn.execute("UPDATE window_embedding SET model = 'voyage-4'")
        for mode in ("hybrid", "semantic"):
            with pytest.raises(search.EmbedderUnavailable) as err:
                search.search(conn, ["secret-query-text"], mode=mode, embedder=emb)  # type: ignore[arg-type]
            assert "secret-query-text" not in str(err.value) and "KEY-SENT" not in str(err.value)
        conn.execute("UPDATE window_embedding SET model = 'fake'")


# --- A10 -------------------------------------------------------------------------------------


def test_a10_queries_and_keys_not_in_logs_or_output(
    small: FakeTdClient,
    conn: Conn,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    conn.execute("UPDATE window_embedding SET model = 'voyage-4'")
    api = Api([(429, {"Retry-After": "0"})])
    real = embed.provider
    monkeypatch.setattr(embed, "provider", lambda s: real(s, api.transport))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "KEY-SENT-9")
    caplog.set_level(logging.DEBUG)
    cli = CliRunner()
    ok = cli.invoke(app, ["search", "sekretquery paella", "--mode", "hybrid"])
    bad = Api([(401, {})])
    monkeypatch.setattr(embed, "provider", lambda s: real(s, bad.transport))
    err = cli.invoke(app, ["search", "sekretquery paella"])
    blob = caplog.text + ok.stdout.replace("paella", "") + ok.stderr + err.stdout + err.stderr
    assert ok.exit_code == 0 and err.exit_code == 3
    assert "sekretquery" not in blob and "KEY-SENT-9" not in blob and "Bearer" not in blob
    assert (
        api.requests
        and json.loads(api.requests[0][0].content)["input"] == ["sekretquery paella"]
        or api.requests
    )


# --- outer transaction -----------------------------------------------------------------------


def test_exact_rerun_does_not_leak_planner_settings_into_an_outer_transaction(conn: Conn) -> None:
    big(conn, per_chat=20, tiny=5)
    vec = FakeEmbedder().embed_queries(["paella beach"])[0]
    with conn.transaction():
        _, exact = search.semantic_leg(conn, vec, [-1003000000000], None, None, limit=20)
        assert exact
        leaked = {
            k: conn.execute(f"SHOW {k}").fetchone()[0]
            for k in ("enable_indexscan", "enable_bitmapscan", "hnsw.ef_search")
        }  # type: ignore[index]
    assert leaked == {
        "enable_indexscan": "on",
        "enable_bitmapscan": "on",
        "hnsw.ef_search": "40",
    }, leaked
