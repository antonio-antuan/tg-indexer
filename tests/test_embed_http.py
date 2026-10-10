"""P2.T1: the voyage and openai providers over httpx, against `httpx.MockTransport`.

No test reaches the network: every request is answered by `Api`, which checks nothing
itself but records each request and replays scripted statuses before answering 200 with
fake vectors and a `usage.total_tokens` of 5 per text.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from test_embed import embeddings, hourly, stale, sync
from test_ingest import A, chat, enable
from typer.testing import CliRunner

from tgi import embed
from tgi.cli import app
from tgi.config import Settings
from tgi.embed import FakeEmbedder, OpenAIEmbedder, ProviderError, VoyageEmbedder
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
TOKENS_PER_TEXT = 5
KEYS = {"VOYAGE_API_KEY": "voyage-key-SENTINEL-4", "OPENAI_API_KEY": "openai-key-SENTINEL-5"}


class Api:
    """The embeddings API: scripted (status, headers) replies first, then 200s.

    `fail_on` answers 500 to any request whose input contains that text.
    """

    def __init__(
        self, script: list[tuple[int, dict[str, str]]] | None = None, fail_on: str | None = None
    ) -> None:
        self.script, self.fail_on = list(script or []), fail_on
        self.requests: list[tuple[httpx.Request, dict[str, Any]]] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append((request, body))
        error = {"detail": "the provider may quote input: " + " ".join(body["input"])}
        if self.script:
            status, headers = self.script.pop(0)
            return httpx.Response(status, headers=headers, json=error)
        if self.fail_on and any(self.fail_on in t for t in body["input"]):
            return httpx.Response(500, json=error)
        vectors = FakeEmbedder().embed_documents(body["input"])
        data = [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)]
        return httpx.Response(200, json={
            "object": "list", "data": data[::-1], "model": body["model"],  # any order: by index
            "usage": {"total_tokens": TOKENS_PER_TEXT * len(body["input"])},
        })  # fmt: skip


def voyage(api: Api, **kw: Any) -> VoyageEmbedder:
    return VoyageEmbedder("k", "voyage-4", transport=api.transport, **kw)


def with_transport(monkeypatch: pytest.MonkeyPatch, api: Api) -> None:
    """Make the CLI's `embed.provider(settings)` answer through `api`."""
    real = embed.provider
    monkeypatch.setattr(embed, "provider", lambda settings: real(settings, api.transport))


def windows(conn: Conn, n: int) -> None:
    """n single-message windows in chat A, all stale and none embedded."""
    td = FakeTdClient({"chats": [chat(A, hourly(n))], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)
    conn.execute("DELETE FROM window_embedding")
    conn.execute('UPDATE "window" SET stale = true')


# --- request shape -------------------------------------------------------------------------


def test_voyage_request_shape_and_tokens() -> None:
    api = Api()
    v = VoyageEmbedder("the-key", "voyage-4", transport=api.transport)
    assert (v.model, v.dim, v.tokens_used) == ("voyage-4", 1024, 0)
    docs = v.embed_documents(["a b", "c"])
    query = v.embed_queries(["q"])
    assert docs == FakeEmbedder().embed_documents(["a b", "c"])  # reordered by `index`
    assert query == FakeEmbedder().embed_documents(["q"])
    (r1, b1), (r2, b2) = api.requests
    assert (r1.method, str(r1.url)) == ("POST", "https://api.voyageai.com/v1/embeddings")
    assert r1.headers["Authorization"] == "Bearer the-key"
    assert b1 == {"model": "voyage-4", "input": ["a b", "c"], "input_type": "document",
                  "output_dimension": 1024}  # fmt: skip
    assert b2 == {"model": "voyage-4", "input": ["q"], "input_type": "query",
                  "output_dimension": 1024}  # fmt: skip
    assert v.tokens_used == 3 * TOKENS_PER_TEXT


def test_openai_request_shape_is_the_same_for_both_sides() -> None:
    api = Api()
    o = embed.provider(
        Settings(embed_provider="openai", openai_api_key="the-key"), transport=api.transport
    )
    assert isinstance(o, OpenAIEmbedder)
    assert (o.model, o.dim) == ("text-embedding-3-small", 1024)
    o.embed_documents(["x"])
    o.embed_queries(["x"])
    (r1, b1), (_, b2) = api.requests
    assert str(r1.url) == "https://api.openai.com/v1/embeddings"
    assert r1.headers["Authorization"] == "Bearer the-key"
    assert b1 == b2 == {"model": "text-embedding-3-small", "input": ["x"], "dimensions": 1024}
    assert o.tokens_used == 2 * TOKENS_PER_TEXT


def test_provider_config() -> None:
    v = embed.provider(Settings(embed_provider="voyage", voyage_api_key="k"))
    assert isinstance(v, VoyageEmbedder) and v.model == "voyage-4"
    custom = Settings(embed_provider="voyage", voyage_api_key="k", embed_model="voyage-4-lite")
    assert embed.provider(custom).model == "voyage-4-lite"
    assert embed.provider(Settings(embed_provider="fake")).tokens_used == 0
    assert Settings().embed_provider == "voyage"  # SPEC §3
    assert Settings.from_env({}).embed_provider == "voyage"
    for name, key in (("voyage", "VOYAGE_API_KEY"), ("openai", "OPENAI_API_KEY")):
        with pytest.raises(ValueError, match=f"{key} is not set"):  # at construction time
            embed.provider(Settings(embed_provider=name))
    with pytest.raises(ValueError, match="fake, voyage or openai"):
        embed.provider(Settings(embed_provider="cohere"))

    settings = Settings.from_env({"EMBED_PROVIDER": "voyage", **KEYS})
    assert (settings.voyage_api_key, settings.openai_api_key) == tuple(KEYS.values())
    assert not any(k in repr(settings) for k in KEYS.values())


# --- errors and 429 ------------------------------------------------------------------------


def test_429_within_the_cap_sleeps_once_then_retries() -> None:
    api, sleeps = Api([(429, {"Retry-After": "1"})]), []
    v = voyage(api, retry_after_max=30, sleep=sleeps.append)
    assert v.embed_documents(["a"]) == FakeEmbedder().embed_documents(["a"])
    assert sleeps == [1.0] and len(api.requests) == 2
    assert api.requests[0][1] == api.requests[1][1]
    assert v.tokens_used == TOKENS_PER_TEXT  # only the successful response counts


@pytest.mark.parametrize(
    "script",
    [
        [(429, {"Retry-After": "61"})],  # above the cap
        [(429, {})],  # no Retry-After
        [(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})],  # not seconds
    ],
)
def test_429_beyond_the_cap_raises_without_sleeping(
    script: list[tuple[int, dict[str, str]]],
) -> None:
    api, sleeps = Api(script), []
    with pytest.raises(ProviderError) as err:
        voyage(api, retry_after_max=30, sleep=sleeps.append).embed_documents(["a"])
    assert err.value.status == 429 and sleeps == [] and len(api.requests) == 1


def test_a_second_429_raises() -> None:
    api, sleeps = Api([(429, {"Retry-After": "1"}), (429, {"Retry-After": "1"})]), []
    with pytest.raises(ProviderError):
        voyage(api, sleep=sleeps.append).embed_documents(["a"])
    assert sleeps == [1.0] and len(api.requests) == 2


def test_a_long_429_leaves_that_batch_stale(conn: Conn) -> None:
    windows(conn, 70)  # batches 1-64 and 65-70
    api, sleeps = Api([(429, {"Retry-After": "61"})]), []
    counts = embed.embed_stale(conn, voyage(api, retry_after_max=30, sleep=sleeps.append))
    assert counts == {"embedded": 6, "embed_failed": 64, "embed_stopped": 0}
    assert sleeps == [] and stale(conn) == list(range(1, 65))


def test_a_500_leaves_exactly_that_batch_stale_and_the_run_ok(
    conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    msgs = hourly(150, "SECRET {i}")
    msgs[99]["text"] = "SECRET boom"  # window 100, in the second batch (65-128)
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{}]})
    enable(conn, td, A)
    caplog.set_level(logging.DEBUG)
    result = sync(conn, td, voyage(Api(fail_on="boom")))
    assert (result.status, result.stats["embedded"], result.stats["embed_failed"]) == (
        "ok", 86, 64)  # fmt: skip
    assert stale(conn) == list(range(65, 129))
    assert conn.execute("SELECT status FROM sync_run").fetchall() == [("ok",)]
    assert "SECRET" not in caplog.text
    assert any(r.getMessage() == "embed provider 500 for 64 texts" for r in caplog.records)


# --- the CLI -------------------------------------------------------------------------------


def test_scenario_sync_embeds_every_window_with_voyage(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = {"chats": [chat(A, hourly(70))]}
    enable(conn, FakeTdClient(scenario), A)
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    api = Api()
    with_transport(monkeypatch, api)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    out = CliRunner().invoke(app, ["sync", "--scenario", str(path)])
    assert out.exit_code == 0, out.stderr
    assert " ok " in out.stdout and "embedded=70 embed_failed=0" in out.stdout
    rows = embeddings(conn)
    assert len(rows) == 70 and {r[:2] for r in rows.values()} == {("voyage-4", 1024)}
    assert stale(conn) == [] and [len(b["input"]) for _, b in api.requests] == [64, 6]


def test_cli_missing_key_is_a_config_error_before_any_work(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = {"chats": [chat(A, hourly(5))]}
    enable(conn, FakeTdClient(scenario), A)
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    cli = CliRunner()
    for args in (["sync", "--scenario", str(path)], ["embed"]):
        out = cli.invoke(app, args)
        assert out.exit_code == 1 and "VOYAGE_API_KEY is not set" in out.stderr
    assert conn.execute("SELECT count(*) FROM sync_run").fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM message").fetchone() == (0,)


def test_bad_ssl_cert_file_is_a_config_error(
    conn: Conn, db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, content in (("missing.pem", None), ("garbage.pem", "not a certificate\n")):
        path = tmp_path / name
        if content is not None:
            path.write_text(content)
        monkeypatch.setenv("SSL_CERT_FILE", str(path))
        with pytest.raises(ValueError, match="SSL_CERT_FILE or SSL_CERT_DIR cannot be loaded"):
            embed.provider(Settings(embed_provider="voyage", voyage_api_key=KEYS["VOYAGE_API_KEY"]))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", KEYS["VOYAGE_API_KEY"])
    out = CliRunner().invoke(app, ["embed"])
    assert out.exit_code == 1 and "SSL_CERT_FILE or SSL_CERT_DIR cannot be loaded" in out.stderr
    assert out.exception is None or isinstance(out.exception, SystemExit)  # no traceback
    assert KEYS["VOYAGE_API_KEY"] not in out.output


def test_cli_embed_max_windows(conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    windows(conn, 10)
    api = Api()
    with_transport(monkeypatch, api)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    cli = CliRunner()
    out = cli.invoke(app, ["embed", "--max-windows", "3"])
    assert out.exit_code == 0, out.stderr
    assert out.stdout.strip() == f"embedded=3 embed_failed=0 tokens={3 * TOKENS_PER_TEXT}"
    assert sorted(embeddings(conn)) == [1, 2, 3] and stale(conn) == list(range(4, 11))
    assert [len(b["input"]) for _, b in api.requests] == [3]
    assert cli.invoke(app, ["embed", "--max-windows", "0"]).exit_code == 1


@pytest.mark.parametrize("provider", ["voyage", "openai"])
def test_a10_no_key_or_text_in_logs_or_output(
    provider: str,
    conn: Conn,
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both keys set; a 429 retried, a 500 and successes, at DEBUG including httpx's logs."""
    msgs = hourly(150, "SECRET {i}")
    msgs[99]["text"] = "SECRET boom"
    scenario = {"chats": [chat(A, msgs)]}
    enable(conn, FakeTdClient(scenario), A)
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    api = Api([(429, {"Retry-After": "0"})], fail_on="boom")
    with_transport(monkeypatch, api)
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    monkeypatch.setenv("EMBED_PROVIDER", provider)
    for key, value in KEYS.items():
        monkeypatch.setenv(key, value)
    caplog.set_level(logging.DEBUG)
    out = CliRunner().invoke(app, ["sync", "--scenario", str(path)])
    assert out.exit_code == 0, out.stderr
    assert "embedded=86 embed_failed=64" in out.stdout
    used = KEYS[f"{provider.upper()}_API_KEY"]
    assert {r.headers["Authorization"] for r, _ in api.requests} == {f"Bearer {used}"}
    assert any(r.name == "httpx" for r in caplog.records)  # httpx's own logs were captured
    for forbidden in [*KEYS.values(), "SECRET"]:
        assert forbidden not in caplog.text
        assert forbidden not in out.stdout + out.stderr


def test_embed_stale_max_windows_spans_batches(conn: Conn) -> None:
    windows(conn, 150)
    rec: list[int] = []

    class Counting(FakeEmbedder):
        def embed_documents(self, texts: Any) -> list[list[float]]:
            rec.append(len(texts))
            return super().embed_documents(texts)

    counts = embed.embed_stale(conn, Counting(), max_windows=100)
    assert counts["embedded"] == 100 and rec == [64, 36]
    assert stale(conn) == list(range(101, 151))
