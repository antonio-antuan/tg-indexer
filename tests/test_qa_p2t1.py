# ruff: noqa: E501
"""QA P2.T1: retry edges, batch-stale semantics, --max-windows, A10 and request shape for the
voyage/openai embedders. Every request goes through httpx.MockTransport; no network."""

import json
import logging
from typing import Any

import httpx
import psycopg
import pytest
from test_embed import embeddings, stale
from test_embed_http import KEYS, Api, voyage, windows, with_transport
from typer.testing import CliRunner

from tgi import embed
from tgi.cli import app
from tgi.config import Settings
from tgi.embed import FakeEmbedder, OpenAIEmbedder, ProviderError, VoyageEmbedder

Conn = psycopg.Connection[tuple[object, ...]]


def run(script: list[tuple[int, dict[str, str]]], cap: float = 30) -> tuple[Any, list[float], Api]:
    api, sleeps = Api(script), []
    v = voyage(api, retry_after_max=cap, sleep=sleeps.append)
    try:
        out: Any = v.embed_documents(["a"])
    except ProviderError as exc:
        out = exc
    return out, sleeps, api


# --- retry edges ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "sleeps", "retries"),
    [
        ("0", [0.0], True),  # zero: sleeps 0 and retries
        ("1", [1.0], True),
        ("30", [30.0], True),  # exactly the cap
        ("30.0", [30.0], True),
        ("30.5", [], False),
        ("31", [], False),
        ("-1", [], False),
        ("nan", [], False),
        ("inf", [], False),
        ("", [], False),
        (" 2 ", [2.0], True),
        ("Wed, 21 Oct 2026 07:28:00 GMT", [], False),
    ],
)
def test_retry_after_forms(header: str, sleeps: list[float], retries: bool) -> None:
    out, slept, api = run([(429, {"Retry-After": header})])
    assert slept == sleeps
    assert len(api.requests) == (2 if retries else 1)
    assert (
        (out == FakeEmbedder().embed_documents(["a"]))
        if retries
        else isinstance(out, ProviderError)
    )


def test_missing_header_and_second_429_variants() -> None:
    out, slept, api = run([(429, {})])
    assert isinstance(out, ProviderError) and slept == [] and len(api.requests) == 1
    out, slept, api = run([(429, {"Retry-After": "0"}), (429, {"Retry-After": "0"})])
    assert isinstance(out, ProviderError) and slept == [0.0] and len(api.requests) == 2
    out, slept, api = run([(429, {"Retry-After": "1"}), (500, {})])  # 429 then 5xx: no third try
    assert isinstance(out, ProviderError) and out.status == 500 and len(api.requests) == 2
    out, slept, api = run([(500, {"Retry-After": "1"})])  # Retry-After on a non-429 is ignored
    assert isinstance(out, ProviderError) and slept == [] and len(api.requests) == 1


def test_default_cap_is_30_and_the_retry_budget_is_per_call() -> None:
    assert embed.RETRY_AFTER_MAX == 30
    api, slept = Api([(429, {"Retry-After": "30"})]), []
    v = voyage(api, sleep=slept.append)  # default cap
    assert v.embed_documents(["a"])[0] and slept == [30.0]  # first call: retried once, ok
    api.script.append((429, {"Retry-After": "30"}))
    assert v.embed_documents(["a"])[0] and slept == [30.0, 30.0]  # next call gets its own retry


def test_retry_with_openai_and_tokens() -> None:
    api, slept = Api([(429, {"Retry-After": "2"})]), []
    o = OpenAIEmbedder("k", "text-embedding-3-small", transport=api.transport, sleep=slept.append)
    o.embed_queries(["x", "y"])
    assert slept == [2.0] and o.tokens_used == 10


# --- batch-stale semantics -----------------------------------------------------------------


def custom(handler: Any) -> VoyageEmbedder:
    return VoyageEmbedder("k", "voyage-4", transport=httpx.MockTransport(handler))


def ok_reply(request: httpx.Request, **over: Any) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [
        {"embedding": v, "index": i} for i, v in enumerate(FakeEmbedder().embed_documents(texts))
    ]
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 7}, **over})


@pytest.mark.parametrize(
    "bad",
    [
        lambda r: httpx.Response(200, text="not json"),
        lambda r: httpx.Response(200, json={"data": [], "usage": {"total_tokens": 1}}),  # short
        lambda r: httpx.Response(200, json={"usage": {"total_tokens": 1}}),  # no data
        lambda r: ok_reply(r, usage={}),  # no total_tokens
        lambda r: httpx.Response(
            200,
            json={
                "data": [{"embedding": [0.5] * 512, "index": 0}] * 64,
                "usage": {"total_tokens": 1},
            },
        ),
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("boom")),
        lambda r: (_ for _ in ()).throw(httpx.ReadTimeout("slow")),
        lambda r: httpx.Response(302, headers={"Location": "https://elsewhere.invalid/"}),
        lambda r: httpx.Response(401, json={"error": "bad key"}),
        lambda r: httpx.Response(204),
    ],
    ids=[
        "notjson",
        "short",
        "nodata",
        "nousage",
        "dim512",
        "connect",
        "timeout",
        "redirect",
        "401",
        "204",
    ],
)
def test_any_bad_answer_leaves_only_that_batch_stale(conn: Conn, bad: Any) -> None:
    windows(conn, 70)
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return bad(request) if state["n"] == 1 else ok_reply(request)

    v = custom(handler)
    counts = embed.embed_stale(conn, v)
    assert counts == {"embedded": 6, "embed_failed": 64, "embed_stopped": 0}
    assert stale(conn) == list(range(7, 71)) and v.tokens_used in (  # newest first: 70-7 failed
        7,
        8,
    )  # a rejected 200 was still billed: counted
    again = embed.embed_stale(conn, custom(ok_reply))  # next run picks the batch up
    assert again["embedded"] == 64 and stale(conn) == []


def test_unordered_and_unindexed_rows() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        texts = json.loads(r.content)["input"]
        vs = FakeEmbedder().embed_documents(texts)
        data = [{"embedding": v, "index": i} for i, v in enumerate(vs)][::-1]
        return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 1}})

    assert custom(handler).embed_documents(["a", "b c", "d"]) == FakeEmbedder().embed_documents(
        ["a", "b c", "d"]
    )


# --- --max-windows -------------------------------------------------------------------------


def cli_env(mp: pytest.MonkeyPatch, db_url: str, api: Api | None = None) -> CliRunner:
    if api is not None:
        with_transport(mp, api)
    mp.setenv("DATABASE_URL", db_url)
    mp.setenv("EMBED_PROVIDER", "voyage")
    mp.setenv("VOYAGE_API_KEY", "k")
    return CliRunner()


@pytest.mark.parametrize(
    ("n", "stale_n", "sizes"),
    [
        (1, 10, [1]),
        (10, 10, [10]),
        (11, 10, [10]),
        (1000, 10, [10]),
        (65, 130, [64, 1]),
        (128, 130, [64, 64]),
    ],
)
def test_max_windows_sizes(
    conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch, n: int, stale_n: int, sizes: list[int]
) -> None:
    windows(conn, stale_n)
    api = Api()
    out = cli_env(monkeypatch, db_url, api).invoke(app, ["embed", "--max-windows", str(n)])
    done = sum(sizes)
    assert out.exit_code == 0, out.stderr
    assert out.stdout.strip() == f"embedded={done} embed_failed=0 tokens={5 * done}"
    assert [len(b["input"]) for _, b in api.requests] == sizes
    assert len(stale(conn)) == stale_n - done


@pytest.mark.parametrize("arg", ["0", "-1", "-5", "abc", "1.5", ""])
def test_max_windows_bad_values_refuse_before_any_work(
    conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch, arg: str
) -> None:
    windows(conn, 4)
    api = Api()
    cli = cli_env(monkeypatch, db_url, api)
    for extra in ([], ["--reindex"]):
        out = cli.invoke(app, ["embed", "--max-windows", arg, *extra])
        assert out.exit_code in (1, 2), (arg, out.exit_code)
        assert api.requests == []
    assert len(stale(conn)) == 4
    n_before = conn.execute("SELECT count(*) FROM window_embedding").fetchone()
    assert n_before == (0,)


def test_max_windows_with_reindex(conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    td_fake = FakeEmbedder()
    windows(conn, 10)
    embed.embed_stale(conn, td_fake)  # 10 embedded with the fake provider
    assert stale(conn) == [] and len(embeddings(conn)) == 10
    api = Api()
    cli = cli_env(monkeypatch, db_url, api)
    out = cli.invoke(app, ["embed", "--reindex", "--max-windows", "3"])
    assert out.exit_code == 0, out.stderr
    assert out.stdout.strip() == "dropped=10 embedded=3 embed_failed=0 tokens=15"
    assert {r[0] for r in embeddings(conn).values()} == {"voyage-4"} and len(embeddings(conn)) == 3
    assert stale(conn) == list(range(1, 8))  # newest first: 10, 9, 8 were embedded
    out = cli.invoke(app, ["embed", "--max-windows", "4"])  # a plain run continues, no mismatch
    assert out.exit_code == 0 and out.stdout.strip() == "embedded=4 embed_failed=0 tokens=20"
    out = cli.invoke(app, ["embed"])
    assert out.stdout.strip() == "embedded=3 embed_failed=0 tokens=15" and stale(conn) == []


def test_max_windows_counts_failed_windows(
    conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows(conn, 70)
    api = Api([(500, {})])
    out = cli_env(monkeypatch, db_url, api).invoke(app, ["embed", "--max-windows", "10"])
    assert out.exit_code == 0 and out.stdout.strip() == "embedded=0 embed_failed=10 tokens=0"
    assert len(stale(conn)) == 70  # nothing consumed beyond the 10 attempted


def test_reindex_missing_key_drops_nothing(
    conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    windows(conn, 3)
    embed.embed_stale(conn, FakeEmbedder())
    cli = cli_env(monkeypatch, db_url)
    monkeypatch.delenv("VOYAGE_API_KEY")
    for extra in ([], ["--reindex"], ["--reindex", "--max-windows", "2"]):
        out = cli.invoke(app, ["embed", *extra])
        assert out.exit_code == 1 and "VOYAGE_API_KEY is not set" in out.stderr
    assert len(embeddings(conn)) == 3
    monkeypatch.setenv("VOYAGE_API_KEY", "   ")  # blank counts as missing
    assert cli.invoke(app, ["embed"]).exit_code == 1


def test_embed_stale_direct_zero_and_negative(conn: Conn) -> None:
    windows(conn, 3)
    for n in (0, -4):
        assert embed.embed_stale(conn, FakeEmbedder(), max_windows=n) == {
            "embedded": 0,
            "embed_failed": 0,
            "embed_stopped": 0,
        }
    assert len(stale(conn)) == 3


# --- A10 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["voyage", "openai"])
def test_a10_key_echoed_in_error_bodies_and_exceptions(
    conn: Conn,
    db_url: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    name: str,
) -> None:
    windows(conn, 70)
    key = KEYS[f"{name.upper()}_API_KEY"]
    echo = {"error": {"message": f"Incorrect API key provided: {key}", "request": "Bearer " + key}}
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(
            401 if len(seen) % 2 else 429, headers={"Retry-After": "0", "X-Echo": key}, json=echo
        )

    real = embed.provider
    monkeypatch.setattr(embed, "provider", lambda s: real(s, httpx.MockTransport(handler)))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_PROVIDER", name)
    for k, v in KEYS.items():
        monkeypatch.setenv(k, v)
    caplog.set_level(logging.DEBUG)
    cli = CliRunner()
    out = cli.invoke(app, ["embed"])
    out2 = cli.invoke(app, ["embed", "--reindex", "--max-windows", "2"])
    blob = caplog.text + out.stdout + out.stderr + out2.stdout + out2.stderr
    assert out.exit_code == 0 and out2.exit_code == 0
    assert key not in blob and "Incorrect API key" not in blob and "Bearer" not in blob
    assert seen  # the handler really ran

    exc: ProviderError | None = None
    try:
        real(
            Settings(embed_provider=name, **{f"{name}_api_key": key}), httpx.MockTransport(handler)
        ).embed_documents(["t"])
    except ProviderError as e:
        exc = e
    assert exc is not None
    import traceback

    text = "".join(traceback.format_exception(exc)) + repr(exc) + str(exc)
    assert key not in text and "Incorrect" not in text


def test_a10_repr_and_settings_never_show_keys() -> None:
    s = Settings(embed_provider="voyage", voyage_api_key="SENT-V", openai_api_key="SENT-O")
    v = embed.provider(s)
    assert "SENT" not in repr(s) + repr(v) + repr(vars(v).get("_http")) + str(s)


# --- request shape -------------------------------------------------------------------------


def test_request_shape_details(monkeypatch: pytest.MonkeyPatch) -> None:
    api = Api()
    for cls, url in (
        (VoyageEmbedder, "https://api.voyageai.com/v1/embeddings"),
        (OpenAIEmbedder, "https://api.openai.com/v1/embeddings"),
    ):
        api.requests.clear()
        e = cls("SENTK", "m-x", transport=api.transport)
        e.embed_documents(["a"] * 64)
        e.embed_queries(["q"])
        for req, body in api.requests:
            assert str(req.url) == url and req.url.query == b""
            assert req.headers["content-type"].startswith("application/json")
            assert "SENTK" not in str(req.url) + json.dumps(body)
            assert body["model"] == "m-x" and body["input"] and set(body) >= {"model", "input"}
            assert [h for h in req.headers if h.lower() in ("x-api-key", "api-key")] == []
        bodies = [b for _, b in api.requests]
        if cls is VoyageEmbedder:
            assert [b["input_type"] for b in bodies] == ["document", "query"]
            assert all(b["output_dimension"] == 1024 and "dimensions" not in b for b in bodies)
        else:
            assert all(
                b["dimensions"] == 1024 and "input_type" not in b and "output_dimension" not in b
                for b in bodies
            )
        assert len(bodies[0]["input"]) == 64


def test_embed_model_env_override_for_both() -> None:
    for name, default in (("voyage", "voyage-4"), ("openai", "text-embedding-3-small")):
        s = Settings.from_env({"EMBED_PROVIDER": name, f"{name.upper()}_API_KEY": "k"})
        assert embed.provider(s).model == default
        s = Settings.from_env(
            {"EMBED_PROVIDER": name, f"{name.upper()}_API_KEY": "k", "EMBED_MODEL": "custom-1"}
        )
        assert embed.provider(s).model == "custom-1"
    s = Settings.from_env(
        {"EMBED_PROVIDER": "voyage", "OPENAI_API_KEY": "k"}
    )  # the other provider's key does not help
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        embed.provider(s)
