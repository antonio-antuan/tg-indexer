# ruff: noqa: E501, F811, F401, E402, E731, S101, S105, S106, S603, S607, PLR2004, SIM117, E741
# mypy: disable-error-code="unused-ignore,index,arg-type,misc,type-arg,no-untyped-def,no-untyped-call,union-attr,attr-defined,return-value"
"""QA P2.T3: MCP server probes. Loopback only; the fake embedder; MockTransport-free (no provider is called)."""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from test_mcp import (  # noqa: F401
    NOW,
    TOKEN,
    A,
    B,
    Conn,
    archive,
    mid,
    outcome,
    over_http,
    over_stdio,
)
from typer.testing import CliRunner

from tgi.cli import app  # noqa: F401

SECRET_Q = "zqxsecretquery"
SECRET_TAG = "zqxsecrettag"
LONG = "L" * 43 + "".join(chr(65 + i % 26) for i in range(43))
TGI = str(Path(sys.executable).parent / "tgi")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live(
    archive: None, db_url: str, tmp_path: Path
) -> Iterator[tuple[str, subprocess.Popen[bytes], Path]]:
    port = free_port()
    err = tmp_path / "server.err"
    env = {
        **os.environ,
        "DATABASE_URL": db_url,
        "MCP_TOKEN": LONG,
        "MCP_BIND": f"127.0.0.1:{port}",
        "HTTPS_PROXY": "",
        "HTTP_PROXY": "",
        "NO_PROXY": "127.0.0.1",
    }
    with err.open("wb") as f:
        p = subprocess.Popen([TGI, "mcp", "--http"], env=env, stdout=f, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.1)
        else:
            p.kill()
            pytest.fail("server did not start: " + err.read_text())
        yield base, p, err
        p.terminate()
        p.wait(10)


INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "qa", "version": "0"},
    },
}
HDRS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def raw(base: str, method: str, path: str, headers: list[str], body: str = "") -> tuple[int, bytes]:
    host, port = base.split("//")[1].split(":")
    req = f"{method} {path} HTTP/1.1\r\nHost: {host}:{port}\r\n" + "".join(
        h + "\r\n" for h in headers
    )
    req += f"Content-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}"
    for attempt in range(5):
        try:
            with socket.create_connection((host, int(port)), timeout=5) as sock:
                sock.sendall(req.encode())
                data = b""
                while chunk := sock.recv(65536):
                    data += chunk
            break
        except ConnectionResetError:  # the server answered and closed with the body unread: the
            if attempt == 4:  # RST can discard its answer, so ask again
                raise
    head, _, rest = data.partition(b"\r\n\r\n")
    return int(head.split()[1]), rest


def test_live_http_auth_matrix(live: Any) -> None:
    base, _, _ = live
    px = httpx.Client(trust_env=False, timeout=5)
    bad: list[dict[str, str]] = [
        {}, {"Authorization": ""}, {"Authorization": "Bearer"}, {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {LONG[:-1]}"}, {"Authorization": f"Bearer {LONG}x"},         {"Authorization": f"Bearer  {LONG}"}, {"Authorization": f"bearer {LONG}"}, {"Authorization": f"BEARER {LONG}"},
        {"Authorization": f"Basic {LONG}"}, {"Authorization": LONG}, {"Authorization": f"Bearer {LONG.lower()}"},
        {"Authorization": f"Bearer {LONG[:43]}"}, {"X-Authorization": f"Bearer {LONG}"}, {"Proxy-Authorization": f"Bearer {LONG}"},
        {"Authorization": "Bearer " + "A" * 100000},
    ]  # fmt: skip
    for h in bad:
        for method in ("POST", "GET", "DELETE", "OPTIONS"):
            status, body = raw(
                base, method, "/mcp", [f"{k}: {v}" for k, v in h.items()], json.dumps(INIT)
            )
            assert (status, body) == (401, b""), (method, list(h), status)
    # the token in the URL does not authenticate
    for url in (
        f"{base}/mcp?token={LONG}",
        f"{base}/mcp?authorization=Bearer%20{LONG}",
        f"{base}/{LONG}/mcp",
    ):
        assert px.post(url, headers=HDRS, content=json.dumps(INIT)).status_code == 401
    # 401 for unauthenticated unknown paths too (no path oracle)
    for path in ("/", "/nope", "/mcp/", "/sse", "/.well-known/oauth-protected-resource"):
        assert px.get(base + path).status_code == 401, path
    ok = px.post(
        f"{base}/mcp", headers={**HDRS, "Authorization": f"Bearer {LONG}"}, content=json.dumps(INIT)
    )
    assert ok.status_code == 200 and "tgi" in ok.text
    # raw non-ASCII / duplicate header bytes
    with socket.create_connection(
        (base.split("//")[1].split(":")[0], int(base.rsplit(":", 1)[1])), timeout=5
    ) as s:
        s.sendall(
            f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer wrong\r\nAuthorization: Bearer {LONG}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode()
        )
        first = s.recv(200)
    print("DUPLICATE HEADER ->", first.split(b"\r\n")[0])
    with socket.create_connection(
        (base.split("//")[1].split(":")[0], int(base.rsplit(":", 1)[1])), timeout=5
    ) as s:
        s.sendall(
            b"GET /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer \xff\xfe\r\nConnection: close\r\n\r\n"
        )
        assert s.recv(200).startswith(b"HTTP/1.1 401")


def test_live_http_request_targets_are_not_logged(live: Any) -> None:
    """Issue #29: uvicorn's access log wrote the path and query string of every request."""
    base, proc, err = live
    marker = "zq7target"
    px = httpx.Client(trust_env=False, timeout=5)
    auth = {**HDRS, "Authorization": f"Bearer {LONG}"}
    for headers in (HDRS, auth):
        for url in (f"{base}/mcp?token={marker}", f"{base}/{marker}/mcp?q={marker}"):
            assert px.post(url, headers=headers, content=json.dumps(INIT)).status_code in (
                200, 401, 404,
            )  # fmt: skip
    assert px.post(f"{base}/mcp?q={marker}", headers=auth, content=json.dumps(INIT)).is_success
    proc.terminate()
    proc.wait(10)
    log = err.read_text()
    assert "rejected an unauthorized request" in log  # the log was written
    assert marker not in log and LONG not in log


def test_live_http_calls_log_no_secret(live: Any) -> None:
    base, proc, err = live
    seen: list[Any] = []

    async def go() -> None:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {LONG}"}, trust_env=False
        ) as h:
            async with Client(streamable_http_client(f"{base}/mcp", http_client=h)) as c:
                calls = [
                    ("search", {"queries": [SECRET_Q]}),
                    ("search", {"queries": [SECRET_Q], "tags": [SECRET_TAG]}),
                    ("search", {"queries": [SECRET_Q], "chat_ids": [424242]}),
                    (
                        "search",
                        {"queries": [SECRET_Q], "since": "2026-02-01", "until": "2026-01-01"},
                    ),
                    ("search", {"queries": [SECRET_Q], "limit": 0}),
                    ("search", {"queries": [" "]}),
                    ("search", {"queries": SECRET_Q}),
                    ("search", {"queries": [SECRET_Q], "mode": SECRET_Q}),
                    ("list_chats", {"tag": SECRET_TAG}),
                    ("get_thread", {"chat_id": 424242, "message_id": 1}),
                    ("get_context", {"chat_id": A, "message_id": SECRET_Q}),
                    (
                        "suggest_tags",
                        {"suggestions": [{"chat_id": A, "tag": SECRET_TAG, "reason": SECRET_Q}]},
                    ),
                    (
                        "suggest_tags",
                        {"suggestions": [{"chat_id": 424242, "tag": SECRET_TAG + "2"}]},
                    ),
                    ("nonexistent_tool", {"x": SECRET_Q}),
                ]
                for name, args in calls:
                    try:
                        seen.append((name, outcome(await c.call_tool(name, args))))
                    except Exception as exc:  # noqa: BLE001
                        seen.append((name, f"EXC {type(exc).__name__}"))

    anyio.run(go)
    time.sleep(0.5)
    log = err.read_text()
    print("LOG:", log[:3000])
    for n, o in seen:
        print(n, str(o)[:200])
    leaks = [w for w in (LONG, SECRET_Q, SECRET_TAG) if w in log]
    assert not leaks, f"logged: {leaks}"


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "localhost", "172.17.0.1", "host.docker.internal", "10.1.2.3", "evil.example"],
)
def test_live_http_host_header(live: Any, host: str) -> None:
    """The deployed client reaches the server by the docker gateway address or host.docker.internal (SPEC §5.6)."""
    base, _, _ = live
    port = base.rsplit(":", 1)[1]
    status, body = (
        raw(
            base.replace("127.0.0.1", "127.0.0.1"),
            "POST",
            "/mcp",
            [f"Authorization: Bearer {LONG}", "X-Dummy: 1"],
            json.dumps(INIT),
        )
        if False
        else (0, b"")
    )
    h, p = base.split("//")[1].split(":")
    req = f"POST /mcp HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer {LONG}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {len(json.dumps(INIT))}\r\nConnection: close\r\n\r\n{json.dumps(INIT)}"
    with socket.create_connection((h, int(p)), timeout=5) as sock:
        sock.sendall(req.encode())
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    print("HOSTHDR", host, data.split(b"\r\n")[0], data.partition(b"\r\n\r\n")[2][:60])


def test_live_non_127_0_0_1_loopback_bind_rejects_a_gateway_host_header(
    archive: None, db_url: str, tmp_path: Path
) -> None:
    """P2.T4: every bind gets a Host check. A loopback bind (here 127.0.0.2) accepts its own
    address and localhost but not a gateway name, which no container could reach it by; the
    docker-gateway bind accepting host.docker.internal is tested in test_bind_guard."""
    port = free_port()
    env = {**os.environ, "DATABASE_URL": db_url, "MCP_TOKEN": LONG, "MCP_BIND": f"127.0.0.2:{port}"}
    p = subprocess.Popen(
        [TGI, "mcp", "--http"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.2", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        for host, status in [
            ("host.docker.internal", b"421"), ("172.17.0.1", b"421"),
            ("127.0.0.2", b"200"), ("localhost", b"200"),
        ]:  # fmt: skip
            body = json.dumps(INIT)
            req = f"POST /mcp HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer {LONG}\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}"
            with socket.create_connection(("127.0.0.2", port), timeout=5) as sock:
                sock.sendall(req.encode())
                data = b""
                while chunk := sock.recv(65536):
                    data += chunk
            assert data.startswith(b"HTTP/1.1 " + status), (host, data[:80])
    finally:
        p.terminate()
        p.wait(10)


# --- CLI start-up ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "args", "needle"),
    [
        ({}, ["--http"], "MCP_TOKEN is not set"),
        ({"MCP_TOKEN": "x" * 42}, ["--http"], "too short"),
        ({"MCP_TOKEN": " " * 50}, ["--http"], ""),
        ({"MCP_TOKEN": LONG, "MCP_BIND": "nope"}, ["--http"], "MCP_BIND"),
        ({"MCP_TOKEN": LONG, "MCP_BIND": "127.0.0.1:99999"}, ["--http"], "MCP_BIND"),
        ({"MCP_TOKEN": LONG, "MCP_BIND": "::1:8765"}, ["--http"], "MCP_BIND"),
        ({}, [], "exactly one"),
        ({}, ["--http", "--stdio"], "exactly one"),
        ({}, ["--stdio", "token"], ""),
    ],
)
def test_cli_start_refusals_and_no_token_echo(
    env: dict[str, str], args: list[str], needle: str, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in ("MCP_TOKEN", "MCP_BIND"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DATABASE_URL", db_url)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    r = CliRunner().invoke(app, ["mcp", *args])
    out = r.stdout + r.stderr
    assert r.exit_code != 0, out
    assert needle in out
    assert (env.get("MCP_TOKEN") or "<none>").strip() not in out or not env.get(
        "MCP_TOKEN", ""
    ).strip()


def test_token_command_unique_and_long() -> None:
    toks = {CliRunner().invoke(app, ["mcp", "token"]).stdout.strip() for _ in range(20)}
    assert len(toks) == 20 and all(len(t) >= 43 for t in toks)


def test_stdio_stderr_has_no_secrets_and_stdout_is_protocol_only(
    archive: None, db_url: str, tmp_path: Path
) -> None:
    from mcp import StdioServerParameters

    err = tmp_path / "stdio.err"
    params = StdioServerParameters(
        command=TGI,
        args=["mcp", "--stdio"],
        env={"DATABASE_URL": db_url, "MCP_TOKEN": LONG, "EMBED_PROVIDER": "fake"},
        cwd=tmp_path,
    )

    async def go() -> list[Any]:
        from mcp.client.stdio import stdio_client

        out = []
        with err.open("w") as ef:
            async with stdio_client(params, errlog=ef) as (r, w):
                from mcp import ClientSession

                async with ClientSession(r, w) as s:
                    await s.initialize()
                    calls: list[tuple[str, dict[str, Any]]] = [
                        ("search", {"queries": [SECRET_Q], "tags": [SECRET_TAG]}),
                        ("list_chats", {"tag": SECRET_TAG}),
                        ("suggest_tags", {"suggestions": [{"chat_id": 424242, "tag": SECRET_TAG}]}),
                    ]
                    for name, args in calls:
                        out.append(outcome(await s.call_tool(name, args)))
        return out

    res = anyio.run(go)
    assert all(isinstance(x, dict) and "error" in x for x in res)
    log = err.read_text()
    print("STDIO LOG:", [l for l in log.splitlines() if SECRET_TAG in l or SECRET_Q in l])
    assert SECRET_Q not in log and LONG not in log
    assert SECRET_TAG not in log, "tag text in stdio stderr log"


# --- tool behaviour (in process) ---------------------------------------------------------------
from test_mcp import in_process  # noqa: E402


def call(db_url: str, name: str, args: dict[str, Any], **kw: Any) -> Any:
    return in_process(db_url, [(name, args)], **kw)[0]


def msg_ids(v: Any) -> list[int]:
    return [m["message_id"] >> 20 for m in v]


def test_thread_cycles_orphans_and_tombstones(archive: None, conn: Conn, db_url: str) -> None:
    # cycle 1 <-> 2 in chat B, orphan reply 5 -> 99, tombstoned root
    conn.execute("DELETE FROM message WHERE chat_id = %s", (B,))
    for i, (reply, dele) in {
        1: (2, False),
        2: (1, False),
        5: (99, False),
        7: (None, True),
        8: (7, False),
        9: (8, False),
    }.items():
        conn.execute(
            "INSERT INTO message (chat_id, message_id, server_message_id, date, text, reply_to_message_id, deleted_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (
                B,
                mid(i),
                i,
                NOW.replace(microsecond=0) - __import__("datetime").timedelta(minutes=100 - i),
                f"m{i}",
                None if reply is None else mid(reply),
                NOW if dele else None,
            ),
        )
    cyc = call(db_url, "get_thread", {"chat_id": B, "message_id": mid(1)})
    assert sorted(msg_ids(cyc)) == [1, 2], cyc  # terminates, each once
    assert msg_ids(call(db_url, "get_thread", {"chat_id": B, "message_id": mid(5)})) == [
        5
    ]  # orphan parent
    t = call(db_url, "get_thread", {"chat_id": B, "message_id": mid(9)})
    assert msg_ids(t) == [8, 9] and 7 not in msg_ids(
        t
    )  # tombstoned root is walked through, not returned
    assert msg_ids(call(db_url, "get_thread", {"chat_id": B, "message_id": mid(7)})) == [
        8,
        9,
    ]  # starting at the tombstone
    assert call(db_url, "get_thread", {"chat_id": B, "message_id": mid(7) + 1}) == []
    # same message_id in another chat does not leak across
    assert call(db_url, "get_thread", {"chat_id": A, "message_id": mid(9)}) == []
    # limit edges
    for lim in (0, 51, -1):
        assert (
            call(db_url, "get_thread", {"chat_id": B, "message_id": mid(9), "limit": lim})["error"][
                "code"
            ]
            == "bad_range"
        )
    assert msg_ids(
        call(db_url, "get_thread", {"chat_id": B, "message_id": mid(9), "limit": 1})
    ) in ([9], [8])


def test_get_thread_in_chat_a_never_returns_the_tombstone(archive: None, db_url: str) -> None:
    for start in (3, 6, 7, 1, 4):
        t = call(db_url, "get_thread", {"chat_id": A, "message_id": mid(start)})
        assert 6 not in msg_ids(t), (start, msg_ids(t))
        assert all("удалят" not in m["text"] for m in t)
    assert msg_ids(call(db_url, "get_thread", {"chat_id": A, "message_id": mid(7)})) == [
        1,
        2,
        3,
        4,
        7,
    ]
    ctx = call(
        db_url, "get_context", {"chat_id": A, "message_id": mid(6), "before": 10, "after": 10}
    )
    assert 6 not in msg_ids(ctx) and all("удалят" not in m["text"] for m in ctx)
    c = call(db_url, "get_context", {"chat_id": A, "message_id": mid(5), "before": 1, "after": 1})
    assert msg_ids(c) == [4, 5, 7]  # 6 skipped, window not shrunk


def test_context_bounds_and_errors(archive: None, db_url: str) -> None:
    for b, a in [(-1, 0), (0, -1), (101, 0), (60, 41), (100, 1)]:
        assert (
            call(
                db_url, "get_context", {"chat_id": A, "message_id": mid(3), "before": b, "after": a}
            )["error"]["code"]
            == "bad_range"
        ), (b, a)
    assert msg_ids(
        call(db_url, "get_context", {"chat_id": A, "message_id": mid(3), "before": 0, "after": 0})
    ) == [3]
    assert call(
        db_url, "get_context", {"chat_id": A, "message_id": mid(3), "before": 100, "after": 0}
    )
    assert (
        call(db_url, "get_context", {"chat_id": 424242, "message_id": 1})["error"]["code"]
        == "unknown_chat"
    )
    assert call(db_url, "get_context", {"chat_id": A, "message_id": 12345}) == []


def test_dates_are_utc_whatever_the_server_zone(
    archive: None, conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = [
        ("list_chats", {}),
        ("get_context", {"chat_id": A, "message_id": mid(3)}),
        ("search", {"queries": ["ВНЖ"], "mode": "keyword"}),
    ]
    ref = in_process(db_url, base)
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    monkeypatch.setenv("PGTZ", "Asia/Kolkata")
    assert in_process(db_url, base) == ref
    flat = json.dumps(ref)
    assert "+00:00" in flat and "+05:30" not in flat and "+09:00" not in flat


def test_since_until_whole_utc_days(archive: None, conn: Conn, db_url: str) -> None:
    from datetime import UTC, datetime

    conn.execute("DELETE FROM message WHERE chat_id = %s", (B,))
    conn.execute('DELETE FROM "window" WHERE chat_id = %s', (B,))
    for i, ts in {
        1: datetime(2026, 3, 10, 23, 59, 59, tzinfo=UTC),
        2: datetime(2026, 3, 11, 0, 0, 0, tzinfo=UTC),
    }.items():
        conn.execute(
            "INSERT INTO message (chat_id,message_id,server_message_id,date,text) VALUES (%s,%s,%s,%s,%s)",
            (B, mid(i), i, ts, f"daybound{i}"),
        )

    def got(**kw: Any) -> set[int]:
        v = call(
            db_url,
            "search",
            {
                "queries": ["daybound1 daybound2", "daybound1", "daybound2"],
                "mode": "keyword",
                "chat_ids": [B],
                **kw,
            },
        )
        return {m["message_id"] >> 20 for h in v for m in h["messages"] if "daybound" in m["text"]}

    assert got(until="2026-03-10") == {1}
    assert got(since="2026-03-11") == {2}
    assert got(since="2026-03-10", until="2026-03-10") == {1}
    assert got(since="2026-03-11", until="2026-03-11") == {2}


def test_suggest_tags_matrix(archive: None, conn: Conn, db_url: str) -> None:
    def rows() -> list[tuple[Any, ...]]:
        return [
            tuple(r)
            for r in conn.execute(
                "SELECT chat_id, tag, status, proposed_by FROM chat_tag ORDER BY 1, 2"
            )
        ]

    before = rows()
    s = lambda *sug: call(db_url, "suggest_tags", {"suggestions": list(sug)})  # noqa: E731
    assert s() == {"proposed": 0}
    assert rows() == before
    # confirmed stays confirmed; proposed by the owner stays the owner's
    assert s({"chat_id": A, "tag": "spain"}, {"chat_id": B, "tag": "docs"}) == {"proposed": 0}
    assert rows() == before
    # extra fields (status, proposed_by) are ignored and cannot make a confirmed row
    r = s(
        {"chat_id": A, "tag": "new", "status": "confirmed", "proposed_by": "owner", "reason": "x"}
    )
    assert r == {"proposed": 1}
    assert (A, "new", "proposed", "mcp") in rows()
    # duplicates in one call; whitespace; case
    assert s(
        {"chat_id": A, "tag": "dup"}, {"chat_id": A, "tag": " dup "}, {"chat_id": A, "tag": "Dup"}
    ) == {"proposed": 2}
    assert (A, "dup", "proposed", "mcp") in rows() and (A, "Dup", "proposed", "mcp") in rows()
    # all or nothing
    n = len(rows())
    for bad in (
        [{"chat_id": A, "tag": "ok1"}, {"chat_id": 424242, "tag": "ok2"}],
        [{"chat_id": A, "tag": "ok1"}, {"chat_id": A, "tag": "  "}],
        [{"chat_id": A, "tag": "ok1"}, {"chat_id": A, "tag": "nul\x00byte"}],
        [{"chat_id": A, "tag": "ok1"}, {"chat_id": A, "tag": "x" * 5_000_000}],
    ):
        out = in_process_raw(db_url, [("suggest_tags", {"suggestions": bad})])[0]
        print("BAD", str(out)[:160])
        assert len(rows()) == n, bad[1]
    # a huge batch
    out = s(*[{"chat_id": A, "tag": f"bulk{i}"} for i in range(2000)])
    assert out == {"proposed": 2000}


def test_embedder_failure_is_an_error_not_a_silent_fallback(archive: None, db_url: str) -> None:
    from tgi.embed import FakeEmbedder
    from tgi.search import EmbedderUnavailable

    class Down(FakeEmbedder):
        def embed_queries(self, texts: Any) -> Any:
            raise EmbedderUnavailable("down")

    for mode in ("hybrid", "semantic"):
        out = call(db_url, "search", {"queries": ["ВНЖ"], "mode": mode}, embedder=Down())
        assert (
            out["error"]["code"] == "embedder_unavailable" and "keyword" in out["error"]["message"]
        ), out
    assert call(db_url, "search", {"queries": ["ВНЖ"], "mode": "keyword"}, embedder=Down())
    # negation-only needs no embedder
    assert isinstance(call(db_url, "search", {"queries": ["-zzz"]}, embedder=Down()), list)


def test_type_errors_have_no_code(archive: None, db_url: str) -> None:
    out = in_process_raw(
        db_url,
        [
            ("search", {"queries": "str"}),
            ("search", {"queries": ["a"], "mode": "fuzzy"}),
            ("get_thread", {"chat_id": "x", "message_id": 1}),
            ("search", {"queries": ["a"], "since": "yesterday"}),
            ("search", {"queries": []}),
            ("search", {"queries": ["a"], "limit": True}),
        ],
    )
    for o in out:
        print("TYPEERR", o[:150].replace("\n", " "))
    codes = []
    for o in out:
        try:
            codes.append(json.loads(o.partition(": ")[2])["code"])
        except Exception:  # noqa: BLE001
            codes.append(None)
    print("CODES", codes)
    assert all(codes), "argument-type errors carry no `code` (SPEC §5.6: errors carry a code)"


def in_process_raw(db_url: str, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    from mcp import Client

    from tgi import mcp as m

    async def go() -> list[str]:
        async with Client(m.build_server(db_url)) as c:
            return [(await c.call_tool(n, a)).content[0].text for n, a in calls]

    return anyio.run(go)


def test_db_down_error_does_not_leak_the_url(archive: None, db_url: str) -> None:
    import re

    bad = (
        re.sub(r"(postgresql://)([^/]*)", r"\1leakuser:leakpass@127.0.0.1:1", db_url)
        if "://" in db_url
        else "postgresql://leakuser:leakpass@127.0.0.1:1/x"
    )
    out = in_process_raw(bad, [("list_tags", {})])[0]
    print("DBDOWN", out[:300])
    assert "leakpass" not in out


def test_list_chats_includes_disabled_and_counts_skip_tombstones(
    archive: None, conn: Conn, db_url: str
) -> None:
    conn.execute("UPDATE chat SET enabled = false WHERE chat_id = %s", (B,))
    v = call(db_url, "list_chats", {})
    by = {c["chat_id"]: c for c in v}
    assert B in by and by[A]["message_count"] == 6  # 7 minus the tombstone
    assert [c["title"] for c in v] == sorted(c["title"] for c in v)
    hits = call(db_url, "search", {"queries": ["renewal TIE"], "mode": "keyword"})
    print("DISABLED chat in search:", B in {h["chat_id"] for h in hits})


def test_debug_log_over_http_has_no_query_or_message_text(
    archive: None, db_url: str, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    caplog.set_level(logging.DEBUG)
    calls: Any = [
        ("search", {"queries": [SECRET_Q, "ВНЖ"]}),
        ("search", {"queries": ["ВНЖ"], "chat_ids": [A], "mode": "keyword"}),
        ("get_context", {"chat_id": A, "message_id": mid(3)}),
        ("get_thread", {"chat_id": A, "message_id": mid(3)}),
        ("suggest_tags", {"suggestions": [{"chat_id": A, "tag": "okay-tag", "reason": SECRET_Q}]}),
    ]
    over_http(db_url, calls)
    owners = {
        r.name
        for r in caplog.records
        if any(
            n in r.getMessage() for n in (SECRET_Q, "Где продлить", "Patraix", "okay-tag", TOKEN)
        )
    }
    print("LOGGERS WITH TEXT", owners)
    text = "\n".join(
        r.getMessage()
        for r in caplog.records
        if r.name.startswith(("tgi", "mcp.server", "uvicorn", "mcp.os", "mcp.shared"))
        or r.name == "mcp"
    )
    for needle in (SECRET_Q, "Где продлить", "Patraix", "gestoría", "paella на", "okay-tag", TOKEN):
        assert needle not in text, needle


def test_live_concurrent_calls_and_stateless_sessions(live: Any) -> None:
    base, _, _ = live

    async def one(i: int) -> Any:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {LONG}"}, trust_env=False
        ) as h:
            async with Client(streamable_http_client(f"{base}/mcp", http_client=h)) as c:
                return outcome(
                    await c.call_tool("search", {"queries": ["ВНЖ"], "mode": "keyword"})
                ), outcome(
                    await c.call_tool(
                        "suggest_tags", {"suggestions": [{"chat_id": A, "tag": f"t{i % 5}"}]}
                    )
                )

    async def many() -> list[Any]:
        out: list[Any] = []

        async def w(i: int) -> None:
            out.append(await one(i))

        async with anyio.create_task_group() as tg:
            for i in range(20):
                tg.start_soon(w, i)
        return out

    res = anyio.run(many)
    assert len(res) == 20 and all(r[0] for r in res)
    assert sum(r[1]["proposed"] for r in res) == 5  # five distinct tags, each written once


def test_unrepresentable_tags_are_coded_errors(archive: None, db_url: str) -> None:
    for tag in ("nul\x00byte", "x" * 5_000_000):
        text = in_process_raw(
            db_url, [("suggest_tags", {"suggestions": [{"chat_id": A, "tag": tag}]})]
        )[0]
        assert json.loads(text.partition(": ")[2])["code"]
