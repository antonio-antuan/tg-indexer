"""P2.T4: the `tgi mcp --http` bind guard and Host check (SPEC §5.6, §7, A12).

The guard runs on a faked interface list; the one real start listens on loopback only.
"""

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from typer.testing import CliRunner

from tgi import mcp
from tgi.cli import app

TOKEN = "t" * 43
ROOT = Path(__file__).resolve().parent.parent

# `ip -j addr` output, cut down: loopback, a LAN card, Docker's bridge, a user-defined
# Docker network and an interface without an address.
IP_ADDR = json.dumps([
    {"ifname": "lo", "addr_info": [{"family": "inet", "local": "127.0.0.1"},
                                   {"family": "inet6", "local": "::1"}]},
    {"ifname": "eth0", "addr_info": [{"family": "inet", "local": "192.168.1.20"},
                                     {"family": "inet6", "local": "fe80::1c2:3ff:fe04:506"}]},
    {"ifname": "docker0", "addr_info": [{"family": "inet", "local": "172.17.0.1"}]},
    {"ifname": "br-3f2a9c1d7e44", "addr_info": [{"family": "inet", "local": "172.18.0.1"}]},
    {"ifname": "wlan0", "addr_info": []},
])  # fmt: skip
FAKE = mcp.parse_interfaces(IP_ADDR)


def test_parse_interfaces() -> None:
    assert FAKE == [
        ("lo", "127.0.0.1"),
        ("lo", "::1"),
        ("eth0", "192.168.1.20"),
        ("eth0", "fe80::1c2:3ff:fe04:506"),
        ("docker0", "172.17.0.1"),
        ("br-3f2a9c1d7e44", "172.18.0.1"),
    ]


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("::1", True),
        ("localhost", True),
        ("172.17.0.1", True),  # docker0
        ("172.18.0.1", True),  # br-…
        ("192.168.1.20", False),  # eth0: the LAN
        ("fe80::1c2:3ff:fe04:506", False),
        ("0.0.0.0", False),  # noqa: S104 - the guard's input
        ("::", False),
        ("10.9.9.9", False),  # no local interface has it
        ("archive.example", False),
    ],
)
def test_a12_guard_decision(host: str, allowed: bool) -> None:
    refusal = mcp.check_bind(host, FAKE)
    assert (refusal is None) == allowed
    if refusal is not None:
        assert "\n" not in refusal and "--allow-any-bind" in refusal
    assert mcp.check_bind(host, FAKE, allow_any=True) is None


def test_a_docker_address_without_its_interface_is_refused() -> None:
    """Docker not up yet, or `ip` missing ([]): only loopback passes."""
    assert mcp.check_bind("172.17.0.1", [("eth0", "172.17.0.1")]) is not None
    assert mcp.check_bind("172.17.0.1", []) is not None
    assert mcp.check_bind("127.0.0.1", []) is None


def test_read_interfaces_without_ip_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert mcp.read_interfaces() == []


# --- the CLI ------------------------------------------------------------------------------


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """`tgi mcp --http` up to `uvicorn.run`, which records its arguments instead."""
    import uvicorn

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append({"app": app, **kw}))
    monkeypatch.setattr(mcp, "read_interfaces", lambda: FAKE)
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused@127.0.0.1:1/unused")
    monkeypatch.setenv("MCP_TOKEN", TOKEN)
    monkeypatch.delenv("MCP_ALLOWED_HOSTS", raising=False)
    return calls


@pytest.mark.parametrize("bind", ["0.0.0.0:8765", "[::]:8765", "192.168.1.20:8765"])
def test_a12_cli_refuses_a_wide_bind_without_allow_any_bind(
    bind: str, started: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_BIND", bind)
    result = CliRunner().invoke(app, ["mcp", "--http"])
    assert result.exit_code == 1
    assert "refusing to listen on" in result.output and "--allow-any-bind" in result.output
    assert started == []

    result = CliRunner().invoke(app, ["mcp", "--http", "--allow-any-bind"])
    assert result.exit_code == 0, result.output
    assert "--allow-any-bind: listening on" in result.output
    (call,) = started
    assert (call["host"], call["port"]) == (bind.rpartition(":")[0].strip("[]"), 8765)


@pytest.mark.parametrize(
    ("bind", "host"), [("127.0.0.1:8765", "127.0.0.1"), ("172.17.0.1:8765", "172.17.0.1")]
)
def test_cli_starts_on_loopback_and_on_a_docker_network(
    bind: str, host: str, started: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MCP_BIND", bind)
    result = CliRunner().invoke(app, ["mcp", "--http"])
    assert result.exit_code == 0, result.output
    assert "--allow-any-bind" not in result.output
    (call,) = started
    assert (call["host"], call["port"]) == (host, 8765)
    assert isinstance(call["app"], mcp.BearerAuth)


def test_allow_any_bind_goes_with_http_only(started: list[dict[str, Any]]) -> None:
    result = CliRunner().invoke(app, ["mcp", "--stdio", "--allow-any-bind"])
    assert result.exit_code == 1
    assert "--allow-any-bind goes with --http" in result.output


# --- the Host check -----------------------------------------------------------------------

LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def statuses(security: Any, requests: list[dict[str, str]]) -> list[int]:
    """The status of a tools/list POST with the token and each set of extra headers."""

    async def go() -> list[int]:
        server = mcp.build_server("postgresql://unused@127.0.0.1:1/unused")
        asgi = mcp.http_app(server, TOKEN, security)
        headers = {
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json, text/event-stream",
        }
        async with (
            server.session_manager.run(),
            httpx2.AsyncClient(transport=httpx2.ASGITransport(app=asgi), headers=headers) as h,
        ):
            out = []
            for extra in requests:
                url = f"http://{extra.pop('host')}/mcp"
                out.append((await h.post(url, json=LIST, headers=extra)).status_code)
            return out

    return anyio.run(go)


def test_a_docker_network_bind_keeps_dns_rebinding_protection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The SDK would leave protection off for 172.17.0.1; the Host must name this server."""
    security = mcp.transport_security("172.17.0.1", 8765, FAKE)
    assert security.enable_dns_rebinding_protection
    assert statuses(security, [
        {"host": "172.17.0.1:8765"},
        {"host": "host.docker.internal:8765"},  # the name Nano's container uses (§5.6)
        {"host": "172.17.0.1:8765", "Origin": "http://172.17.0.1:8765"},
        {"host": "rebound.example:8765"},
        {"host": "172.17.0.1:9999"},
        {"host": "127.0.0.1:8765"},
        {"host": "172.17.0.1:8765", "Origin": "http://rebound.example"},
    ]) == [200, 200, 200, 421, 421, 421, 403]  # fmt: skip
    assert "Invalid Host header" in caplog.text and "Invalid Origin header" in caplog.text
    assert "rebound.example" not in caplog.text  # the caller's header value is not logged


def test_loopback_and_any_bind_and_extra_hosts() -> None:
    loopback = mcp.transport_security("127.0.0.1", 8765, FAKE)
    assert statuses(loopback, [
        {"host": "127.0.0.1:8765"},
        {"host": "localhost:8765"},
        {"host": "host.docker.internal:8765"},
    ]) == [200, 200, 421]  # fmt: skip

    wide = mcp.transport_security("0.0.0.0", 8765, FAKE)  # noqa: S104 - --allow-any-bind
    assert statuses(wide, [
        {"host": "192.168.1.20:8765"},
        {"host": "[::1]:8765"},
        {"host": "172.18.0.1:8765"},
        {"host": "host.docker.internal:8765"},
        {"host": "0.0.0.0:8765"},  # noqa: S104 - not an address a client connects to
        {"host": "rebound.example:8765"},
    ]) == [200, 200, 200, 200, 421, 421]  # fmt: skip

    # Without `ip` no interface is known: an address needs MCP_ALLOWED_HOSTS (README).
    blind = mcp.transport_security("0.0.0.0", 8765, [])  # noqa: S104 - --allow-any-bind
    named = mcp.transport_security("0.0.0.0", 8765, [], ("172.18.0.1",))  # noqa: S104
    for security, expected in ((blind, [421, 200]), (named, [200, 200])):
        gateway = [{"host": "172.18.0.1:8765"}, {"host": "localhost:8765"}]
        assert statuses(security, gateway) == expected

    extra = mcp.transport_security("172.18.0.1", 8765, FAKE, ("tgi-host",))
    assert statuses(extra, [{"host": "tgi-host:8765"}, {"host": "other:8765"}]) == [200, 421]


def test_an_extra_host_behind_an_https_proxy() -> None:
    """`tailscale serve` forwards `Host: <node>.ts.net` without a port to a loopback bind."""
    proxied = mcp.transport_security("127.0.0.1", 8765, FAKE, ("tgi.example.ts.net",))
    assert statuses(proxied, [
        {"host": "tgi.example.ts.net"},
        {"host": "tgi.example.ts.net:8765"},
        {"host": "tgi.example.ts.net:9999"},
        {"host": "other.example.ts.net"},
        {"host": "localhost"},  # bind names stay pinned to the port
        {"host": "127.0.0.1"},
    ]) == [200, 200, 421, 421, 421, 421]  # fmt: skip
    # no bare entry for a name that is not a plain host name
    odd = mcp.transport_security("127.0.0.1", 8765, FAKE, ("*", "a:b", "http://x.example"))
    assert not {"*", "a:b", "http://x.example"} & set(odd.allowed_hosts)


# --- the real start, on loopback ----------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def listening(port: int) -> set[str]:
    """Addresses with a listening TCP socket on `port`: from `ss -ltn` when it is
    installed, else from /proc/net/tcp{,6}, which is what `ss` reports."""
    if ss := shutil.which("ss"):
        out = subprocess.run([ss, "-Hltn"], capture_output=True, text=True, check=True).stdout  # noqa: S603
        found = {line.split()[3].rpartition(":") for line in out.splitlines()}
        return {addr.strip("[]") for addr, _, p in found if p == str(port)}
    found = set()
    for name, size in (("tcp", 4), ("tcp6", 16)):
        for line in Path(f"/proc/net/{name}").read_text().splitlines()[1:]:
            local, state = line.split()[1], line.split()[3]
            addr, _, p = local.partition(":")
            if state == "0A" and int(p, 16) == port:  # 0A: LISTEN
                raw = b"".join(bytes.fromhex(addr[i : i + 8])[::-1] for i in range(0, 2 * size, 8))
                family = socket.AF_INET if size == 4 else socket.AF_INET6
                found.add(socket.inet_ntop(family, raw))
    return found


@pytest.fixture
def server(db_url: str, tmp_path: Path) -> Iterator[tuple[int, subprocess.Popen[str]]]:
    port = free_port()
    env = {**os.environ, "DATABASE_URL": db_url, "MCP_TOKEN": TOKEN,
           "MCP_BIND": f"127.0.0.1:{port}"}  # fmt: skip
    proc = subprocess.Popen(  # noqa: S603 - our own entry point
        [str(Path(sys.executable).parent / "tgi"), "mcp", "--http"],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not listening(port):
            assert proc.poll() is None and time.monotonic() < deadline, "the server did not start"
            time.sleep(0.1)
        yield port, proc
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_a12_the_real_server_listens_on_the_loopback_bind_only(
    server: tuple[int, subprocess.Popen[str]],
) -> None:
    port, proc = server
    assert listening(port) == {"127.0.0.1"}
    url = f"http://127.0.0.1:{port}/mcp"
    headers = {"Accept": "application/json, text/event-stream"}
    with httpx2.Client(trust_env=False) as h:  # never through a proxy
        no_token = h.post(url, json=LIST, headers=headers)
        listed = h.post(url, json=LIST, headers={**headers, "Authorization": f"Bearer {TOKEN}"})
    assert (no_token.status_code, no_token.content) == (401, b"")
    assert listed.status_code == 200
    assert {t["name"] for t in listed.json()["result"]["tools"]} == {
        "list_chats", "list_tags", "search", "recent", "status", "get_thread", "get_context",
        "suggest_tags", "add_chat",
    }  # fmt: skip
    proc.terminate()
    output, _ = proc.communicate(timeout=10)
    assert f"serving http://127.0.0.1:{port}/mcp" in output
    assert TOKEN not in output


def test_a12_compose_publishes_postgres_on_loopback_only() -> None:
    """A10's static check of docker-compose.yml, re-run for A12 (no docker here)."""
    from test_repo_config import test_compose_publishes_postgres_on_loopback_only

    test_compose_publishes_postgres_on_loopback_only()
