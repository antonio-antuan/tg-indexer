# ruff: noqa: E501, F811, F401, E402, E731, S101, S105, S106, S603, S607, PLR2004, SIM117, E741, S104, UP031
# mypy: disable-error-code="unused-ignore,index,arg-type,misc,type-arg,no-untyped-def,no-untyped-call,union-attr,attr-defined,return-value"
"""QA P2.T4: bind guard, Host/Origin allowlist, the systemd unit. Loopback only; no real provider."""

import configparser
import json
import logging
import os
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from tgi import mcp
from tgi.cli import app

TOKEN = "Q" * 43 + "".join(chr(97 + i % 26) for i in range(43))
TGI = str(Path(sys.executable).parent / "tgi")
ROOT = Path(__file__).resolve().parent.parent
INIT = json.dumps({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "qa", "version": "0"}},
})  # fmt: skip
NOPROXY = {
    k: ""
    for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy")
}


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def ready(host: str, port: int, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + 25
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    while time.monotonic() < deadline:
        assert proc.poll() is None, "server exited"
        with socket.socket(fam) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                return
        time.sleep(0.2)
    raise AssertionError("not listening")


@contextmanager
def live(
    db_url: str,
    tmp_path: Path,
    host: str = "127.0.0.1",
    extra_env: dict[str, str] | None = None,
    level: str | None = None,
) -> Iterator[tuple[int, Path]]:
    port = free_port(host)
    bind = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
    log = tmp_path / f"srv-{port}.log"
    env = {
        **os.environ,
        **NOPROXY,
        "DATABASE_URL": db_url,
        "MCP_TOKEN": TOKEN,
        "MCP_BIND": bind,
        **(extra_env or {}),
    }
    code = "import logging,sys;logging.basicConfig(level=logging.DEBUG);from tgi.cli import app;sys.argv=['tgi','mcp','--http'];app()"
    if level == "nofilter":
        code = "import logging,sys;logging.basicConfig(level=logging.DEBUG);from tgi import mcp;mcp._quiet_header_values=lambda: None;from tgi.cli import app;sys.argv=['tgi','mcp','--http'];app()"
    cmd = [sys.executable, "-c", code] if level else [TGI, "mcp", "--http"]
    with log.open("wb") as fh:
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        try:
            ready(host, port, proc)
            yield port, log
        finally:
            proc.terminate()
            proc.wait(timeout=10)


def raw(
    host: str,
    port: int,
    method: str,
    headers: list[str],
    body: str = INIT,
    auth: bool = True,
    target: str = "/mcp",
    version: str = "1.1",
) -> tuple[int, bytes]:
    hs = list(headers)
    if auth:
        hs.append(f"Authorization: Bearer {TOKEN}")
    req = f"{method} {target} HTTP/{version}\r\n" + "".join(h + "\r\n" for h in hs)
    req += f"Content-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n{body}"
    for attempt in range(5):
        try:
            with socket.create_connection((host, port), timeout=5) as sock:
                sock.sendall(req.encode())
                data = b""
                while chunk := sock.recv(65536):
                    data += chunk
            break
        except ConnectionResetError:
            if attempt == 4:
                raise
    head, _, rest = data.partition(b"\r\n\r\n")
    return (int(head.split()[1]) if head else 0), rest


def post(host, port, hostheader, origin=None, auth=True, method="POST"):
    hs = [f"Host: {hostheader}"] + ([f"Origin: {origin}"] if origin is not None else [])
    return raw(host, port, method, hs, auth=auth)[0]


# --- Host / Origin on a live loopback server ---------------------------------------------------


@pytest.mark.parametrize("bind", ["127.0.0.1", "127.0.0.2", "::1"])
def test_live_host_matrix_on_loopback_binds(db_url: str, tmp_path: Path, bind: str) -> None:
    try:
        free_port(bind)
    except OSError:
        pytest.skip(f"{bind} not available")
    with live(db_url, tmp_path, bind) as (port, _):
        me = f"[{bind}]" if ":" in bind else bind
        ok = [f"{me}:{port}", f"localhost:{port}"]
        assert (
            post(bind, port, f"LOCALHOST:{port}") == 421
        )  # OBSERVATION: Host names are case-insensitive (RFC 9110), the match is exact-case
        bad = [
            f"{me}", f"{me}:{port + 1}", f"{me}:", "host.docker.internal:%d" % port, "172.17.0.1:%d" % port,
            "evil.example:%d" % port, f"localhost.:{port}", f"localhost:{port}.", f"{me}:{port}@evil.example",
            f"evil.example:{port}#{me}:{port}", f"{me}:{port}, evil.example", "", " ", f"{me}:0{port}", f"{me}:{port} x",
            "0.0.0.0:%d" % port, f"[::1]:{port}" if bind != "::1" else f"127.0.0.1:{port}",
            "localhost:%d\x00" % port.__int__(),
        ]  # fmt: skip
        for h in ok:
            assert post(bind, port, h) == 200, h
        for h in bad:
            try:
                got = post(bind, port, h)
            except UnicodeEncodeError:
                continue
            assert got in (421, 400), (h, got)
        # no Host at all (HTTP/1.0)
        assert raw(bind, port, "POST", [], version="1.0")[0] in (400, 421), "no Host"
        # duplicate Host headers: valid then evil, evil then valid
        for pair in (
            [f"Host: {me}:{port}", "Host: evil.example"],
            ["Host: evil.example", f"Host: {me}:{port}"],
        ):
            assert raw(bind, port, "POST", pair)[0] in (400, 421)
        # without the token the answer is 401 whatever the Host (no Host oracle)
        for h in [f"{me}:{port}", "evil.example", ""]:
            assert post(bind, port, h, auth=False) == 401, h
        # every method gets the same Host answer
        for m in ("GET", "DELETE", "OPTIONS"):
            assert post(bind, port, "evil.example", method=m) in (400, 421), m


def test_live_origin_matrix(db_url: str, tmp_path: Path) -> None:
    with live(db_url, tmp_path) as (port, _):
        h = f"127.0.0.1:{port}"
        good = [None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"]
        bad = [
            "null", "https://127.0.0.1:%d" % port, "http://127.0.0.1", "http://127.0.0.1:%d/" % port, "http://127.0.0.1:%d/x" % port,
            "http://evil.example", "http://evil.example:%d" % port, "http://host.docker.internal:%d" % port, "http://127.0.0.1:%d.evil.example" % port,
            "http://127.0.0.1:%d@evil.example" % port, "file://", "HTTP://127.0.0.1:%d" % port, "http://127.0.0.1:%d, http://evil.example" % port,
            "chrome-extension://abc", "http://[::1]:%d" % port,
        ]  # fmt: skip
        for o in good:
            assert post("127.0.0.1", port, h, origin=o) == 200, o
        for o in bad:
            assert post("127.0.0.1", port, h, origin=o) == 403, o
        assert (
            post("127.0.0.1", port, h, origin="") == 200
        )  # OBSERVATION: an empty Origin counts as absent (SDK); browsers never send one
        # an Origin with a bad Host: Host is checked first (421), both reject
        assert post("127.0.0.1", port, "evil.example", origin="http://evil.example") == 421
        # the answer carries no echo of the Origin/Host
        status, body = raw("127.0.0.1", port, "POST", [f"Host: {h}", "Origin: http://evil.example"])
        assert status == 403 and b"evil" not in body
        status, body = raw("127.0.0.1", port, "POST", ["Host: evil.example"])
        assert status == 421 and b"evil" not in body
        # no token: 401 even with a bad Origin
        assert post("127.0.0.1", port, h, origin="http://evil.example", auth=False) == 401


@pytest.mark.parametrize(
    ("allowed", "accepted", "rejected"),
    [
        ("host.docker.internal", ["host.docker.internal"], ["other.example", "172.17.0.1"]),
        ("a.example, b.example ,,", ["a.example", "b.example"], ["c.example", ""]),
        ("172.17.0.1", ["172.17.0.1"], ["172.17.0.2"]),
        ("EVIL.example", ["EVIL.example"], []),
        ("*", [], ["anything.example"]),
        ("evil.example:9999", [], ["evil.example"]),
        ("http://evil.example", [], ["evil.example"]),
        ("[::1]", [], []),
    ],
)
def test_live_allowed_hosts(
    db_url: str, tmp_path: Path, allowed: str, accepted: list[str], rejected: list[str]
) -> None:
    with live(db_url, tmp_path, extra_env={"MCP_ALLOWED_HOSTS": allowed}) as (port, _):
        for name in accepted:
            assert post("127.0.0.1", port, f"{name}:{port}") == 200, name
            assert (
                post("127.0.0.1", port, f"{name}:{port}", origin=f"http://{name}:{port}") == 200
            ), name
            assert post("127.0.0.1", port, f"{name}:{port + 1}") == 421, (
                name
            )  # the port stays pinned
        for name in rejected:
            assert post("127.0.0.1", port, f"{name}:{port}") == 421, name
        assert post("127.0.0.1", port, f"127.0.0.1:{port}") == 200  # the bind address still works
        assert post("127.0.0.1", port, f"zzz.example:{port}") == 421


def test_live_rejected_header_values_never_reach_the_log(db_url: str, tmp_path: Path) -> None:
    marker_host, marker_origin = "qamarkerhost.example", "http://qamarkerorigin.example"
    with live(db_url, tmp_path, level="debug") as (port, log):
        post("127.0.0.1", port, f"{marker_host}:{port}")
        post("127.0.0.1", port, f"127.0.0.1:{port}", origin=marker_origin)
        post("127.0.0.1", port, f"{marker_host}:{port}", origin=marker_origin)
        raw(
            "127.0.0.1",
            port,
            "POST",
            [f"Host: 127.0.0.1:{port}", "Content-Type: text/qamarkerct"],
            auth=True,
        )
        raw(
            "127.0.0.1",
            port,
            "POST",
            [f"Host: 127.0.0.1:{port}", "Mcp-Protocol-Version: qamarkerproto"],
            auth=True,
        )
        raw(
            "127.0.0.1",
            port,
            "POST",
            [f"Host: 127.0.0.1:{port}", "Mcp-Session-Id: qamarkersession"],
            auth=True,
        )
        raw(
            "127.0.0.1",
            port,
            "POST",
            [f"Host: 127.0.0.1:{port}"],
            body='{"qamarkerbody": 1',
            auth=True,
        )
        raw(
            "127.0.0.1",
            port,
            "GET",
            [f"Host: 127.0.0.1:{port}"],
            target="/qamarkerpath?x=qamarkerquery",
        )
        time.sleep(0.5)
    text = log.read_text(errors="replace")
    assert "Invalid Host header" in text and "Invalid Origin header" in text, (
        "rejections are still logged"
    )
    assert "qamarker" not in text, [l for l in text.splitlines() if "qamarker" in l][:5]
    assert TOKEN not in text


def test_the_filter_is_what_hides_the_value(db_url: str, tmp_path: Path) -> None:
    """Control (mutant): with the filter switched off the SDK does log the Host value, so the test above can fail."""
    with live(db_url, tmp_path, level="nofilter") as (port, log):
        post("127.0.0.1", port, f"qamarkerctl.example:{port}")
        time.sleep(0.5)
    assert "qamarkerctl" in log.read_text(errors="replace")


# --- the guard, with the real `ip` lookup and a faked `ip` -----------------------------------


def fake_ip(tmp_path: Path, body: str) -> dict[str, str]:
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / "ip"
    f.write_text("#!/bin/sh\n" + body + "\n")
    f.chmod(0o755)
    return {"PATH": f"{d}:{os.environ['PATH']}"}


def test_read_interfaces_with_a_real_process_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sample = json.dumps(
        [
            {"ifname": "docker0", "addr_info": [{"family": "inet", "local": "172.17.0.1"}]},
            {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}]},
            {"ifname": "x", "addr_info": [{"family": "inet6", "prefixlen": 64}]},
        ]
    )
    cases = {
        "ok": (f"cat <<'EOF'\n{sample}\nEOF", [("docker0", "172.17.0.1"), ("lo", "127.0.0.1")]),
        "exit1": ("echo '[]'; exit 1", []),
        "garbage": ("echo not json", []),
        "empty": ("true", []),
        "no-ifname": ("echo '[{\"addr_info\": []}]'", []),
        "null-addr_info": ('echo \'[{"ifname": "docker0", "addr_info": null}]\'', []),
        "slow": ("sleep 30", []),
    }
    for name, (body, expected) in cases.items():
        monkeypatch.setenv("PATH", fake_ip(tmp_path, body)["PATH"])
        t = time.monotonic()
        assert mcp.read_interfaces() == expected, name
        assert time.monotonic() - t < 15, name


def run_cli(
    env: dict[str, str], args: list[str], tmp_path: Path, timeout: float = 15
) -> subprocess.CompletedProcess:
    full = {
        **os.environ,
        **NOPROXY,
        "DATABASE_URL": "postgresql://unused@127.0.0.1:1/unused",
        **env,
    }
    return subprocess.run(
        [TGI, "mcp", *args], env=full, capture_output=True, text=True, timeout=timeout, cwd=tmp_path
    )


@pytest.mark.parametrize(
    "bind",
    [
        "0.0.0.0:8765",
        "[::]:8765",
        "10.9.9.9:8765",
        "192.168.1.20:8765",
        "archive.example:8765",
        "[fe80::1]:8765",
        "[::ffff:10.1.2.3]:8765",
        "8.8.8.8:8765",
        "0:8765",
        "0.0.0.0.:8765",
        "224.0.0.1:8765",
        "255.255.255.255:8765",
        "169.254.1.1:8765",
    ],
)
def test_cli_refuses_wide_binds_for_real_and_never_echoes_the_token(
    tmp_path: Path, bind: str
) -> None:
    for env in (
        {"MCP_BIND": bind},
        {"MCP_BIND": bind, "PATH": fake_ip(tmp_path, "exit 1")["PATH"]},
    ):
        r = run_cli({**env, "MCP_TOKEN": TOKEN}, ["--http"], tmp_path)
        assert r.returncode != 0 and "Traceback" not in r.stderr + r.stdout
        text = r.stdout + r.stderr
        assert TOKEN not in text
        assert "refusing" in text or "MCP_BIND must be" in text, text


def test_cli_flag_combinations(tmp_path: Path) -> None:
    base = {"MCP_TOKEN": TOKEN, "MCP_BIND": "127.0.0.1:%d" % free_port()}
    for args in (
        ["--allow-any-bind"],
        ["--stdio", "--allow-any-bind"],
        ["--http", "--stdio", "--allow-any-bind"],
        ["--allow-any-bind", "token"],
        ["token", "--allow-any-bind"],
    ):
        r = run_cli(base, args, tmp_path)
        assert r.returncode != 0 and "Traceback" not in r.stderr, args


# --- host/origin security settings, as data ---------------------------------------------------

FAKE = [
    ("lo", "127.0.0.1"),
    ("eth0", "192.168.1.20"),
    ("docker0", "172.17.0.1"),
    ("br-abc", "172.18.0.1"),
    ("br-v6", "fd00::1"),
]


def test_transport_security_settings() -> None:
    def hs(*a):
        s = mcp.transport_security(*a)
        assert s.enable_dns_rebinding_protection
        return s.allowed_hosts, s.allowed_origins

    h, o = hs("127.0.0.1", 8765, FAKE)
    assert h == ["127.0.0.1:8765", "localhost:8765"] and o == [
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ]
    h, _ = hs("::1", 8765, FAKE)
    assert h == ["[::1]:8765", "localhost:8765"]
    h, _ = hs("172.17.0.1", 8765, FAKE)
    assert h == ["172.17.0.1:8765", "host.docker.internal:8765"]
    h, _ = hs("fd00::1", 8765, FAKE)
    assert h == ["[fd00::1]:8765", "host.docker.internal:8765"]
    h, _ = hs("0.0.0.0", 8765, FAKE)
    assert (
        "192.168.1.20:8765" in h
        and "host.docker.internal:8765" in h
        and "0.0.0.0:8765" not in h
        and "[fd00::1]:8765" in h
    )
    h, _ = hs("::", 8765, [])
    assert h == ["localhost:8765", "host.docker.internal:8765"]
    # a wildcard or a stray scheme in MCP_ALLOWED_HOSTS never becomes a pattern
    h, _ = hs("127.0.0.1", 8765, FAKE, ("*", "a:b", "x.example"))
    assert not any("*" in x.split(":")[0] and x != "*:8765" for x in h)
    assert "x.example:8765" in h


def test_settings_parses_allowed_hosts() -> None:
    from tgi.config import Settings

    base = {"DATABASE_URL": "postgresql://u@h/d"}
    assert Settings.from_env({**base}).mcp_allowed_hosts == ()
    assert Settings.from_env({**base, "MCP_ALLOWED_HOSTS": " a , ,b,, "}).mcp_allowed_hosts == (
        "a",
        "b",
    )
    assert Settings.from_env({**base, "MCP_ALLOWED_HOSTS": ""}).mcp_allowed_hosts == ()


# --- the unit ---------------------------------------------------------------------------------


def test_the_systemd_unit() -> None:
    unit = ROOT / "deploy" / "tgi-mcp.service"
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read(unit)
    assert set(cp.sections()) == {"Unit", "Service", "Install"}
    s = cp["Service"]
    assert (
        s["ExecStart"].endswith("/.venv/bin/tgi mcp --http")
        and "--allow-any-bind" not in s["ExecStart"]
    )
    assert (
        s["Restart"] == "on-failure"
        and s["EnvironmentFile"].startswith("%h/")
        and not s["EnvironmentFile"].startswith("-")
    )
    assert cp["Install"]["WantedBy"] == "default.target"
    assert "tg-indexer" in s["WorkingDirectory"]
    # the secrets are in the EnvironmentFile, never in the unit
    text = unit.read_text()
    for needle in ("MCP_TOKEN=", "API_KEY", "PASSWORD", "TDLIB"):
        assert needle not in text, needle
    assert stat.S_IMODE(unit.stat().st_mode) & 0o022 == 0


def test_systemd_analyze_verify_accepts_the_unit(tmp_path: Path) -> None:
    """System-mode `verify` parses the unit without a user manager; `--user` needs a session bus."""
    import shutil

    if not shutil.which("systemd-analyze"):
        pytest.skip("no systemd-analyze")
    unit = tmp_path / "tgi-mcp.service"
    unit.write_text((ROOT / "deploy" / "tgi-mcp.service").read_text())
    r = subprocess.run(
        ["systemd-analyze", "verify", str(unit)], capture_output=True, text=True, timeout=30
    )
    problems = [
        l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "is not executable" not in l
    ]  # the venv is not here
    assert problems == [], problems
