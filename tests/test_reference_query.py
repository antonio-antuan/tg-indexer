"""P2.T5: the reference query of SPEC §5.7 through `tgi selftest` (A9).

`--scenario` loads `tests/scenarios/reference.json` into a throwaway database, serves it
on loopback and runs a real MCP client over HTTP; without it, `tgi selftest` queries the
server at MCP_BIND. The embedder is the fake one throughout.
"""

import json
import logging
import re
import socket
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from test_bind_guard import TOKEN, server  # noqa: F401 - the `server` fixture
from typer.testing import CliRunner

from tgi import selftest
from tgi.cli import app
from tgi.config import Settings

Conn = psycopg.Connection[tuple[object, ...]]
REFERENCE = Path(__file__).parent / "scenarios" / "reference.json"
SCENARIO = json.loads(REFERENCE.read_text())
TEXTS = [m["text"] for c in SCENARIO["chats"] for m in c["messages"]]
URL = re.compile(r"https?://127\.0\.0\.1:\d+")  # the MCP server selftest serves and queries


def selftest_databases(url: str) -> list[str]:
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT datname FROM pg_database WHERE datname LIKE %s", (selftest.PREFIX + "%",)
        )
        return [r[0] for r in rows]


def invoke(db_url: str, monkeypatch: pytest.MonkeyPatch, *args: str) -> Any:
    monkeypatch.setenv("DATABASE_URL", db_url)
    return CliRunner().invoke(app, ["selftest", *args])


@pytest.fixture
def no_leftovers(db_url: str) -> Iterator[None]:
    """No selftest database may outlive the test, whatever its outcome."""
    before = selftest_databases(db_url)
    yield
    assert selftest_databases(db_url) == before


def test_a9_the_reference_scenario_passes(
    db_url: str,
    conn: Conn,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
    no_leftovers: None,
) -> None:
    caplog.set_level(logging.DEBUG)
    result = invoke(db_url, monkeypatch, "--scenario", str(REFERENCE))
    assert result.exit_code == 0, result.output
    # four seeded windows and the three unrelated ones (ranked last); context of the top hit
    assert result.stdout == "selftest ok: tags=2 hits=7 seeded=4 unrelated_returned=3 context=4\n"
    assert "search mode=hybrid queries=4" in caplog.text  # the logs were captured
    assert any(r.name.startswith("mcp.server") for r in caplog.records)
    # The SDK's client logs what it sends at DEBUG; selftest keeps it at INFO.
    assert not any(r.name.startswith("mcp.client") and r.levelno < logging.INFO
                   for r in caplog.records)  # fmt: skip
    # The HTTP client logs each request's URL at INFO; selftest keeps it at WARNING.
    assert not any(r.name.startswith(("httpx", "httpcore")) and r.levelno < logging.WARNING
                   for r in caplog.records)  # fmt: skip
    for text in (result.output, result.stderr, caplog.text, *capfd.readouterr()):
        assert [t for t in TEXTS if t in text] == []  # A10: counts and ids only
        assert [q for q in selftest.QUERIES if q in text] == []
        assert URL.findall(text) == []
    # The configured database is never written to.
    assert conn.execute("SELECT count(*) FROM chat").fetchone() == (0,)


def moved(kind: str, link: str) -> dict[str, Any]:
    """The scenario with `link` moved from its expectation list into `kind`."""
    scenario = json.loads(json.dumps(SCENARIO))
    expect = scenario["reference"]
    for key in ("seeded", "excluded", "unrelated"):
        if link in expect[key]:
            expect[key].remove(link)
    expect[kind].append(link)
    return scenario


@pytest.mark.parametrize(
    ("scenario", "error"),
    [
        (moved("excluded", "https://t.me/rus_valencia/2"), "1 excluded messages returned"),
        (moved("unrelated", "https://t.me/c/1700000003/1"), "1 unrelated messages rank with"),
        # an unrelated window is returned (last), so it cannot be expected absent
        (moved("excluded", "https://t.me/avisos_vlc/2"), "1 excluded messages returned"),
    ],
)
def test_a9_a_failed_expectation_exits_1(
    db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenario: dict[str, Any],
    error: str,
    no_leftovers: None,
) -> None:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    result = invoke(db_url, monkeypatch, "--scenario", str(path))
    assert result.exit_code == 1
    assert error in result.stderr


def test_a9_too_few_seeded_hits_fail(
    db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_leftovers: None
) -> None:
    """With two seeded messages moved out of the last 30 days only two can be found."""
    scenario = json.loads(json.dumps(SCENARIO))
    for m in scenario["chats"][0]["messages"]:
        if m["id"] in (2, 4):
            m["date"] = "2026-07-01T10:00:00+00:00"
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    result = invoke(db_url, monkeypatch, "--scenario", str(path))
    assert result.exit_code == 1
    assert "2 of 4 seeded messages found" in result.stderr


def test_a_failure_while_loading_still_drops_the_database(
    db_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_leftovers: None
) -> None:
    scenario = json.loads(json.dumps(SCENARIO))
    scenario["reference"]["tags"]["valencia"].append(-1009999999999)  # not in the scenario
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario))
    result = invoke(db_url, monkeypatch, "--scenario", str(path))
    assert result.exit_code == 1
    assert (
        result.stderr
        == "selftest failed: the scenario tags chat -1009999999999, which it does not have\n"
    )


def test_an_interruption_right_after_create_still_drops_the_database(
    db_url: str, monkeypatch: pytest.MonkeyPatch, no_leftovers: None
) -> None:
    """SIGTERM raises SystemExit (see `_sigterm_exits`); here it lands just after the
    CREATE DATABASE has run, before `_database` yields."""
    connect = psycopg.connect

    class Interrupting:
        def __init__(self, conn: Conn) -> None:
            self.conn = conn

        def __enter__(self) -> "Interrupting":
            return self

        def __exit__(self, *exc: object) -> None:
            self.conn.close()

        def execute(self, query: Any) -> None:
            self.conn.execute(query)
            if "CREATE DATABASE" in query.as_string(self.conn):
                raise SystemExit(143)

    wrapped: list[Interrupting] = []

    def interrupting_connect(*args: Any, **kwargs: Any) -> Any:
        conn = connect(*args, **kwargs)
        if wrapped:  # the DROP runs on a plain connection
            return conn
        wrapped.append(Interrupting(conn))
        return wrapped[0]

    monkeypatch.setattr(selftest.psycopg, "connect", interrupting_connect)
    with pytest.raises(SystemExit), selftest._database(db_url):
        pytest.fail("not reached")
    monkeypatch.undo()
    assert wrapped  # the CREATE ran; `no_leftovers` checks that the database is gone
    assert not selftest._created


def test_dropping_a_database_that_does_not_exist_is_harmless(db_url: str) -> None:
    name = f"{selftest.PREFIX}{'f' * 16}"  # registered, as before the CREATE, never created
    selftest._created.add(name)
    selftest.drop_database(db_url, name)
    assert name not in selftest._created
    assert name not in selftest_databases(db_url)


def test_the_drop_guard_refuses_any_database_it_did_not_create(
    db_url: str, admin_url: str, no_leftovers: None
) -> None:
    configured = psycopg.conninfo.conninfo_to_dict(db_url)["dbname"]
    assert isinstance(configured, str)
    stranger = f"{selftest.PREFIX}{'0' * 16}"  # the right prefix, but not created by it
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(stranger)))
    try:
        for name in (configured, stranger, "postgres"):
            with pytest.raises(ValueError, match="refusing to drop"):
                selftest.drop_database(db_url, name)
        # A name it did create is refused when it is the configured database's name.
        selftest._created.add(configured)
        try:
            with pytest.raises(ValueError, match="refusing to drop"):
                selftest.drop_database(db_url, configured)
        finally:
            selftest._created.discard(configured)
        with psycopg.connect(admin_url) as admin:
            names = {r[0] for r in admin.execute("SELECT datname FROM pg_database")}
        assert {configured, stranger} <= names
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(stranger)))


def test_selftest_against_a_running_server(
    db_url: str,
    server: tuple[int, Any],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without --scenario: the reference query against `tgi mcp --http` at MCP_BIND, here
    loaded with the reference scenario; the link checks run, the A9 expectations do not."""
    now = datetime.now(UTC)
    selftest._load(db_url, selftest._shifted(SCENARIO, now), Settings.from_env({}))
    port, _ = server
    monkeypatch.setenv("MCP_BIND", f"127.0.0.1:{port}")
    monkeypatch.setenv("MCP_TOKEN", TOKEN)
    result = invoke(db_url, monkeypatch)
    assert result.exit_code == 0, result.output
    assert result.stdout == "selftest ok: tags=2 hits=7 context=4\n"
    monkeypatch.setenv("MCP_TOKEN", "x" * 43)
    result = invoke(db_url, monkeypatch)
    assert result.exit_code == 3, result.output
    assert "selftest failed: the server refused the token (HTTP 401)" in result.stderr
    assert "x" * 43 not in result.output
    monkeypatch.delenv("MCP_TOKEN")
    assert invoke(db_url, monkeypatch).exit_code == 1


def outcome(*windows: list[str]) -> selftest.Outcome:
    hits = [{"messages": [{"message_id": i, "link": link} for i, link in enumerate(w)]}
            for w in windows]  # fmt: skip
    return selftest.Outcome(tags=1, hits=hits, context=[])


EXPECT = {"seeded": ["s1", "s2", "s3"], "excluded": ["x"], "unrelated": ["u1", "u2"]}


def test_unrelated_hits_must_rank_below_every_seeded_hit() -> None:
    ok = outcome(["s1"], ["s2", "s3"], ["u1"], ["u2"])
    assert selftest.check_expected(ok, EXPECT) == {"seeded": 3, "unrelated_returned": 2}
    assert selftest.check_expected(outcome(["s1"], ["s2"], ["s3"]), EXPECT)["seeded"] == 3
    for bad in (
        outcome(["s1"], ["u1"], ["s2"], ["s3"]),  # between seeded hits
        outcome(["s1"], ["s2"], ["s3", "u2"]),  # in the same window as a seeded hit
        outcome(["u1"], ["s1"], ["s2"], ["s3"]),  # first
    ):
        with pytest.raises(selftest.Failed, match="1 unrelated messages rank with or above"):
            selftest.check_expected(bad, EXPECT)
    with pytest.raises(selftest.Failed, match="1 excluded"):
        selftest.check_expected(outcome(["s1"], ["s2"], ["s3"], ["x"]), EXPECT)
    with pytest.raises(selftest.Failed, match="2 of 3 seeded"):
        selftest.check_expected(outcome(["s1"], ["s2"]), EXPECT)


def test_link_checks() -> None:
    good = selftest.Outcome(
        tags=1,
        hits=[{"messages": [{"message_id": 7, "link": "https://t.me/c/1700000003/1"}]}],
        context=[{"message_id": 6, "link": None}, {"message_id": 7, "link": "https://t.me/c/1700000003/1"}],
    )  # fmt: skip
    selftest.check_links(good)
    bad_link = json.loads(json.dumps(good.__dict__))
    bad_link["context"][0]["link"] = "https://example.com/c/1/2"
    with pytest.raises(selftest.Failed, match="1 links are not t.me"):
        selftest.check_links(selftest.Outcome(**bad_link))
    other = json.loads(json.dumps(good.__dict__))
    other["context"][1]["link"] = "https://t.me/c/1700000003/2"
    with pytest.raises(selftest.Failed, match="get_context does not return the top message"):
        selftest.check_links(selftest.Outcome(**other))


def test_the_client_log_level_is_restored_on_failure_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)  # the root at DEBUG: the client would log at DEBUG
    client = logging.getLogger("mcp.client")
    assert client.level == logging.NOTSET
    with socket.socket() as s:  # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(selftest.Unreachable, match="^cannot connect to the MCP server$"):
        selftest.run(f"http://127.0.0.1:{port}/mcp", "x" * 43)
    assert client.level == logging.NOTSET
    with selftest._quiet_client():
        assert client.level == logging.INFO
    assert client.level == logging.NOTSET
    for name in selftest._CLIENT_LEVELS:  # the request URL is logged at INFO
        assert logging.getLogger(name).level == logging.NOTSET
    assert URL.findall(caplog.text) == []
    with selftest._quiet_client():
        assert logging.getLogger("httpx2").level == logging.WARNING
    assert logging.getLogger("httpx2").level == logging.NOTSET
