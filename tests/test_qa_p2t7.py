# ruff: noqa: E501, S603, ANN001, ANN202, S101
"""QA P2.T7: SIGTERM at the CREATE DATABASE edge, no request URL in selftest logs."""

import logging
import secrets  # the real module, unaffected by the monkeypatch of `selftest.secrets`
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from selftest_dbs import PIN, Owned, child_env, drop, exists
from typer.testing import CliRunner

from tgi import selftest
from tgi.cli import app

REFERENCE = Path(__file__).parent / "scenarios" / "reference.json"

# Wraps psycopg's execute: SIGTERM is sent to this process just before (`before`) or just
# after (`after`) the CREATE DATABASE statement, which is where the gap was.
CHILD = (
    PIN
    + """
import os, signal, sys
import psycopg
from tgi.cli import app
when = sys.argv[1]
real = psycopg.Connection.execute
def execute(self, query, *a, **k):
    text = query.as_string(self) if hasattr(query, "as_string") else str(query)
    is_create = text.startswith("CREATE DATABASE")
    if is_create and when == "before":
        os.kill(os.getpid(), signal.SIGTERM)
    out = real(self, query, *a, **k)
    if is_create and when == "after":
        os.kill(os.getpid(), signal.SIGTERM)
    return out
psycopg.Connection.execute = execute
app(sys.argv[2:])
"""
)


@pytest.mark.parametrize("when", ["before", "after"])
def test_sigterm_at_the_create_database_edge_leaves_no_database(owned: Owned, when: str) -> None:
    done = subprocess.run(
        [sys.executable, "-c", CHILD, when, "selftest", "--scenario", str(REFERENCE)],
        env=child_env(owned), capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert done.returncode == 143, (done.returncode, done.stderr[-400:])
    assert not any(exists(owned.db_url, n) for n in owned.names)
    assert "Traceback" not in done.stderr


def test_a_foreign_selftest_database_survives_a_sigterm_child(owned: Owned) -> None:
    # DR-8 (1): a concurrent selftest's database is neither the child's nor the fixture's.
    foreign = selftest.PREFIX + "foreign" + secrets.token_hex(8)
    with psycopg.connect(owned.db_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(foreign)))
    try:
        done = subprocess.run(
            [sys.executable, "-c", CHILD, "after", "selftest", "--scenario", str(REFERENCE)],
            env=child_env(owned), capture_output=True, text=True, timeout=120,
        )  # fmt: skip
        assert done.returncode == 143, (done.returncode, done.stderr[-400:])
        assert not any(exists(owned.db_url, n) for n in owned.names)
        assert exists(owned.db_url, foreign)
    finally:
        drop(owned.db_url, foreign)


def test_a_taken_candidate_name_is_skipped_never_registered_never_dropped(
    owned_over_taken: Owned, taken: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # DR-8 (2): the first random candidate is already on the server.
    owned, second = owned_over_taken, secrets.token_hex(8)
    owned.fresh = iter([taken, second]).__next__
    monkeypatch.setenv("DATABASE_URL", owned.db_url)
    result = CliRunner().invoke(app, ["selftest", "--scenario", str(REFERENCE)])
    assert result.exit_code == 0, result.output
    assert owned.names == [selftest.PREFIX + second]
    assert exists(owned.db_url, selftest.PREFIX + taken)
    assert not exists(owned.db_url, selftest.PREFIX + second)


def test_dropping_a_registered_name_that_does_not_exist_is_harmless(db_url: str) -> None:
    name = selftest.PREFIX + "0" * 16
    selftest._created.add(name)
    try:
        selftest.drop_database(db_url, name)
        assert name not in selftest._created
    finally:
        selftest._created.discard(name)


@pytest.mark.parametrize("name", ["postgres", selftest.PREFIX + "ffff", "other_db"])
def test_the_guard_still_refuses_unregistered_names(db_url: str, name: str) -> None:
    with pytest.raises(ValueError):
        selftest.drop_database(db_url, name)


def test_no_request_url_in_scenario_stderr_even_at_debug(owned: Owned) -> None:
    code = (
        PIN
        + "import logging, sys; logging.basicConfig(level=logging.DEBUG); from tgi.cli import app; app(sys.argv[1:])"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, "selftest", "--scenario", str(REFERENCE)],
        env=child_env(owned), capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-400:]
    out = done.stdout + done.stderr
    assert "127.0.0.1" not in out and "HTTP Request" not in out and "Bearer" not in out


def test_logger_levels_are_restored_after_success_and_failure(db_url: str) -> None:
    names = list(selftest._CLIENT_LEVELS)
    before = {n: logging.getLogger(n).level for n in names}
    with selftest._quiet_client():
        for n, lvl in selftest._CLIENT_LEVELS.items():
            assert logging.getLogger(n).getEffectiveLevel() >= lvl
    assert {n: logging.getLogger(n).level for n in names} == before
    with pytest.raises(RuntimeError), selftest._quiet_client():
        raise RuntimeError
    assert {n: logging.getLogger(n).level for n in names} == before


def test_a_failed_check_still_exits_1_and_leaves_nothing(
    owned: Owned, tmp_path: Path, monkeypatch
) -> None:
    import json

    data = json.loads(REFERENCE.read_text())
    data["reference"]["tags"].pop("valencia")
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data))
    monkeypatch.setenv("DATABASE_URL", owned.db_url)
    result = CliRunner().invoke(app, ["selftest", "--scenario", str(path)])
    assert result.exit_code == 1
    assert owned.names and not any(exists(owned.db_url, n) for n in owned.names)
