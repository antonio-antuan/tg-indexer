# ruff: noqa: E501, F811
"""QA T3: probes of the TDLib adapter and its CLI against the fake libtdjson (no network)."""

import logging
import os
import stat
from pathlib import Path

import psycopg
import pytest
from test_td import (
    CH,
    CODE,
    FORBIDDEN,
    PASSWORD,
    PHONE,
    SECRETS,
    SG,
    FakeLib,
    client,
    fake_tdlib,  # noqa: F401  (fixture)
    settings,
)
from typer.testing import CliRunner

from tgi.cli import app
from tgi.td_port import TdError

Conn = psycopg.Connection[tuple[object, ...]]
POISON = f"leak {PHONE} {CODE} {PASSWORD} {SECRETS['TG_API_HASH']} {SECRETS['TDLIB_DB_KEY']}"


def test_sync_with_an_account_that_is_not_logged_in(
    fake_tdlib: FakeLib,
    db_url: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled chat whose account has no session: sync must not prompt or hang."""
    monkeypatch.setenv("DATABASE_URL", db_url)
    fake_tdlib.logged_in = True
    runner = CliRunner()
    assert runner.invoke(app, ["accounts", "add", "--phone", PHONE]).exit_code == 0
    assert runner.invoke(app, ["chats", "enable", str(SG), "--account", PHONE]).exit_code == 0
    # drop the session: the account directory has no td/ any more
    import shutil

    shutil.rmtree(Path(os.environ["DATA_DIR"]) / "accounts" / PHONE / "td")
    fake_tdlib.logged_in = False
    r = runner.invoke(app, ["sync"])

    assert r.exit_code != 0
    out = r.stdout + r.stderr
    assert [f for f in FORBIDDEN if f in out] == []
    assert fake_tdlib.requests("setAuthenticationPhoneNumber") == []


@pytest.mark.parametrize("kind", ["setTdlibParameters", "getOption"])
def test_open_errors_never_leak_poison(
    fake_tdlib: FakeLib,
    db_url: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("DATABASE_URL", db_url)
    fake_tdlib.logged_in = True
    runner = CliRunner()
    assert runner.invoke(app, ["accounts", "add", "--phone", PHONE]).exit_code == 0
    assert runner.invoke(app, ["chats", "enable", str(SG), "--account", PHONE]).exit_code == 0
    orig = getattr(fake_tdlib, kind)
    fake_tdlib.__dict__[kind] = lambda cid, q: {"@type": "error", "code": 401, "message": POISON}
    outs = []
    for argv in (
        ["sync"],
        ["chats", "list", "--account", PHONE],
        ["td", "spike", str(SG), "--account", PHONE],
        ["accounts", "add", "--phone", PHONE],
    ):
        r = runner.invoke(app, argv)
        print(argv[0], r.exit_code, repr(r.stderr))
        outs.append(r.stdout + r.stderr + (repr(r.exception) if r.exception else ""))
        assert r.exit_code in (1, 3), (argv, r.exit_code)
    assert [f for f in FORBIDDEN + ["leak"] if any(f in o for o in outs)] == []
    assert "leak" not in caplog.text
    del fake_tdlib.__dict__[kind]
    assert orig


def test_account_dirs_are_tightened_even_under_a_permissive_umask(tmp_path: Path) -> None:
    old = os.umask(0)
    try:
        s = settings(tmp_path)
        lib = FakeLib(logged_in=True)
        accounts = tmp_path / "data" / "accounts"
        (accounts / PHONE / "td").mkdir(parents=True, mode=0o755)
        accounts.chmod(0o777)
        (accounts / PHONE).chmod(0o755)
        (tmp_path / "data" / "lib").mkdir()
        (tmp_path / "data" / "lib" / "libtdjson.so").write_bytes(b"")
        import tgi.td as t

        t._hub = None
        mp = pytest.MonkeyPatch()
        mp.setattr(t, "load_tdjson", lambda p: lib)
        try:
            c = t.open_client(s, PHONE)
            c.close()
        finally:
            mp.undo()
            t._hub = None
        for p in (accounts, accounts / PHONE):
            assert stat.S_IMODE(p.stat().st_mode) == 0o700, p
    finally:
        os.umask(old)


def test_silent_tdlib_times_out_instead_of_hanging(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    lib.silent.add("getChat")
    with pytest.raises(TdError) as e:
        c.get_chat(SG)
    assert e.value.code == 408
    lib.silent.clear()
    assert c.get_chat(SG).chat_id == SG  # the late answer to the dead request is skipped
    c.close()


def test_unrelated_and_foreign_events_do_not_break_routing(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    lib.events.append({"@type": "updateOption", "name": "x"})  # no @client_id
    lib.events.append({"@type": "updateNewMessage", "@client_id": 99, "message": {}})
    c = client(tmp_path, lib)
    assert c.get_chat(CH).username == "avisos"
    c.close()
