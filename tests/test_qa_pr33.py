"""QA for PR #33: a cold TDLib client loads the chat lists once before its first chat request.

The lib below is stricter than the PR's `FakeLib(cold=True)`: coldness is per *client id*
(a new client forgets every chat, like a real TDLib without a message database), and a chat
is known only after both lists were loaded to their 404 *and* if it belongs to a list.
Fake lib only; real TDLib is never contacted.
"""

import logging
from pathlib import Path
from typing import Any

import pytest
from test_td import CH, FORBIDDEN, PHONE, SG, FakeLib, Json, client, fake_tdlib  # noqa: F401
from typer.testing import CliRunner

from tgi import td
from tgi.cli import app
from tgi.td_port import TdError

NOWHERE = -5  # in no chat list


class ColdLib(FakeLib):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.lists_done: dict[int, int] = {}  # client id -> 404s seen (2 = both lists loaded)

    def loadChats(self, client_id: int, q: Json) -> Json:  # noqa: N802
        answer = super().loadChats(client_id, q)
        if answer.get("code") == 404:
            self.lists_done[client_id] = self.lists_done.get(client_id, 0) + 1
        return answer

    def _gate(self, client_id: int, q: Json) -> Json | None:
        if self.lists_done.get(client_id, 0) < 2 or q["chat_id"] == NOWHERE:
            return {"@type": "error", "code": 400, "message": "Chat not found"}
        return None

    def getChat(self, client_id: int, q: Json) -> Json:  # noqa: N802
        return self._gate(client_id, q) or super().getChat(client_id, q)

    def getChatHistory(self, client_id: int, q: Json) -> Json:  # noqa: N802
        return self._gate(client_id, q) or super().getChatHistory(client_id, q)


@pytest.fixture
def cold(fake_tdlib: FakeLib, monkeypatch: pytest.MonkeyPatch) -> ColdLib:  # noqa: F811
    lib = ColdLib(logged_in=True)
    monkeypatch.setattr(td, "load_tdjson", lambda path: lib)
    assert CliRunner().invoke(app, ["accounts", "add", "--phone", PHONE]).exit_code == 0
    lib.sent.clear()
    lib.loads = 0
    lib.lists_done.clear()
    return lib


def loads(lib: FakeLib) -> int:
    return len(lib.requests("loadChats"))


def test_cold_spike(cold: ColdLib, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    r = CliRunner().invoke(app, ["td", "spike", str(CH), "--account", PHONE, "--pages", "2"])
    assert r.exit_code == 0, r.output
    assert f"chat {CH} kind=channel" in r.stdout and "page 2:" in r.stdout
    assert loads(cold) == 4  # once for the whole command: getChat then getChatHistory
    assert [f for f in FORBIDDEN if f in r.stdout + r.stderr + caplog.text] == []


def test_cold_chats_enable_then_sync_in_fresh_processes(
    cold: ColdLib, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    r = runner.invoke(app, ["chats", "enable", str(SG), "--account", PHONE])
    assert r.exit_code == 0 and r.stdout == f"enabled {SG}\n", r.output
    assert loads(cold) == 4
    monkeypatch.setattr(td, "_hub", None)  # a new process
    r = runner.invoke(app, ["sync"])
    assert r.exit_code == 0, r.output
    assert "fetched=" in r.stdout and " ok " in r.stdout
    assert loads(cold) == 8  # the second command's client is cold again: one more load
    assert len(cold.requests("getChatHistory")) >= 2  # failed cold try + retry (or ingest)


def test_chat_in_no_list_fails_like_before_and_does_not_reload(
    cold: ColdLib, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    r = runner.invoke(app, ["td", "spike", str(NOWHERE), "--account", PHONE])
    assert r.exit_code == 3 and r.stderr == "spike failed: TdError 400\n"
    assert loads(cold) == 4  # one load, one retry, no more
    assert len(cold.requests("getChat")) == 2
    r = runner.invoke(app, ["chats", "enable", str(NOWHERE), "--account", PHONE])
    assert r.exit_code == 3 and r.stderr == "chats enable failed: TdError 400\n"
    assert loads(cold) == 8  # one load per command (per client), not per request


def test_one_load_per_client_across_chats_and_failures(tmp_path: Path) -> None:
    lib = ColdLib(logged_in=True)
    c = client(tmp_path, lib)
    for _ in range(3):
        with pytest.raises(TdError, match="Chat not found"):
            c.get_chat(NOWHERE)
        with pytest.raises(TdError, match="Chat not found"):
            c.get_chat_history(NOWHERE, 0, 100)
    assert c.get_chat(SG).kind == "supergroup" and c.get_chat(CH).kind == "channel"
    assert len(c.get_chats()) == 3  # an explicit listing reloads by itself; fine
    assert loads(lib) == 8
    c.close()
    c2 = client(tmp_path, lib)  # a new client: cold again
    assert c2.get_chat(SG).kind == "supergroup"
    assert loads(lib) == 12
    c2.close()


def test_get_chats_on_a_cold_client_loads_once(tmp_path: Path) -> None:
    lib = ColdLib(logged_in=True)
    c = client(tmp_path, lib)
    assert [x.chat_id for x in c.get_chats()] == [SG, 42, CH]
    assert loads(lib) == 4  # the per-chat getChat calls trigger no second load
    c.close()


@pytest.mark.parametrize(
    "code,message",
    [
        (400, "Parameter limit must be in [1, 100]"),
        (500, "Chat not found"),
        (403, "Chat not found"),
        (404, "Chat not found"),
    ],  # fmt: skip
)
def test_only_400_chat_not_found_triggers_a_load(tmp_path: Path, code: int, message: str) -> None:
    class Other(ColdLib):
        def getChat(self, client_id: int, q: Json) -> Json:  # noqa: N802
            return {"@type": "error", "code": code, "message": message}

    lib = Other(logged_in=True)
    c = client(tmp_path, lib)
    with pytest.raises(TdError) as caught:
        c.get_chat(SG)
    assert caught.value.code == code and loads(lib) == 0
    assert len(lib.requests("getChat")) == 1  # no retry
    c.close()


def test_flood_wait_is_not_retried_and_loads_nothing(tmp_path: Path) -> None:
    lib = ColdLib(logged_in=True)
    lib.flood["getChat"] = 7
    c = client(tmp_path, lib)
    with pytest.raises(td.FloodWait):
        c.get_chat(SG)
    assert loads(lib) == 0 and len(lib.requests("getChat")) == 1
    c.close()


def test_cap_holds_on_the_cold_path_and_failed_load_is_not_marked_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Endless(ColdLib):
        def loadChats(self, client_id: int, q: Json) -> Json:  # noqa: N802  # never 404
            return {"@type": "ok"}

    monkeypatch.setattr(td, "LOAD_CHATS_MAX_CALLS", 5)
    lib = Endless(logged_in=True)
    c = client(tmp_path, lib)
    with pytest.raises(td.Unfinished):
        c.get_chat(SG)
    assert loads(lib) == 5 and len(lib.requests("getChat")) == 1  # no retry after the cap
    with pytest.raises(td.Unfinished):  # not marked loaded: the next request tries again (bounded)
        c.get_chat(SG)
    assert loads(lib) == 10
    c.close()


def test_a_chat_that_appears_after_the_load_is_not_found_without_a_reload(tmp_path: Path) -> None:
    lib = ColdLib(logged_in=True)
    c = client(tmp_path, lib)
    assert c.get_chat(SG).kind == "supergroup"  # lists loaded
    with pytest.raises(TdError):
        c.get_chat(NOWHERE)
    assert loads(lib) == 4  # by design (spec: load once per client)
    c.close()
