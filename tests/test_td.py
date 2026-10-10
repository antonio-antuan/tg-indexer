"""P1.T3: the TDLib adapter (`tgi.td`) and its commands, against a fake `libtdjson`.

`FakeLib` speaks TDLib's JSON interface: events carry `@client_id`, answers carry the
request's `@extra`, and authorization goes through `updateAuthorizationState`. Real TDLib
is never contacted here; `scripts/tdlib.sh` loads the real library (`td_execute` only).
"""

import base64
import json
import logging
import stat
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tgi import cli, td
from tgi.cli import app
from tgi.config import Settings, env_file_warning
from tgi.td_port import Chat, FloodWait, JoinPending, Message, NotJoinable, TdError, parse_link

Json = dict[str, Any]
PHONE = "+34600000999"
CODE, PASSWORD = "24680", "pw-SENTINEL-4"
SECRETS = {"TDLIB_DB_KEY": "db-key-SENTINEL-1", "TG_API_ID": "123456",
           "TG_API_HASH": "api-hash-SENTINEL-2"}  # fmt: skip
SG = -1001234567890  # a supergroup without username
CH = -1009876543210  # a channel with a username
TEXT = "секретный текст сообщения"
HINT = "pets"  # the 2FA hint, shown at the prompt


def msg(i: int, chat_id: int = SG, **extra: Any) -> Json:
    content = {"@type": "messageText", "text": {"text": f"{TEXT} {i}"}}
    return {"@type": "message", "id": i << 20, "chat_id": chat_id, "date": 1_790_000_000 + i,
            "edit_date": 0, "sender_id": {"@type": "messageSenderUser", "user_id": 7},
            "content": content, **extra}  # fmt: skip


class FakeLib:
    """A scripted TDLib: one account, its chats and history (newest first)."""

    def __init__(self, logged_in: bool = False, two_fa: bool = True, cold: bool = False) -> None:
        self.logged_in, self.two_fa = logged_in, two_fa
        self.cold = cold  # real TDLib without a message database: no chat before loadChats
        self.events: deque[Json] = deque()
        self.sent: list[tuple[int, Json]] = []
        self.clients = 0
        self.started: set[int] = set()
        self.flood: dict[str, int] = {}  # request type -> retry after
        self.silent: set[str] = set()  # request types never answered
        self.loads = 0
        self.history = [msg(i) for i in range(5, 0, -1)]
        self.state_after_phone = "authorizationStateWaitCode"
        self.requests_to_join: set[int] = set()  # public chats whose joinChat sends a request
        self.approved: set[str] = set()  # join-request links whose request was approved

    # --- the four libtdjson functions ---------------------------------------------------

    def td_create_client_id(self) -> int:
        self.clients += 1
        return self.clients

    def td_send(self, client_id: int, request: bytes) -> None:
        query = json.loads(request)
        self.sent.append((client_id, query))
        if client_id not in self.started:  # TDLib's first update for a new client
            self.started.add(client_id)
            self._state(client_id, "authorizationStateWaitTdlibParameters")
        kind = query["@type"]
        if kind in self.silent:
            return
        if kind in self.flood:
            err = {"@type": "error", "code": 429,
                   "message": f"Too Many Requests: retry after {self.flood[kind]}"}  # fmt: skip
            return self._answer(client_id, query, err)
        self._answer(client_id, query, getattr(self, kind)(client_id, query))

    def td_receive(self, timeout: float) -> bytes | None:
        return json.dumps(self.events.popleft()).encode() if self.events else None

    def td_execute(self, request: bytes) -> bytes | None:
        query = json.loads(request)
        if query["@type"] == "getOption":
            return json.dumps({"@type": "optionValueString", "value": "1.8.67"}).encode()
        return json.dumps({"@type": "ok"}).encode()

    # --- helpers ------------------------------------------------------------------------

    def _state(self, client_id: int, state: str, **extra: Any) -> None:
        self.events.append({"@type": "updateAuthorizationState", "@client_id": client_id,
                            "authorization_state": {"@type": state, **extra}})  # fmt: skip

    def _answer(self, client_id: int, query: Json, answer: Json) -> None:
        self.events.append({**answer, "@client_id": client_id, "@extra": query.get("@extra")})

    def requests(self, kind: str) -> list[Json]:
        return [q for _, q in self.sent if q["@type"] == kind]

    # --- requests -----------------------------------------------------------------------

    def getOption(self, client_id: int, q: Json) -> Json:  # noqa: N802
        return {"@type": "optionValueString", "value": "1.8.67"}

    def setTdlibParameters(self, client_id: int, q: Json) -> Json:  # noqa: N802
        Path(q["database_directory"]).mkdir(parents=True, exist_ok=True)  # the session
        ready = "authorizationStateReady" if self.logged_in else "authorizationStateWaitPhoneNumber"
        self._state(client_id, ready)
        return {"@type": "ok"}

    def setAuthenticationPhoneNumber(self, client_id: int, q: Json) -> Json:  # noqa: N802
        self._state(client_id, self.state_after_phone)
        return {"@type": "ok"}

    def checkAuthenticationCode(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if q["code"] != CODE:
            return {"@type": "error", "code": 400, "message": "PHONE_CODE_INVALID"}
        if self.two_fa:
            self._state(client_id, "authorizationStateWaitPassword", password_hint=HINT)
        else:
            self._login(client_id)
        return {"@type": "ok"}

    def checkAuthenticationPassword(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if q["password"] != PASSWORD:
            return {"@type": "error", "code": 400, "message": "PASSWORD_HASH_INVALID"}
        self._login(client_id)
        return {"@type": "ok"}

    def _login(self, client_id: int) -> None:
        self.logged_in = True
        self._state(client_id, "authorizationStateReady")

    def close(self, client_id: int, q: Json) -> Json:
        self._state(client_id, "authorizationStateClosing")
        self._state(client_id, "authorizationStateClosed")
        return {"@type": "ok"}

    def loadChats(self, client_id: int, q: Json) -> Json:  # noqa: N802
        self.loads += 1
        if self.loads % 2 == 0:  # each list: one load, then 404 (all loaded)
            return {"@type": "error", "code": 404, "message": "Not Found"}
        return {"@type": "ok"}

    def getChats(self, client_id: int, q: Json) -> Json:  # noqa: N802
        main = q["chat_list"]["@type"] == "chatListMain"
        return {"@type": "chats", "total_count": 2, "chat_ids": [SG, 42] if main else [CH, SG]}

    def _unknown(self) -> Json | None:
        if self.cold and not self.loads:
            return {"@type": "error", "code": 400, "message": "Chat not found"}
        return None

    def getChat(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if unknown := self._unknown():
            return unknown
        types = {
            SG: {"@type": "chatTypeSupergroup", "supergroup_id": 1234567890, "is_channel": False},
            CH: {"@type": "chatTypeSupergroup", "supergroup_id": 9876543210, "is_channel": True},
            42: {"@type": "chatTypePrivate", "user_id": 42},
            -77: {"@type": "chatTypeBasicGroup", "basic_group_id": 77},
        }
        if q["chat_id"] not in types:
            return {"@type": "error", "code": 400, "message": "Chat not found"}
        return {"@type": "chat", "id": q["chat_id"], "title": f"chat {q['chat_id']}",
                "type": types[q["chat_id"]]}  # fmt: skip

    def searchPublicChat(self, client_id: int, q: Json) -> Json:  # noqa: N802
        ids = {"avisos": CH, "public_sg": SG, "ana_person": 42}
        if q["username"] not in ids:
            return {"@type": "error", "code": 400, "message": "USERNAME_NOT_OCCUPIED"}
        return self.getChat(client_id, {"chat_id": ids[q["username"]]})

    def joinChat(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if q["chat_id"] in self.requests_to_join:
            return {"@type": "error", "code": 400, "message": "INVITE_REQUEST_SENT"}
        return {"@type": "ok"}

    INVITES: dict[str, Json] = {  # noqa: RUF012
        "https://t.me/+sgInvite12": {"type": "inviteLinkChatTypeSupergroup", "chat_id": 0},
        "https://t.me/+memberAlready": {"type": "inviteLinkChatTypeSupergroup", "chat_id": SG},
        "https://t.me/+basicInvite1": {"type": "inviteLinkChatTypeBasicGroup", "chat_id": 0},
        "https://t.me/+askAdmins12": {"type": "inviteLinkChatTypeChannel", "chat_id": 0,
                                      "creates_join_request": True},
    }  # fmt: skip

    def checkChatInviteLink(self, client_id: int, q: Json) -> Json:  # noqa: N802
        info = self.INVITES.get(q["invite_link"])
        if info is None:
            return {"@type": "error", "code": 400, "message": "INVITE_HASH_EXPIRED"}
        if q["invite_link"] in self.approved:  # a member now: the preview has the chat id
            info = {**info, "chat_id": CH}
        return {"@type": "chatInviteLinkInfo", "title": "t", **info,
                "type": {"@type": info["type"]}}  # fmt: skip

    def joinChatByInviteLink(self, client_id: int, q: Json) -> Json:  # noqa: N802
        link = q["invite_link"]
        if link.endswith("memberAlready") or link in self.approved:
            return {"@type": "error", "code": 400, "message": "USER_ALREADY_PARTICIPANT"}
        if link.endswith("askAdmins12"):
            return {"@type": "error", "code": 400, "message": "INVITE_REQUEST_SENT"}
        return self.getChat(client_id, {"chat_id": SG})

    def getSupergroup(self, client_id: int, q: Json) -> Json:  # noqa: N802
        names = ["avisos", "avisos_old"] if q["supergroup_id"] == 9876543210 else []
        return {"@type": "supergroup", "id": q["supergroup_id"],
                "usernames": {"@type": "usernames", "active_usernames": names}}  # fmt: skip

    def getUser(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if q["user_id"] == 8:
            return {"@type": "error", "code": 404, "message": "User not found"}
        return {"@type": "user", "id": q["user_id"], "first_name": "Ana", "last_name": "López"}

    def getChatHistory(self, client_id: int, q: Json) -> Json:  # noqa: N802
        if unknown := self._unknown():
            return unknown
        start = q["from_message_id"]  # inclusive, like TDLib
        older = [m for m in self.history if m is None or not start or m["id"] <= start]
        page = older[: q["limit"]] if q["from_message_id"] else older[:2]  # cold start: short
        return {"@type": "messages", "total_count": len(page), "messages": page}


class Prompt:
    def __init__(self, codes: list[str], passwords: list[str]) -> None:
        self.codes, self.passwords = codes, passwords
        self.hints: list[str] = []
        self.rejections: list[tuple[str, int]] = []

    def code(self) -> str:
        return self.codes.pop(0)

    def password(self, hint: str) -> str:
        self.hints.append(hint)
        return self.passwords.pop(0)

    def rejected(self, what: str, error: TdError) -> None:
        self.rejections.append((what, error.code))


def settings(tmp_path: Path) -> Settings:
    return Settings.from_env({"DATA_DIR": str(tmp_path / "data"), **SECRETS})


def client(tmp_path: Path, lib: FakeLib, prompt: td.Prompt | None = None) -> td.TdlibClient:
    s = settings(tmp_path)
    params = td.tdlib_parameters(s, td.account_dir(s, PHONE))
    return td.TdlibClient(td.Hub(lib), params, PHONE, prompt, timeout=0.2)


# --- login ------------------------------------------------------------------------------


def test_login_with_a_rejected_code_and_2fa(tmp_path: Path) -> None:
    lib = FakeLib()
    prompt = Prompt(["11111", CODE], ["wrong", PASSWORD])
    c = client(tmp_path, lib, prompt)
    assert prompt.rejections == [("code", 400), ("password", 400)]
    assert prompt.hints == [HINT, HINT]
    assert [q["phone_number"] for q in lib.requests("setAuthenticationPhoneNumber")] == [PHONE]
    assert lib.logged_in
    c.close()
    assert len(lib.requests("close")) == 1
    c.close()  # idempotent
    assert len(lib.requests("close")) == 1
    with pytest.raises(TdError) as err:
        c.get_chat(SG)
    assert err.value.code == 500


def test_three_rejected_codes_end_the_login_and_close_the_client(tmp_path: Path) -> None:
    lib = FakeLib()
    with pytest.raises(TdError) as err:
        client(tmp_path, lib, Prompt(["1", "2", "3"], []))
    assert err.value.code == 400
    assert len(lib.requests("checkAuthenticationCode")) == td.CODE_ATTEMPTS
    assert len(lib.requests("close")) == 1


def test_a_logged_in_account_opens_without_prompting(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    assert lib.requests("setAuthenticationPhoneNumber") == []
    c.close()


def test_without_a_prompt_a_logged_out_account_raises_not_logged_in(tmp_path: Path) -> None:
    lib = FakeLib()
    with pytest.raises(td.NotLoggedIn):
        client(tmp_path, lib)
    assert lib.requests("setAuthenticationPhoneNumber") == []
    assert len(lib.requests("close")) == 1


def test_unsupported_login_states_are_refused(tmp_path: Path) -> None:
    lib = FakeLib()
    lib.state_after_phone = "authorizationStateWaitEmailAddress"
    with pytest.raises(TdError, match="email address: not supported") as err:
        client(tmp_path, lib, Prompt([], []))
    assert err.value.code == 400


def test_a_close_timeout_after_a_failed_login_keeps_the_login_error(tmp_path: Path) -> None:
    lib = FakeLib()
    lib.silent.add("close")
    with pytest.raises(td.NotLoggedIn):
        client(tmp_path, lib)


def test_tdlib_parameters(tmp_path: Path) -> None:
    s = settings(tmp_path)
    account = td.account_dir(s, PHONE)
    assert account == tmp_path / "data" / "accounts" / PHONE
    p = td.tdlib_parameters(s, account)
    assert p["database_directory"] == str(account / "td")
    assert base64.b64decode(p["database_encryption_key"]).decode() == SECRETS["TDLIB_DB_KEY"]
    assert (p["api_id"], p["api_hash"]) == (123456, SECRETS["TG_API_HASH"])
    assert p["use_message_database"] is False and p["use_test_dc"] is False


@pytest.mark.parametrize(
    ("env", "phone", "error"),
    [
        ({"TDLIB_DB_KEY": ""}, PHONE, "TDLIB_DB_KEY not set"),
        ({"TG_API_ID": "abc"}, PHONE, "TG_API_ID must be an integer"),
        ({"DATA_DIR": ""}, PHONE, "DATA_DIR is not set"),
        ({}, "34600000999", "the phone must look like"),
        ({}, "+34 600", "the phone must look like"),
        ({}, "+34600000999/../x", "the phone must look like"),
    ],
)
def test_setup_errors(tmp_path: Path, env: dict[str, str], phone: str, error: str) -> None:
    s = Settings.from_env({"DATA_DIR": str(tmp_path), **SECRETS, **env})
    with pytest.raises(td.SetupError, match=error) as err:
        td.open_client(s, phone, Prompt([], []))
    assert phone not in str(err.value) and SECRETS["TG_API_HASH"] not in str(err.value)


def test_open_client_needs_a_session_unless_it_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = FakeLib()
    lib_file = tmp_path / "libtdjson.so"
    lib_file.write_bytes(b"")
    monkeypatch.setattr(td, "_hub", None)
    monkeypatch.setattr(td, "load_tdjson", lambda path: lib)
    s = Settings.from_env({"DATA_DIR": str(tmp_path / "data"), "TDJSON_PATH": str(lib_file),
                           **SECRETS})  # fmt: skip
    with pytest.raises(td.NotLoggedIn):
        td.open_client(s, PHONE)
    assert lib.sent == []  # refused before TDLib is asked
    c = td.open_client(s, PHONE, Prompt([CODE], [PASSWORD]))
    c.close()
    accounts = tmp_path / "data" / "accounts"
    for d in (accounts, accounts / PHONE):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700
    assert td.hub(s) is td.hub(s)


def test_hub_errors_name_the_fix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(td, "_hub", None)
    s = Settings.from_env({"DATA_DIR": str(tmp_path), "TDJSON_PATH": str(tmp_path / "no.so")})
    with pytest.raises(td.SetupError, match="libtdjson not found"):
        td.hub(s)
    junk = tmp_path / "junk.so"
    junk.write_bytes(b"not a library")
    s = Settings.from_env({"DATA_DIR": str(tmp_path), "TDJSON_PATH": str(junk)})
    with pytest.raises(td.SetupError, match="does not load"):
        td.hub(s)


def test_hub_sends_tdlib_logs_to_a_private_file(tmp_path: Path) -> None:
    sent: list[Json] = []

    class Lib(FakeLib):
        def td_execute(self, request: bytes) -> bytes | None:
            sent.append(json.loads(request))
            return super().td_execute(request)

    td.Hub(Lib()).log_to(tmp_path / "tdlib.log")
    assert sent[0] == {"@type": "setLogVerbosityLevel", "new_verbosity_level": 1}
    stream = sent[1]["log_stream"]
    assert stream["path"] == str(tmp_path / "tdlib.log") and stream["redirect_stderr"] is False


# --- requests ---------------------------------------------------------------------------


def test_events_are_routed_by_client_id() -> None:
    hub = td.Hub(FakeLib())
    a, b = hub.create_client(), hub.create_client()
    hub._lib.events.extend([{"@client_id": b, "n": 1}, {"@client_id": 99, "n": 2},  # type: ignore[attr-defined]
                            {"@client_id": a, "n": 3}])  # fmt: skip
    assert hub.receive(a, 0.1) == {"@client_id": a, "n": 3}  # b's event is queued on the way
    assert hub.receive(b, 0.1) == {"@client_id": b, "n": 1}
    assert hub.receive(a, 0.05) is None  # an unknown client's event is dropped


def test_two_accounts_share_the_hub(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    hub = td.Hub(lib)
    s = settings(tmp_path)
    params = td.tdlib_parameters(s, td.account_dir(s, PHONE))
    a = td.TdlibClient(hub, params, PHONE, timeout=0.2)
    b = td.TdlibClient(hub, params, PHONE, timeout=0.2)
    assert (a.get_chat(SG).kind, b.get_chat(CH).kind) == ("supergroup", "channel")
    a.close()
    assert b.get_chat(-77).kind == "basic_group"
    b.close()
    assert {cid for cid, q in lib.sent if q["@type"] == "close"} == {1, 2}


def test_errors_and_flood_waits(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    with pytest.raises(TdError, match="Chat not found") as err:
        c.get_chat(-5)
    assert err.value.code == 400
    lib.flood["getChatHistory"] = 7
    with pytest.raises(FloodWait) as flood:
        c.get_chat_history(SG, 0, 100)
    assert flood.value.retry_after == 7
    lib.silent.add("getChatHistory")
    with pytest.raises(TdError) as timeout:
        c.get_chat_history(SG, 0, 100)
    assert timeout.value.code == 408
    c.close()


def test_get_chats_loads_both_lists_and_maps_kinds(tmp_path: Path) -> None:
    c = client(tmp_path, FakeLib(logged_in=True))
    assert c.get_chats() == [
        Chat(SG, f"chat {SG}", "supergroup"),
        Chat(42, "chat 42", "private"),
        Chat(CH, f"chat {CH}", "channel", "avisos"),
    ]
    assert c.get_chat(-77).kind == "basic_group"
    c.close()


def test_history_maps_messages(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    caption = {"@type": "messagePhoto", "caption": {"text": "фото"}}
    lib.history = [
        msg(
            9,
            edit_date=1_790_000_100,
            reply_to={"@type": "messageReplyToMessage", "chat_id": SG, "message_id": 8 << 20},
        ),  # fmt: skip
        msg(
            8,
            content=caption,
            reply_to={"@type": "messageReplyToMessage", "chat_id": CH, "message_id": 3 << 20},
        ),  # fmt: skip
        msg(
            7,
            content={"@type": "messageSticker"},
            sender_id={"@type": "messageSenderChat", "chat_id": CH},
        ),  # fmt: skip
        msg(
            6,
            content={"@type": "messageChatAddMembers"},
            sender_id={"@type": "messageSenderUser", "user_id": 8},
        ),  # fmt: skip
        msg(5, reply_to={"@type": "messageReplyToStory"}),
    ]
    c = client(tmp_path, lib)
    first = c.get_chat_history(SG, 10 << 20, 100)
    lib.history.insert(0, None)  # type: ignore[arg-type]  # TDLib may send null entries
    second = c.get_chat_history(SG, 10 << 20, 100)
    assert first == second
    at = lambda i: datetime.fromtimestamp(1_790_000_000 + i, UTC)  # noqa: E731
    assert first == [
        Message(
            SG,
            9 << 20,
            at(9),
            f"{TEXT} 9",
            7,
            "Ana López",
            datetime.fromtimestamp(1_790_000_100, UTC),
            8 << 20,
        ),  # fmt: skip
        Message(SG, 8 << 20, at(8), "фото", 7, "Ana López", has_media=True),
        Message(SG, 7 << 20, at(7), "", CH, f"chat {CH}", has_media=True),
        Message(SG, 6 << 20, at(6), "", 8, None),
        Message(SG, 5 << 20, at(5), f"{TEXT} 5", 7, "Ana López"),
    ]
    assert len(lib.requests("getUser")) == 2  # user 7 once (cached), user 8 once
    c.close()


# --- CLI --------------------------------------------------------------------------------


@pytest.fixture
def fake_tdlib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeLib:
    lib = FakeLib()
    lib_file = tmp_path / "libtdjson.so"
    lib_file.write_bytes(b"")
    monkeypatch.setattr(td, "_hub", None)
    monkeypatch.setattr(td, "load_tdjson", lambda path: lib)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TDJSON_PATH", str(lib_file))
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    return lib


FORBIDDEN = [PHONE, PHONE.lstrip("+"), CODE, PASSWORD, *SECRETS.values(), TEXT, "Ana"]


def test_cli_accounts_add_then_spike(fake_tdlib: FakeLib, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    runner = CliRunner()
    added = runner.invoke(app, ["accounts", "add", "--phone", PHONE],
                          input=f"000\n{CODE}\n{PASSWORD}\n")  # fmt: skip
    assert added.exit_code == 0, added.output
    assert added.stdout.endswith("account ready\n")
    assert "the code was rejected (TDLib 400); try again" in added.stderr

    # The session exists now (the fake TDLib remembers the login): the spike never prompts.
    spike = runner.invoke(app, ["td", "spike", str(CH), "--account", PHONE, "--pages", "3"])
    assert spike.exit_code == 0, spike.output
    lines = spike.stdout.splitlines()
    assert lines[:2] == ["tdlib 1.8.67", f"chat {CH} kind=channel username=yes"]
    assert lines[2].startswith("page 1: from_message_id=0 messages=2 ")
    out = spike.stdout
    assert f"message_id={5 << 20} server_message_id=5" in out
    assert f"page 2: from_message_id={4 << 20} messages=4" in out
    assert "from_inclusive=True" in out
    assert f"link for message_id={5 << 20}: https://t.me/avisos/5" in out
    assert out.splitlines()[-1].startswith("done in ")

    outputs = added.stdout + added.stderr + spike.stdout + spike.stderr + caplog.text
    assert [f for f in FORBIDDEN if f in outputs] == []


def test_cli_spike_without_a_session_and_with_a_bad_setup(fake_tdlib: FakeLib) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["td", "spike", str(SG), "--account", PHONE])
    assert result.exit_code == 1
    assert "run `tgi accounts add`" in result.stderr
    result = runner.invoke(app, ["td", "spike", str(SG), "--account", "34600"])
    assert result.exit_code == 1 and "the phone must look like" in result.stderr
    assert fake_tdlib.sent == []


def test_cli_login_flood_wait_is_reported_without_details(fake_tdlib: FakeLib) -> None:
    fake_tdlib.flood["setAuthenticationPhoneNumber"] = 30
    result = CliRunner().invoke(app, ["accounts", "add", "--phone", PHONE])
    assert result.exit_code == 3
    assert result.stderr == "login failed: FloodWait 429, retry after 30s\n"


def test_cli_spike_reports_a_tdlib_error_by_code(fake_tdlib: FakeLib) -> None:
    fake_tdlib.logged_in = True
    runner = CliRunner()
    assert runner.invoke(app, ["accounts", "add", "--phone", PHONE]).exit_code == 0
    result = runner.invoke(app, ["td", "spike", "-5", "--account", PHONE])
    assert result.exit_code == 3 and result.stderr == "spike failed: TdError 400\n"
    assert len(fake_tdlib.requests("close")) == 2  # closed after the failure too


def test_sync_commands_open_the_real_adapter(fake_tdlib: FakeLib, db_url: str,
                                             monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    result = runner.invoke(app, ["chats", "list", "--account", PHONE])
    assert result.exit_code == 1 and "run `tgi accounts add`" in result.stderr
    fake_tdlib.logged_in = True
    assert runner.invoke(app, ["accounts", "add", "--phone", PHONE]).exit_code == 0
    result = runner.invoke(app, ["chats", "list", "--account", PHONE])
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        f"  {SG}\tsupergroup\tchat {SG}",
        "  42\tprivate\tchat 42",
        f"  {CH}\tchannel\tchat {CH}",
    ]
    result = runner.invoke(app, ["chats", "enable", str(SG), "--account", PHONE])
    assert result.exit_code == 0 and result.stdout == f"enabled {SG}\n"
    result = runner.invoke(app, ["sync"])
    assert result.exit_code == 0, result.output
    assert " ok " in result.stdout and "fetched=" in result.stdout
    assert len(fake_tdlib.requests("close")) == 4  # every command closed its client


class _Failing:
    def close(self) -> None:
        raise TdError(408, f"{TEXT} {PHONE}")


def test_clients_close_reports_failures_by_type_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(td, "open_client", lambda settings, phone: _Failing())
    clients = cli._Clients(None)
    clients(PHONE)
    clients("+34600000998")
    clients.close()
    err = capsys.readouterr().err
    assert err == "closing a TDLib client failed: TdError 408\n" * 2
    clients.close()  # already closed: nothing left to close
    assert capsys.readouterr().err == ""


def test_tracebacks_show_no_locals() -> None:
    assert app.pretty_exceptions_show_locals is False


class _Refusing:
    """A client whose every request fails with a payload that must not be printed."""

    def get_chats(self) -> list[Chat]:
        raise TdError(400, f"{TEXT} {PHONE} {SECRETS['TG_API_HASH']}")

    def get_chat(self, chat_id: int) -> Chat:
        raise FloodWait(17)

    def close(self) -> None:
        pass


def test_chats_commands_report_tdlib_errors_by_type_and_code(
    fake_tdlib: FakeLib, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(td, "open_client", lambda settings, phone: _Refusing())
    runner = CliRunner()
    listed = runner.invoke(app, ["chats", "list", "--account", PHONE])
    assert (listed.exit_code, listed.stdout) == (3, "")
    assert listed.stderr == "chats list failed: TdError 400\n"
    enabled = runner.invoke(app, ["chats", "enable", str(SG), "--account", PHONE])
    assert (enabled.exit_code, enabled.stdout) == (3, "")
    assert enabled.stderr == "chats enable failed: FloodWait 429\n"
    outputs = listed.output + enabled.output + caplog.text
    assert [f for f in FORBIDDEN if f in outputs] == []


def test_a_group_or_world_readable_env_file_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    monkeypatch.setattr(cli, "ENV_FILE", env)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runner = CliRunner()

    def stderr() -> list[str]:
        result = runner.invoke(app, ["chats", "disable", "5"])  # any command; this one exits 1
        assert result.exit_code == 1
        return [line for line in result.stderr.splitlines() if "warning" in line]

    assert stderr() == []  # no .env at all
    env.write_text(f"TDLIB_DB_KEY={SECRETS['TDLIB_DB_KEY']}\n")
    env.chmod(0o600)
    assert stderr() == []
    for mode in (0o640, 0o604, 0o644, 0o660, 0o440):
        env.chmod(mode)
        assert stderr() == [f"warning: {env} is mode {mode:04o}, more open than 0600; chmod 600 it"]
    assert (
        env.read_text()
        and SECRETS["TDLIB_DB_KEY"] not in runner.invoke(app, ["chats", "disable", "5"]).output
    )


def test_env_file_warning_looks_at_the_mode_only(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    assert env_file_warning(env) is None
    env.write_text("")
    env.chmod(0o400)
    assert env_file_warning(env) is None  # stricter than 0600 is fine
    env.chmod(0o601)
    assert env_file_warning(env) == (
        f"warning: {env} is mode 0601, more open than 0600; chmod 600 it"
    )


class _NoLogFile(FakeLib):
    """TDLib refusing one log setting, with a reply that quotes a path and a payload."""

    def __init__(self, refuse: str) -> None:
        super().__init__(logged_in=True)
        self.refuse = refuse

    def td_execute(self, request: bytes) -> bytes | None:
        if json.loads(request)["@type"] == self.refuse:
            err = {"@type": "error", "code": 400, "message": f"Can't open {TEXT} {PHONE}"}
            return json.dumps(err).encode()
        return super().td_execute(request)


@pytest.mark.parametrize("refuse", ["setLogStream", "setLogVerbosityLevel"])
def test_a_refused_log_setting_stops_before_any_client(
    fake_tdlib: FakeLib, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    refuse: str,
) -> None:  # fmt: skip
    caplog.set_level(logging.DEBUG)
    lib = _NoLogFile(refuse)
    monkeypatch.setattr(td, "load_tdjson", lambda path: lib)
    result = CliRunner().invoke(app, ["accounts", "add", "--phone", PHONE])
    assert result.exit_code == 1, result.output
    assert refuse in result.stderr and "(TDLib 400)" in result.stderr
    assert "Can't open" not in result.output
    assert [f for f in FORBIDDEN if f in result.output + caplog.text] == []
    assert lib.clients == 0 and lib.sent == []  # no client was created
    assert td._hub is None  # the half-set hub is not kept for a later call


@pytest.mark.parametrize("first", ["getChat", "getChatHistory"])
def test_a_cold_client_loads_the_chat_lists_once(tmp_path: Path, first: str) -> None:
    lib = FakeLib(logged_in=True, cold=True)
    c = client(tmp_path, lib)
    if first == "getChat":
        assert c.get_chat(SG).kind == "supergroup"
    else:
        assert len(c.get_chat_history(SG, 0, 100)) == 2
    assert len(lib.requests("loadChats")) == 4  # main and archive, each until its 404
    assert c.get_chat(CH).kind == "channel"
    assert len(c.get_chat_history(SG, 0, 100)) == 2
    with pytest.raises(TdError, match="Chat not found"):  # not in any list: no reload
        c.get_chat(-5)
    assert len(lib.requests("loadChats")) == 4
    c.close()


@pytest.mark.parametrize(("link", "parsed"), [
    ("https://t.me/valenforum", ("username", "valenforum")),
    ("t.me/valenforum/173867", ("username", "valenforum")),
    ("https://t.me/s/NewsValencia", ("username", "NewsValencia")),
    ("@valenforum", ("username", "valenforum")),
    (" valenforum ", ("username", "valenforum")),
    ("https://telegram.me/valenforum?start=1", ("username", "valenforum")),
    ("https://t.me/+AbCdEf_12-xyz", ("invite", "https://t.me/+AbCdEf_12-xyz")),
    ("t.me/joinchat/AbCdEf_12-xyz", ("invite", "https://t.me/joinchat/AbCdEf_12-xyz")),
    ("-1001831270625", ("chat_id", "-1001831270625")),
    ("https://t.me/c/1831270625/173867", ("chat_id", "-1001831270625")),
])  # fmt: skip
def test_parse_link(link: str, parsed: tuple[str, str]) -> None:
    assert parse_link(link) == parsed


@pytest.mark.parametrize("link", [
    "", "https://example.com/valenforum", "https://t.me/+34600000999", "https://t.me/+",
    "@ab", "two words", "https://t.me/joinchat/", "javascript:alert(1)",
    "https://t.me/addstickers/pack", "https://t.me/c/abc/1",
])  # fmt: skip
def test_parse_link_rejects(link: str) -> None:
    with pytest.raises(ValueError, match="not a chat link"):
        parse_link(link)


def test_join_chat_by_username_invite_and_id(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    kinds = ("supergroup", "channel")
    assert c.join_chat(("username", "avisos"), kinds) == Chat(CH, f"chat {CH}", "channel", "avisos")
    assert c.join_chat(("invite", "https://t.me/+sgInvite12"), kinds).chat_id == SG
    assert c.join_chat(("invite", "https://t.me/+memberAlready"), kinds).chat_id == SG
    assert c.join_chat(("chat_id", str(SG)), kinds).kind == "supergroup"
    assert [q["chat_id"] for q in lib.requests("joinChat")] == [CH, SG]
    assert len(lib.requests("joinChatByInviteLink")) == 2
    c.close()


def test_join_chat_refuses_before_joining(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    kinds = ("supergroup", "channel")
    with pytest.raises(NotJoinable) as private:
        c.join_chat(("username", "ana_person"), kinds)
    with pytest.raises(NotJoinable) as basic:
        c.join_chat(("invite", "https://t.me/+basicInvite1"), kinds)
    assert (private.value.kind, basic.value.kind) == ("private", "basic_group")
    with pytest.raises(JoinPending):
        c.join_chat(("invite", "https://t.me/+askAdmins12"), kinds)
    with pytest.raises(TdError, match="USERNAME_NOT_OCCUPIED"):
        c.join_chat(("username", "nobody_here"), kinds)
    assert lib.requests("joinChat") == []
    assert len(lib.requests("joinChatByInviteLink")) == 1  # only the join request
    c.close()


def test_before_join_runs_before_any_join(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    kinds = ("supergroup", "channel")
    seen: list[int] = []

    def refuse(chat_id: int) -> None:
        seen.append(chat_id)
        raise RuntimeError("held by another account")

    for link in [("username", "avisos"), ("chat_id", str(SG)),
                 ("invite", "https://t.me/+memberAlready")]:  # fmt: skip
        with pytest.raises(RuntimeError):
            c.join_chat(link, kinds, refuse)
    assert seen == [CH, SG, SG]
    assert lib.requests("joinChat") == [] and lib.requests("joinChatByInviteLink") == []
    # An invite to a chat the account cannot see yet has no id before the join.
    assert c.join_chat(("invite", "https://t.me/+sgInvite12"), kinds, refuse).chat_id == SG
    assert seen == [CH, SG, SG]
    c.close()


def test_join_request_then_approval_then_retry_joins(tmp_path: Path) -> None:
    """#39: `creates_join_request` stays on the link after approval; the reply decides."""
    lib = FakeLib(logged_in=True)
    c = client(tmp_path, lib)
    kinds = ("supergroup", "channel")
    link = "https://t.me/+askAdmins12"
    with pytest.raises(JoinPending):
        c.join_chat(("invite", link), kinds)  # the request is sent
    with pytest.raises(JoinPending):
        c.join_chat(("invite", link), kinds)  # not approved yet: still pending
    lib.approved.add(link)  # the admins approve: the account is a member
    assert c.join_chat(("invite", link), kinds) == Chat(CH, f"chat {CH}", "channel", "avisos")
    assert len(lib.requests("joinChatByInviteLink")) == 3
    c.close()


def test_join_request_link_that_joins_returns_the_chat(tmp_path: Path) -> None:
    """A join-request link whose join succeeds (e.g. the request was approved and the
    account then left and rejoined) returns the chat instead of `JoinPending`."""

    class Joins(FakeLib):
        def joinChatByInviteLink(self, client_id: int, q: Json) -> Json:  # noqa: N802
            return self.getChat(client_id, {"chat_id": CH})

    lib = Joins(logged_in=True)
    c = client(tmp_path, lib)
    chat = c.join_chat(("invite", "https://t.me/+askAdmins12"), ("supergroup", "channel"))
    assert chat.chat_id == CH
    c.close()


def test_public_chat_that_sends_a_join_request_is_pending(tmp_path: Path) -> None:
    lib = FakeLib(logged_in=True)
    lib.requests_to_join.add(CH)
    c = client(tmp_path, lib)
    with pytest.raises(JoinPending):
        c.join_chat(("username", "avisos"), ("supergroup", "channel"))
    with pytest.raises(JoinPending):
        c.join_chat(("chat_id", str(CH)), ("supergroup", "channel"))
    assert [q["chat_id"] for q in lib.requests("joinChat")] == [CH, CH]
    c.close()


def test_get_chats_stops_at_its_safety_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Endless(FakeLib):
        def loadChats(self, client_id: int, query: Json) -> Json:  # noqa: N802  # never 404
            return {"@type": "ok"}

    monkeypatch.setattr(td, "LOAD_CHATS_MAX_CALLS", 5)
    lib = Endless(logged_in=True)
    c = client(tmp_path, lib)
    with pytest.raises(td.Unfinished) as caught:
        c.get_chats()
    assert caught.value.code == 0 and len(lib.requests("loadChats")) == 5
    c.close()
