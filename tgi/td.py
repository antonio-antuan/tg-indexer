"""The real `TdClient`: TDLib's JSON interface over `libtdjson` (SPEC §3, §5.1).

`libtdjson.so` comes from `scripts/tdlib.sh` (`TDJSON_PATH`, default
`DATA_DIR/lib/libtdjson.so`). One `Hub` per process wraps it: `td_receive` returns events
for every client of the process, so the hub queues them per `@client_id`. A request
carries a numeric `@extra` and its answer is the event with the same `@extra`; updates
seen meanwhile only feed the authorization state.

An account is `DATA_DIR/accounts/<phone>/` (mode 0700); TDLib keeps its session in `td/`,
encrypted with `TDLIB_DB_KEY`, without a message database (Postgres is the store).
`tgi accounts add` logs in interactively through a `Prompt`; `sync` and `chats` open a
logged-in account only and never prompt (`NotLoggedIn`).

Nothing here logs or raises with message text, phone numbers, codes, passwords, keys or
TDLib payloads: errors carry TDLib's code and its short error message only.
"""

import base64
import contextlib
import ctypes
import json
import logging
import re
import time
from collections import deque
from collections.abc import Callable, Collection
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from tgi import __version__
from tgi.config import Settings
from tgi.td_port import (
    Chat,
    ChatKind,
    ChatLink,
    FloodWait,
    JoinPending,
    Message,
    NotJoinable,
    TdError,
)

log = logging.getLogger(__name__)

Json = dict[str, Any]

PHONE = re.compile(r"\+[1-9][0-9]{6,14}")
TIMEOUT_S = 60.0
CODE_ATTEMPTS = 3
TDLIB_LOG_MAX_BYTES = 10 << 20
INVITE_KINDS: dict[str, ChatKind] = {
    "inviteLinkChatTypeSupergroup": "supergroup",
    "inviteLinkChatTypeChannel": "channel",
    "inviteLinkChatTypeBasicGroup": "basic_group",
}
LOAD_CHATS_MAX_CALLS = 1000  # 100 chats per call: far more than any account has
# Media whose caption is the message text (SPEC §4 `has_media`); other contents give ''.
CAPTIONED = {
    "messageAnimation", "messageAudio", "messageDocument", "messagePhoto", "messageVideo",
    "messageVoiceNote",
}  # fmt: skip
MEDIA = CAPTIONED | {"messageSticker", "messageVideoNote"}
UNSUPPORTED_LOGIN = {
    "authorizationStateWaitEmailAddress": "an email address",
    "authorizationStateWaitEmailCode": "an email code",
    "authorizationStateWaitOtherDeviceConfirmation": "a QR confirmation",
    "authorizationStateWaitRegistration": "registration (sign the account up in an app first)",
    "authorizationStateWaitPremiumPurchase": "a Premium purchase",
}


class SetupError(Exception):
    """The account cannot be opened as configured; the message is safe to print."""


class Unfinished(TdError):
    """A TDLib loop hit its safety cap without TDLib saying it was done."""

    def __init__(self, request: str) -> None:
        super().__init__(0, f"{request} did not finish")


class NotLoggedIn(SetupError):
    """The account has no authorized TDLib session; `tgi accounts add` creates one."""


class TdJson(Protocol):
    """The four `libtdjson` functions (a `ctypes.CDLL`, or a fake in tests)."""

    def td_create_client_id(self) -> int: ...

    def td_send(self, client_id: int, request: bytes) -> None: ...

    def td_receive(self, timeout: float) -> bytes | None: ...

    def td_execute(self, request: bytes) -> bytes | None: ...


def load_tdjson(path: Path) -> TdJson:
    lib = ctypes.CDLL(str(path))
    lib.td_create_client_id.restype = ctypes.c_int
    lib.td_create_client_id.argtypes = []
    lib.td_send.restype = None
    lib.td_send.argtypes = [ctypes.c_int, ctypes.c_char_p]
    lib.td_receive.restype = ctypes.c_char_p
    lib.td_receive.argtypes = [ctypes.c_double]
    lib.td_execute.restype = ctypes.c_char_p
    lib.td_execute.argtypes = [ctypes.c_char_p]
    return cast(TdJson, lib)


class Hub:
    """One `libtdjson` per process: sends, synchronous executes, and routed receives."""

    def __init__(self, lib: TdJson) -> None:
        self._lib = lib
        self._queued: dict[int, deque[Json]] = {}

    def execute(self, request: Json) -> Json:
        raw = self._lib.td_execute(json.dumps(request).encode())
        return cast(Json, json.loads(raw)) if raw else {"@type": "error", "code": 0}

    def version(self) -> str:
        reply = self.execute({"@type": "getOption", "name": "version"})
        return str(reply.get("value", "unknown"))

    def log_to(self, path: Path, verbosity: int = 1) -> None:
        """Keep TDLib's own log off stderr (A10): errors only, to a capped file.

        Fails closed: if TDLib refuses either setting it would keep logging to stderr.
        """
        for request in (
            {"@type": "setLogVerbosityLevel", "new_verbosity_level": verbosity},
            {"@type": "setLogStream", "log_stream": {
                "@type": "logStreamFile", "path": str(path),
                "max_file_size": TDLIB_LOG_MAX_BYTES, "redirect_stderr": False}},
        ):  # fmt: skip
            reply = self.execute(request)
            if reply.get("@type") != "ok":  # the reply's message may quote the path
                raise SetupError(
                    f"TDLib refused {request['@type']} (TDLib {reply.get('code', 0)}):"
                    " check that DATA_DIR/accounts/tdlib.log can be written"
                )

    def create_client(self) -> int:
        client_id = self._lib.td_create_client_id()
        self._queued[client_id] = deque()
        return client_id

    def send(self, client_id: int, request: Json) -> None:
        self._lib.td_send(client_id, json.dumps(request).encode())

    def receive(self, client_id: int, timeout: float) -> Json | None:
        """The next event for `client_id`, queueing other clients' events on the way."""
        deadline = time.monotonic() + timeout
        while True:
            if self._queued[client_id]:
                return self._queued[client_id].popleft()
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            raw = self._lib.td_receive(min(left, 1.0))
            if not raw:
                continue
            event = cast(Json, json.loads(raw))
            owner = event.get("@client_id")
            if owner in self._queued:
                self._queued[owner].append(event)


_hub: Hub | None = None


def hub(settings: Settings) -> Hub:
    """The process's hub, loading `libtdjson` on first use."""
    global _hub
    if _hub is None:
        path = tdjson_path(settings)
        if not path.is_file():
            raise SetupError("libtdjson not found: run scripts/tdlib.sh or set TDJSON_PATH")
        try:
            lib = load_tdjson(path)
        except OSError:
            raise SetupError("libtdjson does not load: run scripts/tdlib.sh again") from None
        accounts = _data_dir(settings) / "accounts"
        _private_dir(accounts)
        new = Hub(lib)
        new.log_to(accounts / "tdlib.log")  # TDLib's errors may name an account: keep it private
        _hub = new  # only once logging is set: a failed start is retried, never reused
    return _hub


def tdjson_path(settings: Settings) -> Path:
    return settings.tdjson_path or _data_dir(settings) / "lib" / "libtdjson.so"


def _data_dir(settings: Settings) -> Path:
    if settings.data_dir is None:
        raise SetupError("DATA_DIR is not set")
    return settings.data_dir


def account_dir(settings: Settings, phone: str) -> Path:
    if not PHONE.fullmatch(phone):
        raise SetupError("the phone must look like +34600000000 (digits after '+')")
    return _data_dir(settings) / "accounts" / phone


def logged_in_accounts(settings: Settings) -> list[str]:
    """Phones with a TDLib session under DATA_DIR/accounts, sorted."""
    root = _data_dir(settings) / "accounts"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if PHONE.fullmatch(d.name) and (d / "td").is_dir())


def _private_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def tdlib_parameters(settings: Settings, account: Path) -> Json:
    required = {"TDLIB_DB_KEY": settings.tdlib_db_key, "TG_API_ID": settings.tg_api_id,
                "TG_API_HASH": settings.tg_api_hash}  # fmt: skip
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise SetupError(f"{', '.join(missing)} not set")
    try:
        api_id = int(cast(str, settings.tg_api_id))
    except ValueError:
        raise SetupError("TG_API_ID must be an integer") from None
    td_dir = account / "td"
    return {
        "@type": "setTdlibParameters",
        "use_test_dc": False,
        "database_directory": str(td_dir),
        "files_directory": str(td_dir / "files"),
        "database_encryption_key": base64.b64encode(
            cast(str, settings.tdlib_db_key).encode()
        ).decode(),
        "use_file_database": False,
        "use_chat_info_database": True,
        "use_message_database": False,  # TDLib is the transport, Postgres the store
        "use_secret_chats": False,
        "api_id": api_id,
        "api_hash": settings.tg_api_hash,
        "system_language_code": "en",
        "device_model": "tgi",
        "system_version": "",
        "application_version": __version__,
    }


class Prompt(Protocol):
    """Interactive login input; implemented by the CLI over the terminal."""

    def code(self) -> str: ...

    def password(self, hint: str) -> str: ...

    def rejected(self, what: str, error: TdError) -> None: ...


def _error(event: Json) -> TdError:
    code, message = int(event.get("code", 0)), str(event.get("message", ""))
    if code == 429 and (m := re.search(r"retry after (\d+)", message)):
        return FloodWait(int(m.group(1)))
    return TdError(code, message)


class TdlibClient:
    """`TdClient` over one TDLib client (one account)."""

    def __init__(
        self,
        hub: Hub,
        parameters: Json,
        phone: str,
        prompt: Prompt | None = None,
        timeout: float = TIMEOUT_S,
    ) -> None:
        self._hub, self._timeout = hub, timeout
        self._client_id = hub.create_client()
        self._extra = 0
        self._auth: Json = {}
        self._auth_seen = 0
        self._closed = False
        self._names: dict[tuple[str, int], str | None] = {}
        self._lists_loaded = False
        try:
            self._authorize(parameters, phone, prompt)
        except BaseException:
            with contextlib.suppress(TdError):  # keep the login error, not a close timeout
                self.close()
            raise

    # --- requests -----------------------------------------------------------------------

    def request(self, query: Json) -> Json:
        """Send `query` and return its answer; a TDLib error raises `TdError`/`FloodWait`."""
        if self._closed:
            raise TdError(500, "client is closed")
        self._extra += 1
        extra = self._extra
        self._hub.send(self._client_id, {**query, "@extra": extra})
        deadline = time.monotonic() + self._timeout
        while True:
            event = self._next(deadline)
            if event is None:
                raise TdError(408, f"no answer to {query['@type']}")
            if event.get("@extra") == extra:
                if event.get("@type") == "error":
                    raise _error(event)
                return event

    def _next(self, deadline: float) -> Json | None:
        """The next event, keeping track of the authorization state."""
        left = deadline - time.monotonic()
        event = self._hub.receive(self._client_id, max(left, 0.0))
        if event is not None and event.get("@type") == "updateAuthorizationState":
            self._auth = cast(Json, event["authorization_state"])
            self._auth_seen += 1
        return event

    def _await_state(self, seen: int) -> str:
        """The authorization state after `seen` updates (waits for a newer one)."""
        deadline = time.monotonic() + self._timeout
        while self._auth_seen <= seen:
            if self._next(deadline) is None:
                raise TdError(408, "no authorization state update")
        return str(self._auth["@type"])

    # --- login --------------------------------------------------------------------------

    def _authorize(self, parameters: Json, phone: str, prompt: Prompt | None) -> None:
        # Any request activates the client; its first update is the authorization state.
        self.request({"@type": "getOption", "name": "version"})
        state = self._await_state(0)
        while state != "authorizationStateReady":
            seen = self._auth_seen
            if state == "authorizationStateWaitTdlibParameters":
                self.request(parameters)
            elif prompt is None:
                raise NotLoggedIn("the account is not logged in: run `tgi accounts add`")
            elif state == "authorizationStateWaitPhoneNumber":
                self.request({"@type": "setAuthenticationPhoneNumber", "phone_number": phone})
            elif state == "authorizationStateWaitCode":
                self._answer(prompt, "code", self._code_query)
            elif state == "authorizationStateWaitPassword":
                self._answer(prompt, "password", self._password_query)
            elif state in UNSUPPORTED_LOGIN:
                raise TdError(400, f"login needs {UNSUPPORTED_LOGIN[state]}: not supported")
            else:  # LoggingOut, Closing, Closed
                raise TdError(500, f"TDLib is in {state}")
            state = self._await_state(seen)
        log.info("td client %s ready", self._client_id)

    @staticmethod
    def _code_query(prompt: Prompt, auth: Json) -> Json:
        return {"@type": "checkAuthenticationCode", "code": prompt.code()}

    @staticmethod
    def _password_query(prompt: Prompt, auth: Json) -> Json:
        hint = str(auth.get("password_hint", ""))
        return {"@type": "checkAuthenticationPassword", "password": prompt.password(hint)}

    def _answer(self, prompt: Prompt, what: str, query: Callable[[Prompt, Json], Json]) -> None:
        """Send a code or password, asking again after a rejection (TDLib keeps its state)."""
        for attempt in range(1, CODE_ATTEMPTS + 1):
            try:
                self.request(query(prompt, self._auth))
                return
            except FloodWait:
                raise
            except TdError as exc:
                if exc.code != 400 or attempt == CODE_ATTEMPTS:
                    raise
                prompt.rejected(what, exc)

    # --- TdClient -----------------------------------------------------------------------

    def get_chats(self) -> list[Chat]:
        """The account's dialogs, main list then archive (TDLib loads them on request)."""
        return [self.get_chat(i) for i in self._load_chat_lists()]

    def _load_chat_lists(self) -> list[int]:
        """Load the main and archive lists; their chat ids, main list first."""
        ids: list[int] = []
        for chat_list in ("chatListMain", "chatListArchive"):
            lst = {"@type": chat_list}
            for _ in range(LOAD_CHATS_MAX_CALLS):
                try:
                    self.request({"@type": "loadChats", "chat_list": lst, "limit": 100})
                except TdError as exc:
                    if exc.code == 404:  # every chat of the list is loaded
                        break
                    raise
            else:
                raise Unfinished("loadChats")
            got = self.request({"@type": "getChats", "chat_list": lst, "limit": 100_000})
            ids += [i for i in cast(list[int], got["chat_ids"]) if i not in ids]
        self._lists_loaded = True
        return ids

    def _chat_request(self, query: Json) -> Json:
        """A request on one chat. Without a message database a new TDLib client knows no
        chat until the chat lists are loaded: on `Chat not found` load them once, retry."""
        try:
            return self.request(query)
        except TdError as exc:
            if self._lists_loaded or exc.code != 400 or "Chat not found" not in str(exc):
                raise
        self._load_chat_lists()
        return self.request(query)

    def get_chat(self, chat_id: int) -> Chat:
        return self._chat(self._chat_request({"@type": "getChat", "chat_id": chat_id}))

    def _chat(self, raw: Json) -> Chat:
        """A `Chat` from TDLib's `chat` object."""
        chat_id = int(raw["id"])
        kind_raw = cast(Json, raw["type"])
        kind: ChatKind
        username = None
        match kind_raw["@type"]:
            case "chatTypeSupergroup":
                kind = "channel" if kind_raw.get("is_channel") else "supergroup"
                group = self.request(
                    {"@type": "getSupergroup", "supergroup_id": kind_raw["supergroup_id"]}
                )
                active = cast(Json, group.get("usernames") or {}).get("active_usernames") or []
                username = str(active[0]) if active else None
            case "chatTypeBasicGroup":
                kind = "basic_group"
            case _:  # chatTypePrivate, chatTypeSecret
                kind = "private"
        return Chat(chat_id=chat_id, title=str(raw.get("title", "")), kind=kind,
                    username=username)  # fmt: skip

    def join_chat(
        self,
        link: ChatLink,
        kinds: Collection[ChatKind],
        before_join: Callable[[int], None] | None = None,
    ) -> Chat:
        how, value = link
        if how == "invite":
            return self._join_invite(value, kinds, before_join)
        if how == "username":
            chat = self._chat(self.request({"@type": "searchPublicChat", "username": value}))
        else:
            chat = self.get_chat(int(value))
        if chat.kind not in kinds:
            raise NotJoinable(chat.kind)
        if before_join is not None:
            before_join(chat.chat_id)
        try:
            self.request({"@type": "joinChat", "chat_id": chat.chat_id})  # a member: no-op
        except TdError as exc:
            if "INVITE_REQUEST_SENT" not in str(exc):
                raise
            raise JoinPending("the chat sends a join request") from None
        return chat

    def _join_invite(
        self,
        link: str,
        kinds: Collection[ChatKind],
        before_join: Callable[[int], None] | None,
    ) -> Chat:
        info = self.request({"@type": "checkChatInviteLink", "invite_link": link})
        kind = INVITE_KINDS.get(str(cast(Json, info.get("type") or {}).get("@type")))
        if kind is None or kind not in kinds:
            raise NotJoinable(kind or "private")
        # 0 when the chat is hidden from this account until it joins.
        if before_join is not None and info.get("chat_id"):
            before_join(int(info["chat_id"]))
        # `creates_join_request` belongs to the link, not to this account: once a request is
        # approved the same link joins (or is already joined), so only the reply decides.
        try:
            raw = self.request({"@type": "joinChatByInviteLink", "invite_link": link})
        except TdError as exc:
            if "INVITE_REQUEST_SENT" in str(exc):
                raise JoinPending("the invite link sends a join request") from None
            # Already a member: the link's preview carries the chat id.
            if "USER_ALREADY_PARTICIPANT" not in str(exc) or not info.get("chat_id"):
                raise
            return self.get_chat(int(info["chat_id"]))
        return self._chat(raw)

    def get_chat_history(self, chat_id: int, from_message_id: int, limit: int) -> list[Message]:
        raw = self._chat_request({
            "@type": "getChatHistory", "chat_id": chat_id, "from_message_id": from_message_id,
            "offset": 0, "limit": limit, "only_local": False})  # fmt: skip
        return [self._message(m) for m in cast(list[Json | None], raw["messages"]) if m]

    def _message(self, raw: Json) -> Message:
        content = cast(Json, raw.get("content") or {})
        kind = str(content.get("@type", ""))
        if kind == "messageText":
            text = str(cast(Json, content["text"]).get("text", ""))
        elif kind in CAPTIONED and content.get("caption"):
            text = str(cast(Json, content["caption"]).get("text", ""))
        else:
            text = ""
        reply = cast(Json | None, raw.get("reply_to"))
        reply_to = None
        # A reply to a message in another chat has no reply_to here.
        if (reply and reply.get("@type") == "messageReplyToMessage"
                and reply.get("chat_id") in (0, raw["chat_id"])):  # fmt: skip
            reply_to = int(reply["message_id"]) or None
        sender_id, sender_name = self._sender(cast(Json | None, raw.get("sender_id")))
        edit = int(raw.get("edit_date") or 0)
        return Message(
            chat_id=int(raw["chat_id"]),
            message_id=int(raw["id"]),
            date=datetime.fromtimestamp(int(raw["date"]), UTC),
            text=text,
            sender_id=sender_id,
            sender_name=sender_name,
            edit_date=datetime.fromtimestamp(edit, UTC) if edit else None,
            reply_to_message_id=reply_to,
            has_media=kind in MEDIA,
        )

    def _sender(self, sender: Json | None) -> tuple[int | None, str | None]:
        """The sender's id and display name (cached; None when TDLib does not know it)."""
        if not sender:
            return None, None
        if sender.get("@type") == "messageSenderUser":
            key = ("user", int(sender["user_id"]))
        else:
            key = ("chat", int(sender["chat_id"]))
        if key not in self._names:
            try:
                if key[0] == "user":
                    user = self.request({"@type": "getUser", "user_id": key[1]})
                    name = f"{user.get('first_name', '')} {user.get('last_name', '')}".strip()
                else:
                    name = str(self.request({"@type": "getChat", "chat_id": key[1]})["title"])
                self._names[key] = name or None
            except FloodWait:
                raise
            except TdError:
                self._names[key] = None
        return key[1], self._names[key]

    def close(self) -> None:
        """Close the TDLib client and wait for `authorizationStateClosed`."""
        if self._closed:
            return
        self._closed = True
        self._hub.send(self._client_id, {"@type": "close"})
        deadline = time.monotonic() + self._timeout
        while self._auth.get("@type") != "authorizationStateClosed":
            if self._next(deadline) is None:
                raise TdError(408, "TDLib did not close")


def open_client(settings: Settings, phone: str, prompt: Prompt | None = None) -> TdlibClient:
    """The account's client. Without `prompt` the account must already be logged in."""
    account = account_dir(settings, phone)
    parameters = tdlib_parameters(settings, account)
    if prompt is None and not (account / "td").is_dir():
        raise NotLoggedIn("the account is not logged in: run `tgi accounts add`")
    _private_dir(account.parent)
    _private_dir(account)
    return TdlibClient(hub(settings), parameters, phone, prompt)
