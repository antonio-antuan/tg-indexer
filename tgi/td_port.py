"""`TdClient` port: everything that touches Telegram sits behind it (SPEC §2).

Ids are TDLib-shaped: a server message has `message_id = server_id << 20`. History is
returned newest first, like TDLib's `getChatHistory`, and a page may hold fewer than
`limit` messages (even one); only an empty page means the end of history (SPEC §7).
"""

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

ChatKind = Literal["supergroup", "channel", "basic_group", "private"]

SERVER_ID_SHIFT = 20


def server_message_id(message_id: int) -> int:
    """`server_id` from a TDLib message id; rejects ids that are not server messages."""
    if message_id <= 0 or message_id % (1 << SERVER_ID_SHIFT):
        raise ValueError(f"not a server message id: {message_id}")
    return message_id >> SERVER_ID_SHIFT


@dataclass(frozen=True)
class Chat:
    chat_id: int
    title: str
    kind: ChatKind
    username: str | None = None


@dataclass(frozen=True)
class Message:
    chat_id: int
    message_id: int
    date: datetime
    text: str
    """Message text or media caption; '' for service messages and media without caption."""
    sender_id: int | None = None
    sender_name: str | None = None
    edit_date: datetime | None = None
    reply_to_message_id: int | None = None
    has_media: bool = False


class TdError(Exception):
    """A TDLib request failed."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


class FloodWait(TdError):
    """TDLib error 429; the caller waits `retry_after` seconds before retrying."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(429, f"Too Many Requests: retry after {retry_after}")
        self.retry_after = retry_after


class NotJoinable(Exception):
    """The link leads to a chat of a kind the caller does not accept; nothing was joined."""

    def __init__(self, kind: ChatKind) -> None:
        super().__init__(f"a {kind} chat")
        self.kind = kind


class JoinPending(Exception):
    """The link sends a join request: the chat's admins must approve it first."""


ChatLink = tuple[Literal["username", "invite", "chat_id"], str]

_USERNAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,31}")
_INVITE_HASH = re.compile(r"[A-Za-z0-9_-]{8,}")
# t.me paths that are not usernames.
_RESERVED = {"joinchat", "c", "s", "addstickers", "addemoji", "addlist", "share", "proxy",
             "socks", "login", "setlanguage", "bg", "addtheme", "invoice", "boost"}  # fmt: skip
_TME = re.compile(
    r"(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me|telegram\.dog)/([^?#]+)(?:[?#].*)?", re.I
)


def parse_link(link: str) -> ChatLink:
    """A chat reference: `https://t.me/name` (also `t.me/name/123`, `t.me/s/name`), `@name`
    or `name` is a public username; `https://t.me/+hash` or `t.me/joinchat/hash` an invite
    link (returned as a canonical t.me URL); `-100…` or a message link `t.me/c/<id>/<n>` a
    chat id. ValueError otherwise."""
    text = link.strip()
    if re.fullmatch(r"-?\d+", text):
        return "chat_id", text
    if m := _TME.fullmatch(text):
        first, *rest = m.group(1).strip("/").split("/")
        if first.startswith("+"):
            if _INVITE_HASH.fullmatch(first[1:]) and not first[1:].isdigit():  # +34… is a phone
                return "invite", f"https://t.me/{first}"
            raise ValueError("not a chat link")
        head = first.lower()
        if head == "joinchat" and rest and _INVITE_HASH.fullmatch(rest[0]):
            return "invite", f"https://t.me/joinchat/{rest[0]}"
        if head == "c" and rest and rest[0].isdigit():  # a message link in a private group
            return "chat_id", f"-100{rest[0]}"
        if head == "s" and rest:
            text = rest[0]
        elif head in _RESERVED:
            raise ValueError("not a chat link")
        else:
            text = first
    elif text.startswith("@"):
        text = text[1:]
    if _USERNAME.fullmatch(text):
        return "username", text
    raise ValueError("not a chat link")


class TdClient(Protocol):
    def get_chats(self) -> list[Chat]: ...

    def get_chat(self, chat_id: int) -> Chat: ...

    def get_chat_history(self, chat_id: int, from_message_id: int, limit: int) -> list[Message]:
        """Messages older than `from_message_id` (0 = from the newest), newest first."""
        ...

    def join_chat(
        self,
        link: ChatLink,
        kinds: Collection[ChatKind],
        before_join: Callable[[int], None] | None = None,
    ) -> Chat:
        """Join the chat behind `link` unless the account is a member already. A chat whose
        kind is not in `kinds` raises `NotJoinable` before any join; a link that sends a
        join request raises `JoinPending`. `before_join(chat_id)` runs before the join and may
        raise to stop it; an invite link to a chat the account cannot see yet has no id
        before the join, so it is not called."""
        ...

    def close(self) -> None: ...
