"""`FakeTdClient`: a scenario-driven stand-in for TDLib in tests (SPEC §8).

A scenario is JSON:

    {
      "from_inclusive": false,
      "chats": [{"chat_id": -1001, "title": "t", "kind": "supergroup", "username": null,
                 "page_sizes": [100, 37, 1],
                 "messages": [{"id": 1, "date": "2026-01-01T10:00:00+00:00", "text": "hi",
                               "sender_id": 7, "sender_name": "A", "reply_to": null,
                               "has_media": false}]}],
      "faults": [{"call": 2, "flood_wait": 2}, {"call": 5, "error": "boom"}],
      "runs": [{"add": [{"chat_id": -1001, "id": 2, "date": "...", "text": "new"}],
                "edit": [{"chat_id": -1001, "id": 1, "text": "hi!", "edit_date": "..."}],
                "delete": [{"chat_id": -1001, "id": 1}],
                "faults": []}]
    }

- `id` and `reply_to` are server ids; the fake exposes `message_id = id << 20`. A message
  may give a raw `message_id` instead of `id` to model non-server (local) TDLib ids.
- `page_sizes`: sizes of successive history pages for that chat (each capped at `limit`);
  once used up, pages are `limit` long. An empty page is returned only at the end.
- `faults`: the n-th `get_chat_history` call of the current run (1-based, counting every
  attempt) raises `FloodWait(flood_wait)` or `TdError(500, error)`, or with `"empty": true`
  returns a spurious empty page. "Raise after page N" is a fault at call N + 1.
- `runs`: changes applied by `next_run()` between sync runs; each run resets the call
  counter and page sequences and installs that run's `faults`.
- `invites`: `[{"link": "https://t.me/+hash", "chat_id": -1001, "request": true}]`, links
  `join_chat` accepts; `"request": true`, or `"join_request": true` on a chat for its
  username and id, makes the join a pending request (`JoinPending`).
- `from_inclusive`: whether a page starts at `from_message_id` itself (true) or strictly
  below it (false). Real TDLib behaviour is confirmed by the T3 spike; ingestion must cope
  with both.
"""

import json
from collections.abc import Callable, Collection
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from tgi.td_port import (
    SERVER_ID_SHIFT,
    Chat,
    ChatKind,
    ChatLink,
    FloodWait,
    JoinPending,
    Message,
    NotJoinable,
    TdError,
)


def _date(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"scenario date without a timezone: {value}")
    return parsed


def _message(chat_id: int, raw: dict[str, Any]) -> Message:
    reply_to = raw.get("reply_to")
    edit_date = raw.get("edit_date")
    return Message(
        chat_id=chat_id,
        message_id=int(raw["message_id"])
        if "message_id" in raw
        else int(raw["id"]) << SERVER_ID_SHIFT,
        date=_date(raw["date"]),
        text=raw.get("text", ""),
        sender_id=raw.get("sender_id"),
        sender_name=raw.get("sender_name"),
        edit_date=_date(edit_date) if edit_date else None,
        reply_to_message_id=int(reply_to) << SERVER_ID_SHIFT if reply_to else None,
        has_media=bool(raw.get("has_media", False)),
    )


class FakeTdClient:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self._from_inclusive = bool(scenario.get("from_inclusive", False))
        self._chats: dict[int, Chat] = {}
        self._messages: dict[int, dict[int, Message]] = {}
        self._page_sizes: dict[int, list[int]] = {}
        self._join_requests: set[int] = set()  # public chats whose join sends a request
        for raw in scenario["chats"]:
            chat_id = int(raw["chat_id"])
            self._chats[chat_id] = Chat(
                chat_id=chat_id,
                title=raw["title"],
                kind=cast(ChatKind, raw["kind"]),
                username=raw.get("username"),
            )
            if raw.get("join_request"):
                self._join_requests.add(chat_id)
            self._page_sizes[chat_id] = [int(n) for n in raw.get("page_sizes", [])]
            if any(n < 1 for n in self._page_sizes[chat_id]):
                raise ValueError("page_sizes must be >= 1; an empty page means end of history")
            self._messages[chat_id] = {}
            for m in raw.get("messages", []):
                message = _message(chat_id, m)
                self._messages[chat_id][message.message_id] = message
        # Chats the account can join but is not in yet: invite link -> chat (join request
        # if "request": true); public ones are found by username among all chats.
        self._invites: dict[str, tuple[int, bool]] = {
            str(i["link"]): (int(i["chat_id"]), bool(i.get("request", False)))
            for i in scenario.get("invites", [])
        }
        self.joined: list[int] = []
        self._runs: list[dict[str, Any]] = list(scenario.get("runs", []))
        self._closed = False
        self.history_calls: list[tuple[int, int, int]] = []
        self._start_run(scenario.get("faults", []))

    @classmethod
    def from_file(cls, path: Path) -> "FakeTdClient":
        return cls(json.loads(path.read_text()))

    def _start_run(self, faults: list[dict[str, Any]]) -> None:
        self._calls = 0
        self._faults = {int(f["call"]): f for f in faults}
        self._pending_sizes = {cid: list(sizes) for cid, sizes in self._page_sizes.items()}

    def next_run(self) -> None:
        """Apply the next scripted run's additions, edits and deletions."""
        if not self._runs:
            raise IndexError("scenario has no more runs")
        run = self._runs.pop(0)
        for raw in run.get("add", []):
            message = _message(int(raw["chat_id"]), raw)
            self._messages[message.chat_id][message.message_id] = message
        for raw in run.get("edit", []):
            chat_msgs = self._messages[int(raw["chat_id"])]
            message_id = int(raw["id"]) << SERVER_ID_SHIFT
            chat_msgs[message_id] = replace(
                chat_msgs[message_id], text=raw["text"], edit_date=_date(raw["edit_date"])
            )
        for raw in run.get("delete", []):
            del self._messages[int(raw["chat_id"])][int(raw["id"]) << SERVER_ID_SHIFT]
        self._start_run(run.get("faults", []))

    def _check_open(self) -> None:
        if self._closed:
            raise TdError(500, "client is closed")

    def get_chats(self) -> list[Chat]:
        self._check_open()
        return list(self._chats.values())

    def get_chat(self, chat_id: int) -> Chat:
        self._check_open()
        if chat_id not in self._chats:
            raise TdError(400, "Chat not found")
        return self._chats[chat_id]

    def join_chat(
        self,
        link: ChatLink,
        kinds: Collection[ChatKind],
        before_join: Callable[[int], None] | None = None,
    ) -> Chat:
        self._check_open()
        how, value = link
        if how == "invite":
            if value not in self._invites:
                raise TdError(400, "INVITE_HASH_INVALID")
            chat_id, request = self._invites[value]
        elif how == "username":
            found = [c for c in self._chats.values() if c.username == value]
            if not found:
                raise TdError(400, "USERNAME_NOT_OCCUPIED")
            chat_id = found[0].chat_id
            request = chat_id in self._join_requests
        else:
            chat_id = int(value)
            request = chat_id in self._join_requests
        chat = self.get_chat(chat_id)
        if chat.kind not in kinds:
            raise NotJoinable(chat.kind)
        if before_join is not None:
            before_join(chat_id)
        if request:
            raise JoinPending("the link sends a join request")
        self.joined.append(chat_id)
        return chat

    def get_chat_history(self, chat_id: int, from_message_id: int, limit: int) -> list[Message]:
        self._check_open()
        self._calls += 1
        self.history_calls.append((chat_id, from_message_id, limit))
        fault = self._faults.get(self._calls)
        if fault is not None:
            if fault.get("empty"):
                return []
            if "flood_wait" in fault:
                raise FloodWait(int(fault["flood_wait"]))
            raise TdError(500, str(fault["error"]))
        if chat_id not in self._chats:
            raise TdError(400, "Chat not found")
        if not 0 < limit <= 100:
            raise TdError(400, "Parameter limit must be in [1, 100]")

        def in_page(message_id: int) -> bool:
            if from_message_id == 0:
                return True
            if self._from_inclusive:
                return message_id <= from_message_id
            return message_id < from_message_id

        older = sorted((m for m in self._messages[chat_id] if in_page(m)), reverse=True)
        if not older:
            return []
        sizes = self._pending_sizes[chat_id]
        size = min(limit, sizes.pop(0)) if sizes else limit
        return [self._messages[chat_id][m] for m in older[:size]]

    def close(self) -> None:
        self._closed = True
