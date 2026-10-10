"""T10, the P1 gate: the CLI end to end against the fake TDLib and a real Postgres.

Two `tgi sync` runs (a backfill, then a catch-up with an edit and a deletion), then
`tgi search` finds a Russian and a Spanish message with their links (A6, A11), and no log
line of the whole session carries message text, the phone or a secret (A10).
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

from tgi.cli import app

Conn = psycopg.Connection[tuple[object, ...]]
ACCOUNT = "+34600000777"
RU, ES = -1001500000001, -1001500000002
SECRETS = {  # sentinel values: none of them may reach a log line or the CLI output
    "TDLIB_DB_KEY": "tdlib-db-key-SENTINEL-1",
    "TG_API_ID": "987654321",
    "TG_API_HASH": "api-hash-SENTINEL-2",
    "MCP_TOKEN": "mcp-token-SENTINEL-3",
}

RU_KEY = "Кто-нибудь знает, сколько сейчас занимает продление ВНЖ в Валенсии?"
RU_EDITED = "Записались на отпечатки пальцев для карты резидента"
RU_DELETED = "Продам велосипед недорого, пишите в личку"
ES_KEY = "Ya se puede pedir cita para las renovaciones de la TIE por internet"


def scenario(now: datetime, second_run: bool) -> dict[str, Any]:
    """The chats as TDLib shows them on the first run, or on the second (after the edit,
    the deletion and the new messages)."""

    def msg(i: int, age: timedelta, text: str, sender: str, **extra: Any) -> dict[str, Any]:
        return {"id": i, "date": (now - age).isoformat(), "text": text, "sender_id": 100 + i,
                "sender_name": sender, **extra}  # fmt: skip

    ru = [msg(i, timedelta(days=120 - i), f"обычное сообщение номер {i}", "Оля")
          for i in range(1, 101)]  # fmt: skip
    ru[39] = msg(40, timedelta(days=80), RU_KEY, "Оля")  # old: arrives by backfill
    ru += [
        msg(101, timedelta(days=2), "Начальная версия текста про карту", "Дима"),
        msg(102, timedelta(days=1), RU_DELETED, "Дима"),
        msg(103, timedelta(hours=20), "обычное сообщение номер 103", "Оля"),
    ]
    es = [msg(i, timedelta(days=30 - i), f"aviso corriente número {i}", "Ayuntamiento")
          for i in range(1, 21)]  # fmt: skip
    if second_run:
        edited = now - timedelta(hours=1)
        ru[100] = msg(101, timedelta(days=2), RU_EDITED, "Дима", edit_date=edited.isoformat())
        del ru[101]
        ru.append(msg(104, timedelta(minutes=30), "обычное сообщение номер 104", "Оля"))
        es.append(msg(21, timedelta(minutes=20), ES_KEY, "Ayuntamiento"))
    return {
        "chats": [
            {"chat_id": RU, "title": "Русские в Валенсии", "kind": "supergroup",
             "username": None, "messages": ru, "page_sizes": [30, 30, 30]},
            {"chat_id": ES, "title": "Avisos Valencia", "kind": "channel",
             "username": "avisos_valencia", "messages": es},
        ],
    }  # fmt: skip


def texts(now: datetime) -> set[str]:
    both = scenario(now, False)["chats"] + scenario(now, True)["chats"]
    return {m["text"] for c in both for m in c["messages"]}


def test_p1_end_to_end(
    db_url: str,
    conn: Conn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    first.write_text(json.dumps(scenario(now, False)))
    second.write_text(json.dumps(scenario(now, True)))
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("PAGE_DELAY_MS", "0")
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    caplog.set_level(logging.DEBUG)
    cli = CliRunner()
    outputs: list[str] = []

    def tgi(*args: str) -> str:
        result = cli.invoke(app, list(args))
        assert result.exit_code == 0, (args, result.stdout, result.stderr)
        outputs.append(result.stdout + result.stderr)
        return result.stdout

    for chat_id in (RU, ES):
        tgi("chats", "enable", str(chat_id), "--account", ACCOUNT, "--scenario", str(first))

    # Run 1: the first sync catches up to the newest message and backfills the history.
    run1 = tgi("sync", "--scenario", str(first))
    assert " ok " in run1 and "embed_failed=0" in run1
    stored = conn.execute("SELECT chat_id, count(*) FROM message GROUP BY 1 ORDER BY 1")
    assert stored.fetchall() == [(ES, 20), (RU, 103)]

    # Run 2: catch-up with one edit, one deletion and new messages in both chats.
    run2 = tgi("sync", "--scenario", str(second))
    assert " ok " in run2 and "edited=1" in run2 and "tombstoned=1" in run2
    stored = conn.execute(
        "SELECT chat_id, count(*) FILTER (WHERE deleted_at IS NULL), count(*) FROM message"
        " GROUP BY 1 ORDER BY 1"
    )
    assert stored.fetchall() == [(ES, 21, 21), (RU, 103, 104)]
    left = conn.execute('SELECT count(*) FILTER (WHERE stale), count(*) FROM "window"')
    stale, windows = left.fetchone()  # type: ignore[misc]
    assert stale == 0 and windows > 0
    embedded = conn.execute("SELECT count(*) FROM window_embedding").fetchone()
    assert embedded == (windows,)

    def search(*args: str) -> list[dict[str, Any]]:
        return json.loads(tgi("search", "--mode", "keyword", *args, "--json"))  # type: ignore[no-any-return]

    def messages(hits: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {m["text"]: {**m, "chat_id": h["chat_id"], "window_id": h["window_id"]}
                for h in hits for m in h["messages"]}  # fmt: skip

    ru_hit = messages(search("продление ВНЖ"))[RU_KEY]
    assert (ru_hit["chat_id"], ru_hit["link"]) == (RU, "https://t.me/c/1500000001/40")
    assert ru_hit["message_id"] == 40 << 20 and ru_hit["window_id"] is not None
    es_hits = search("renovación TIE")
    assert es_hits[0]["chat_title"] == "Avisos Valencia"
    es_hit = messages(es_hits)[ES_KEY]
    assert es_hit["link"] == "https://t.me/avisos_valencia/21"

    assert RU_EDITED in messages(search("отпечатки"))
    assert search("Начальная версия") == []
    assert search("велосипед") == []
    (gone,) = search("велосипед", "--include-deleted")
    assert gone["window_id"] is None and gone["messages"][0]["text"] == RU_DELETED
    assert gone["start_date"] == gone["end_date"] == gone["messages"][0]["date"]
    human = tgi("search", "--mode", "keyword", "продление ВНЖ", "renovación TIE")
    assert "https://t.me/c/1500000001/40" in human and "https://t.me/avisos_valencia/21" in human

    # A10 over the whole session: logs carry ids and counts only.
    assert "sync" in caplog.text and str(RU) in caplog.text  # the logs were captured
    forbidden = texts(now) | {ACCOUNT, ACCOUNT.lstrip("+"), *SECRETS.values()}
    forbidden |= {"продление", "ВНЖ", "renovación", "отпечатки", "велосипед"}  # queries
    leaked = sorted(f for f in forbidden if f in caplog.text)
    assert leaked == []
    # The sync and chats commands print no message text or secrets either (search does
    # print messages: that is its output).
    for out in outputs[:4]:
        assert [f for f in forbidden if f in out] == []
