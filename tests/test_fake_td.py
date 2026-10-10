"""T4: the fake reproduces TDLib paging quirks, flood waits, faults and between-run changes."""

import json
from pathlib import Path

import pytest

from tgi.td_fake import FakeTdClient
from tgi.td_port import FloodWait, Message, TdClient, TdError, server_message_id

SCENARIOS = Path(__file__).parent / "scenarios"
PAGING_CHAT = -1001000000001
FLOOD_CHAT = -1001000000002
RUNS_CHAT = -1001000000003


def load(name: str) -> FakeTdClient:
    return FakeTdClient.from_file(SCENARIOS / f"{name}.json")


def page_all(td: TdClient, chat_id: int, limit: int) -> list[list[Message]]:
    """Page like ingestion will: advance on the last returned id until an empty page."""
    pages, from_id = [], 0
    while True:
        page = td.get_chat_history(chat_id, from_id, limit)
        pages.append(page)
        if not page:
            return pages
        from_id = page[-1].message_id


def test_fake_satisfies_port() -> None:
    td: TdClient = load("paging")
    assert [c.chat_id for c in td.get_chats()] == [PAGING_CHAT]
    assert td.get_chat(PAGING_CHAT).username == "paging_group"


def test_short_single_and_empty_pages() -> None:
    pages = page_all(load("paging"), PAGING_CHAT, limit=20)

    # page_sizes [25, 7, 1] capped at limit 20, then full pages; empty page ends paging.
    assert [len(p) for p in pages] == [20, 7, 1, 20, 12, 0]
    ids = [m.message_id for p in pages for m in p]
    assert ids == sorted(ids, reverse=True)
    assert [server_message_id(i) for i in ids] == list(range(60, 0, -1))


def test_tdlib_shaped_ids_and_fields() -> None:
    td = load("paging")
    (newest,) = td.get_chat_history(PAGING_CHAT, 0, 1)
    assert newest.message_id == 60 << 20 == 62914560
    by_id = {m.message_id: m for p in page_all(load("paging"), PAGING_CHAT, 100) for m in p}
    assert by_id[41 << 20].reply_to_message_id == 38 << 20
    assert by_id[11 << 20].has_media and by_id[11 << 20].text == ""


@pytest.mark.parametrize("from_inclusive", [False, True])
def test_from_message_id_boundary(from_inclusive: bool) -> None:
    scenario = json.loads((SCENARIOS / "paging.json").read_text())
    td = FakeTdClient({**scenario, "from_inclusive": from_inclusive})
    page = td.get_chat_history(PAGING_CHAT, 30 << 20, 3)
    expected = [30, 29, 28] if from_inclusive else [29, 28, 27]
    assert [server_message_id(m.message_id) for m in page] == expected


def test_flood_waits_then_error_then_success() -> None:
    td = load("flood")
    for _ in range(2):
        with pytest.raises(FloodWait) as flood:
            td.get_chat_history(FLOOD_CHAT, 0, 100)
        assert flood.value.retry_after == 2 and flood.value.code == 429
    assert len(td.get_chat_history(FLOOD_CHAT, 0, 100)) == 3
    with pytest.raises(TdError, match="connection lost"):
        td.get_chat_history(FLOOD_CHAT, 0, 100)
    assert len(td.get_chat_history(FLOOD_CHAT, 0, 100)) == 3
    assert len(td.history_calls) == 5


def test_next_run_applies_add_edit_delete_and_resets_faults() -> None:
    td = load("runs")
    before = {m.message_id: m for m in td.get_chat_history(RUNS_CHAT, 0, 100)}
    assert set(before) == {1 << 20, 2 << 20, 3 << 20}

    td.next_run()
    with pytest.raises(FloodWait):
        td.get_chat_history(RUNS_CHAT, 0, 100)
    after = {m.message_id: m for m in td.get_chat_history(RUNS_CHAT, 0, 100)}

    assert set(after) == {1 << 20, 2 << 20, 4 << 20}
    assert after[2 << 20].text == "renovación TIE (editado)"
    assert after[2 << 20].edit_date is not None and before[2 << 20].edit_date is None
    assert after[1 << 20] == before[1 << 20]
    with pytest.raises(IndexError):
        td.next_run()


def test_unknown_chat_bad_limit_and_close() -> None:
    td = load("runs")
    with pytest.raises(TdError, match="Chat not found"):
        td.get_chat(1)
    with pytest.raises(TdError, match="limit"):
        td.get_chat_history(RUNS_CHAT, 0, 101)
    td.close()
    with pytest.raises(TdError, match="closed"):
        td.get_chats()


@pytest.mark.parametrize("bad", [0, -(1 << 20), (1 << 20) + 1])
def test_server_message_id_rejects_non_server_ids(bad: int) -> None:
    with pytest.raises(ValueError):
        server_message_id(bad)


def test_server_message_id() -> None:
    assert server_message_id(1_048_576) == 1
