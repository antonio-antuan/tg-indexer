"""Why a hit was found (`matched_by`, `matched_queries`, `semantic_distance`) and compact
results for chat assistants (`compact`)."""

import base64
import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, A, B, chat, enable
from test_mcp import in_process
from typer.testing import CliRunner

from tgi import search
from tgi.cli import app
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
LONG = "word " * 200  # 1000 characters: over COMPACT_CHARS, within one window


def chatter() -> list[dict[str, Any]]:
    """20 messages a minute apart (one window): a TIE question at 10, a long text at 3."""
    msgs = []
    for i in range(1, 21):
        text = "TIE renewal question" if i == 10 else LONG.strip() if i == 3 else f"chatter {i}"
        msgs.append({"id": i, "date": (NOW - timedelta(minutes=30 - i)).isoformat(),
                     "text": text, "sender_name": "S"})  # fmt: skip
    return msgs


@pytest.fixture
def archive(conn: Conn) -> None:
    other = [{"id": 1, "date": (NOW - timedelta(days=2)).isoformat(),
              "text": "renovación de la TIE", "sender_name": "R"}]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, chatter(), username="a"), chat(B, other)], "runs": []})
    enable(conn, td, A, B)
    assert sync(conn, td).status == "ok"


def ids(hit: search.Hit) -> list[int]:
    return [m.message_id >> SHIFT for m in hit.messages]


# --- provenance ----------------------------------------------------------------------------------


def test_keyword_hits_say_which_leg_and_query_found_them(archive: None, conn: Conn) -> None:
    hits = search.search(conn, ["", "renewal", "renovación"], mode="keyword")
    by_chat = {h.chat_id: h for h in hits}
    # positions are in the caller's list: the empty query at 0 still counts
    assert by_chat[A].matched_queries == (1,) and by_chat[B].matched_queries == (2,)
    assert "keyword" in by_chat[A].matched_by and by_chat[A].semantic_distance is None


def test_trigram_only_and_semantic_hits(archive: None, conn: Conn) -> None:
    (tie,) = [h for h in search.search(conn, ["TIE"], mode="keyword") if h.chat_id == A]
    assert "trigram" in tie.matched_by
    semantic = search.search(conn, ["TIE renewal question"], mode="semantic")
    assert semantic and all(h.matched_by == ("semantic",) for h in semantic)
    assert all(h.semantic_distance is not None for h in semantic)
    assert semantic[0].semantic_distance <= semantic[-1].semantic_distance  # type: ignore[operator]


def test_hybrid_merges_the_legs(archive: None, conn: Conn) -> None:
    (hit,) = [h for h in search.search(conn, ["TIE renewal question"]) if h.chat_id == A]
    assert hit.matched_by[0] == "keyword" and hit.matched_by[-1] == "semantic"
    assert hit.semantic_distance is not None and hit.matched_queries == (0,)


# --- compact -------------------------------------------------------------------------------------


def test_full_hits_are_unchanged(archive: None, conn: Conn) -> None:
    (hit,) = [h for h in search.search(conn, ["renewal"], mode="keyword") if h.chat_id == A]
    assert ids(hit) == list(range(1, 21)) and hit.omitted_messages == 0
    assert not any(m.truncated for m in hit.messages)
    assert "truncated" not in hit.as_dict()["messages"][0]


def test_compact_keeps_the_match_and_its_neighbours_and_cuts_long_texts(
    archive: None, conn: Conn
) -> None:
    (hit,) = [h for h in search.search(conn, ["renewal"], mode="keyword", compact=True)
              if h.chat_id == A]  # fmt: skip
    assert ids(hit) == [8, 9, 10, 11, 12] and hit.omitted_messages == 15
    (cut,) = [h for h in search.search(conn, ["chatter 4"], mode="keyword", compact=True)
              if h.chat_id == A]  # fmt: skip
    long = next(m for m in cut.messages if m.message_id >> SHIFT == 3)
    assert long.truncated and len(long.text) <= search.COMPACT_CHARS + 1  # cut, then "…"
    assert long.text.endswith("…") and cut.as_dict()["messages"][0].get("truncated") in (None, True)


def test_compact_semantic_only_hits_keep_the_first_messages(archive: None, conn: Conn) -> None:
    hits = search.search(conn, ["chatter"], mode="semantic", compact=True)
    (hit,) = [h for h in hits if h.chat_id == A]
    assert ids(hit) == [1, 2, 3, 4, 5] and hit.omitted_messages == 15


def test_compact_recent_keeps_every_message_and_cuts_texts(archive: None, conn: Conn) -> None:
    answer = search.recent(conn, since=NOW - timedelta(hours=1), until=NOW, compact=True, now=NOW)
    (hit,) = answer.hits
    assert ids(hit) == list(range(1, 21)) and hit.omitted_messages == 0
    assert [m.message_id >> SHIFT for m in hit.messages if m.truncated] == [3]


def test_compact_over_mcp_and_a_strict_boolean(archive: None, db_url: str) -> None:
    full, compact, bad = in_process(db_url, [
        ("search", {"queries": ["renewal"], "mode": "keyword"}),
        ("search", {"queries": ["renewal"], "mode": "keyword", "compact": True}),
        ("recent", {"hours": 1, "compact": "yes"}),
    ])  # fmt: skip
    (a_full,) = [h for h in full if h["chat_id"] == A]
    (a_compact,) = [h for h in compact if h["chat_id"] == A]
    assert len(a_full["messages"]) == 20 and a_full["omitted_messages"] == 0
    assert len(a_compact["messages"]) == 5 and a_compact["omitted_messages"] == 15
    assert a_full["matched_by"] == a_compact["matched_by"]
    assert bad["error"]["code"] == "bad_argument"


def test_cli_compact_and_provenance(
    archive: None, db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    out = runner.invoke(app, ["search", "renewal", "--mode", "keyword", "--compact", "--json"])
    assert out.exit_code == 0, out.output
    (hit,) = [h for h in json.loads(out.output) if h["chat_id"] == A]
    assert hit["omitted_messages"] == 15
    text = runner.invoke(app, ["search", "renewal", "--mode", "keyword"])
    assert "; keyword" in text.output
    since, until = (NOW - timedelta(hours=1)).isoformat(), NOW.isoformat()
    recent = runner.invoke(app, ["recent", "--since", since, "--until", until, "--compact"])
    assert recent.exit_code == 0 and "…" in recent.output


# --- review round 1 ------------------------------------------------------------------------------


def a_hit(conn: Conn, **kw: Any) -> search.Hit:
    (hit,) = [h for h in search.search(conn, ["renewal"], mode="keyword", **kw) if h.chat_id == A]
    return hit


def test_a_typo_is_found_by_the_trigram_leg_alone(archive: None, conn: Conn) -> None:
    (hit,) = [h for h in search.search(conn, ["renewl"], mode="keyword") if h.chat_id == A]
    assert hit.matched_by == ("trigram",) and hit.matched_message_ids == (10 << SHIFT,)


def test_duplicate_and_mixed_queries_keep_their_positions(archive: None, conn: Conn) -> None:
    twice = search.search(conn, ["renewal", "renewal"], mode="keyword")
    assert {h.matched_queries for h in twice if h.chat_id == A} == {(0, 1)}
    mixed = search.search(conn, ["", "-zzzq", "TIE renewal question"])
    (hit,) = [h for h in mixed if h.chat_id == A]
    assert 2 in hit.matched_queries and hit.semantic_distance is not None
    assert 1 in hit.matched_queries  # the negation-only query matches as keyword, by design


def test_semantic_distance_is_the_smallest_over_the_queries(archive: None, conn: Conn) -> None:
    def distance(queries: list[str]) -> float:
        (hit,) = [h for h in search.search(conn, queries, mode="semantic") if h.chat_id == A]
        assert hit.semantic_distance is not None
        return hit.semantic_distance

    one, two = distance(["chatter 4"]), distance(["word word"])
    assert distance(["chatter 4", "word word"]) == pytest.approx(min(one, two))


def test_without_embeddings_hybrid_reports_keyword_only(archive: None, conn: Conn) -> None:
    conn.execute("DELETE FROM window_embedding")
    (hit,) = [h for h in search.search(conn, ["TIE renewal question"]) if h.chat_id == A]
    assert "semantic" not in hit.matched_by and hit.semantic_distance is None


def test_compact_keeps_a_single_message_hit(archive: None, conn: Conn) -> None:
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s",
        (A, 10 << SHIFT),
    )
    hits = search.search(conn, ["renewal"], mode="keyword", include_deleted=True, compact=True)
    (hit,) = [h for h in hits if h.chat_id == A]
    assert hit.window_id is None and ids(hit) == [10] and hit.omitted_messages == 0


def test_compacted_edges(archive: None, conn: Conn) -> None:
    full = a_hit(conn)
    by_id = {m.message_id >> SHIFT: m.message_id for m in full.messages}
    assert ids(search.compacted(full, {by_id[9], by_id[10]})) == [7, 8, 9, 10, 11, 12]  # merged
    assert ids(search.compacted(full, {999})) == [1, 2, 3, 4, 5]  # none present: the sample
    assert ids(search.compacted(full, {by_id[1]})) == [1, 2, 3]  # clipped at the start
    exact = replace(full.messages[0], text="x" * search.COMPACT_CHARS)
    over = replace(full.messages[0], text="x" * (search.COMPACT_CHARS + 1))
    kept = search.compacted(replace(full, messages=[exact, over]), None).messages
    assert (kept[0].truncated, kept[0].text) == (False, exact.text)
    assert kept[1].truncated and kept[1].text == "x" * search.COMPACT_CHARS + "…"


def test_the_recent_cursor_keeps_compact(archive: None, conn: Conn, db_url: str) -> None:
    since, until = (NOW - timedelta(days=3)).isoformat(), NOW.isoformat()
    (first,) = in_process(
        db_url, [("recent", {"since": since, "until": until, "limit": 1, "compact": True})]
    )
    assert any(m.get("truncated") for h in first["hits"] for m in h["messages"])
    assert search._parse_cursor(first["next_cursor"])[4] is True
    (full,) = in_process(db_url, [("recent", {"since": since, "until": until, "limit": 1})])
    assert search._parse_cursor(full["next_cursor"])[4] is False


def test_a_compact_cursor_wins_over_the_argument_and_must_be_a_boolean(
    archive: None, conn: Conn
) -> None:
    first = search.recent(conn, since=NOW - timedelta(days=3), until=NOW, limit=1, compact=True,
                          now=NOW)  # fmt: skip
    assert first.next_cursor is not None
    rest = search.recent(conn, cursor=first.next_cursor, compact=False, now=NOW)
    page_one = search.recent(conn, since=NOW - timedelta(days=3), until=NOW, limit=1, now=NOW)
    assert any(m.truncated for m in first.hits[0].messages)
    assert not any(m.truncated for m in page_one.hits[0].messages)
    assert rest.hits  # B's short message: nothing to cut, but the walk goes on
    state = json.loads(base64.urlsafe_b64decode(first.next_cursor))
    for bad in ("yes", 1, None):
        forged = base64.urlsafe_b64encode(json.dumps({**state, "compact": bad}).encode()).decode()
        with pytest.raises(ValueError, match="cursor"):
            search.recent(conn, cursor=forged, now=NOW)


def test_matched_message_ids_in_every_shape(archive: None, conn: Conn) -> None:
    (several,) = [h for h in search.search(conn, ["chatter 4", "chatter 12"], mode="keyword")
                  if h.chat_id == A]  # fmt: skip
    assert [i >> SHIFT for i in several.matched_message_ids][:2] == sorted(
        i >> SHIFT for i in several.matched_message_ids
    )[:2]  # window order
    assert {4, 12} <= {i >> SHIFT for i in several.matched_message_ids}
    semantic = [h for h in search.search(conn, ["chatter"], mode="semantic") if h.chat_id == A]
    assert semantic[0].matched_message_ids == ()
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND message_id = %s",
        (A, 10 << SHIFT),
    )
    hits = search.search(conn, ["renewal"], mode="keyword", include_deleted=True)
    (single,) = [h for h in hits if h.chat_id == A]
    assert single.matched_message_ids == (10 << SHIFT,)
