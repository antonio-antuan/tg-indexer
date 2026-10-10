# ruff: noqa: E501, S311
"""QA T9: search probes beyond the PR's tests (A6, filters, links, CLI, tags, oracle fuzz)."""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, A, B, chat, enable
from test_search import TEXTS, found
from typer.testing import CliRunner

from tgi import chats, search
from tgi.cli import app
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20


def build(conn: Conn, texts: list[str], gap_h: int = 3) -> FakeTdClient:
    msgs = [{"id": i, "date": (NOW - timedelta(hours=gap_h * (len(texts) - i + 1))).isoformat(),
             "text": t, "sender_name": "Ana"} for i, t in enumerate(texts, 1)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)
    return td


def ids(conn: Conn, *qs: str, **kw: Any) -> list[int]:
    return sorted(m for _, m in found(search.search(conn, list(qs), **kw, mode="keyword")))


def close(conn: Conn, texts: list[str]) -> None:
    """Messages two minutes apart: one window."""
    msgs = [{"id": i, "date": (NOW - timedelta(minutes=2 * (len(texts) - i))).isoformat(),
             "text": t, "sender_name": "Ana"} for i, t in enumerate(texts, 1)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)


# --- A6 -------------------------------------------------------------------------------------


def test_short_tokens_and_case(conn: Conn) -> None:
    build(conn, ["ВНЖ", "вне дома", "внук приехал", "ВНЖ-карта готова", "la tienda tiene TIE",
                 "tie a knot", "tiene hijos", "NIE y TIE renovación", "mi tía", "TI", "ТИЕ"])  # fmt: skip
    sids = lambda *q: ids(conn, *q)  # noqa: E731
    assert sids("ВНЖ") == [1, 4] and sids("внж") == [1, 4]  # not вне, внук
    assert 7 not in sids("TIE") and 9 not in sids("TIE") and {5, 6, 8} <= set(sids("TIE"))
    assert sids("tiene") == [5, 7]  # Spanish stop word: FTS drops it, nothing floods
    assert sids("TI") == [10] and sids("NIE") == [8]
    assert sids("ТИЕ") == [11]  # Cyrillic look-alike is a different word


def test_stop_words_do_not_flood(conn: Conn) -> None:
    build(
        conn, ["la casa de mi tía", "el gato de Ana", "the cat and the dog", "и да и нет", "hola"]
    )
    n = lambda *q: ids(conn, *q)  # noqa: E731
    assert n("de") == [1, 2] and n("the") == [3] and n("и") == [4]  # only literal matches
    assert n("de la") == [1]  # websearch ANDs the words
    assert len(n("de", "the", "и", "a", "el", "and")) <= 4


def test_negation_is_honoured_on_the_trigram_leg(conn: Conn) -> None:
    """websearch syntax: `-word` excludes. Short tokens bypass it (probe, not in the spec's A6)."""
    build(conn, ["ВНЖ карта готова", "ВНЖ отказ"])
    got = ids(conn, "ВНЖ -карта")
    assert got == [2], got


def test_tombstoned_message_not_shown_inside_a_window_hit(conn: Conn) -> None:
    close(conn, ["сообщение про ВНЖ", "SECRET-DELETED text", "ещё одно"])
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE server_message_id = 2"
    )  # sync crashed before the rebuild
    texts = [m.text for h in search.search(conn, ["ВНЖ"], mode="keyword") for m in h.messages]
    assert "SECRET-DELETED text" not in texts, texts


def test_dates_are_utc_days_whatever_the_session_zone(conn: Conn) -> None:
    build(conn, ["ВНЖ a", "ВНЖ b", "ВНЖ c"], gap_h=1)
    conn.execute(
        "UPDATE message SET date = '2026-09-30T23:59:59.999999+00' WHERE server_message_id = 1"
    )
    conn.execute("UPDATE message SET date = '2026-10-01T00:00:00+00' WHERE server_message_id = 2")
    conn.execute(
        "UPDATE message SET date = '2026-10-01T23:59:59.999999+00' WHERE server_message_id = 3"
    )
    d = lambda y, m, dd: datetime(y, m, dd).date()  # noqa: E731
    for zone in ("UTC", "Pacific/Kiritimati", "America/Los_Angeles"):
        conn.execute(f"SET TimeZone = '{zone}'")
        s = lambda **kw: ids(conn, "ВНЖ", **kw)  # noqa: E731
        assert s(until=d(2026, 9, 30)) == [1], zone
        assert s(since=d(2026, 10, 1)) == [2, 3], zone
        assert s(since=d(2026, 10, 1), until=d(2026, 10, 1)) == [2, 3], zone
        assert s(since=d(2026, 10, 2)) == [] and s(until=d(2026, 9, 29)) == [], zone
        h = search.search(conn, ["ВНЖ"], since=d(2026, 10, 1), mode="keyword")[0]
        assert h.start_date.utcoffset() == timedelta(0)


# --- fuzz against an independent oracle ----------------------------------------------------

WORDS = ["ВНЖ", "внж", "продление", "продлении", "renovación", "renovaciones", "renewal", "renewals",
         "tiene", "tie", "TIE", "tied", "casa", "de", "la", "the", "и", "в", "NIE", "visa",
         "hola", "паспорт", "cita", "tía", "карта", "ties"]  # fmt: skip


def oracle(conn: Conn, tokens: list[str]) -> set[int]:
    """A single-token query: FTS in any config alone, or (short non-stop token) strict word similarity >= .5."""
    q = tokens[0]
    rows = conn.execute(
        "SELECT server_message_id FROM message WHERE deleted_at IS NULL AND ("
        " to_tsvector('russian', text) @@ websearch_to_tsquery('russian', %(q)s)"
        " OR to_tsvector('spanish', text) @@ websearch_to_tsquery('spanish', %(q)s)"
        " OR to_tsvector('english', text) @@ websearch_to_tsquery('english', %(q)s)"
        " OR (length(%(q)s) <= 6 AND to_tsvector('russian', %(q)s) <> ''::tsvector"
        "     AND to_tsvector('spanish', %(q)s) <> ''::tsvector AND to_tsvector('english', %(q)s) <> ''::tsvector"
        "     AND strict_word_similarity(%(q)s, text) >= 0.5))",
        {"q": q},
    )
    return {r[0] for r in rows}


@pytest.mark.parametrize("seed", range(5))
def test_single_token_queries_match_the_oracle(conn: Conn, seed: int) -> None:
    rnd = random.Random(seed)
    texts = [" ".join(rnd.choices(WORDS, k=rnd.randint(1, 12))) for _ in range(60)]
    build(conn, texts, gap_h=1)
    for w in WORDS:
        got = {
            mm.message_id >> SHIFT for h in search.search(conn, [w], limit=50, mode="keyword") for mm in h.messages}  # fmt: skip
        want = oracle(conn, [w])
        # windows are single messages here (1h gaps); K=100 > 60 so nothing is cut
        assert got == want, (w, sorted(got ^ want))


# --- include_deleted, filters ----------------------------------------------------------------


def test_include_deleted_returns_single_message_hits(conn: Conn) -> None:
    close(conn, ["ВНЖ первый", "ВНЖ второй", "ВНЖ третий"])
    conn.execute("UPDATE message SET deleted_at = now() WHERE server_message_id = 2")
    assert ids(conn, "ВНЖ") == [1, 3] or ids(conn, "ВНЖ") == [
        1,
        2,
        3,
    ]  # see the #16 probe for the stale window
    # rebuild as sync would
    conn.execute("UPDATE chat SET windows_dirty_from = %s", (1 << 20,))
    from test_ingest import SETTINGS

    from tgi import windows

    windows.rebuild_dirty(conn, SETTINGS)
    assert ids(conn, "ВНЖ") == [1, 3]
    hits = search.search(conn, ["ВНЖ"], include_deleted=True, mode="keyword")
    single = [h for h in hits if h.window_id is None]
    assert len(single) == 1 and [m.message_id >> SHIFT for m in single[0].messages] == [2]
    assert single[0].start_date == single[0].end_date == single[0].messages[0].date
    assert sorted(m.message_id >> SHIFT for h in hits for m in h.messages) == [1, 2, 3]


def test_filters_combine_and_proposed_tags_do_not_filter(conn: Conn) -> None:
    msgs = lambda t: [{"id": 1, "date": NOW.isoformat(), "text": t, "sender_name": "x"}]  # noqa: E731
    td = FakeTdClient({"chats": [chat(A, msgs("ВНЖ раз")), chat(B, msgs("ВНЖ два"))], "runs": [{}]})
    enable(conn, td, A, B)
    sync(conn, td)
    chats.propose_tag(conn, A, "visa", "agent")
    assert search.search(conn, ["ВНЖ"], tags=["visa"], mode="keyword") == []  # proposed only
    chats.confirm_tag(conn, A, "visa")
    chats.confirm_tag(conn, B, "other")
    ch = lambda **kw: sorted(h.chat_id for h in search.search(conn, ["ВНЖ"], **kw, mode="keyword"))  # noqa: E731
    assert ch(tags=["visa"]) == [A] and ch(tags=["visa", "other"]) == sorted([A, B])
    assert ch(tags=["visa"], chat_ids=[B]) == [] and ch(tags=["visa"], chat_ids=[A, B]) == [A]
    assert ch(chat_ids=[B]) == [B] and ch(chat_ids=[123]) == [] and ch(tags=["nope"]) == []
    hit = search.search(conn, ["ВНЖ"], chat_ids=[A], mode="keyword")[0]
    assert hit.tags == ["visa"]  # only confirmed tags are shown
    chats.propose_tag(conn, A, "extra", "agent")
    assert search.search(conn, ["ВНЖ"], chat_ids=[A], mode="keyword")[0].tags == ["visa"]
    chats.propose_tag(conn, A, "visa", "agent")  # does not demote a confirmed tag
    assert ch(tags=["visa"]) == [A]
    chats.remove_tag(conn, A, "visa")
    assert ch(tags=["visa"]) == []


def test_modes_and_arguments(conn: Conn) -> None:
    close(conn, ["ВНЖ"])
    assert search.search(conn, ["ВНЖ"], mode="hybrid")  # P2.T2: no longer NotImplementedError
    search.search(conn, ["ВНЖ"], mode="semantic")
    with pytest.raises(ValueError):
        search.search(conn, ["ВНЖ"], mode="bogus")  # type: ignore[arg-type]
    for q in ([], [""], ["  "]):
        with pytest.raises(ValueError):
            search.search(conn, q, mode="keyword")
    for lim in (0, 51, -1):
        with pytest.raises(ValueError):
            search.search(conn, ["ВНЖ"], limit=lim, mode="keyword")
    assert search.search(conn, ["ВНЖ", "  "], mode="keyword")  # blank queries are dropped


def test_rrf_across_queries_and_ranking(conn: Conn) -> None:
    build(
        conn,
        ["renewal and paella", "renewal only", "paella only", "tienda TIE", "tiempo de visa"],
        gap_h=2,
    )
    hits = search.search(conn, ["renewal", "paella"], mode="keyword")
    order = [h.messages[0].message_id >> SHIFT for h in hits]
    assert order[0] == 1  # in both lists
    assert hits[0].score == pytest.approx(
        2 / 62
    )  # rank 2 in both (ties: newest first) and {o for o in order[1:]} == {2, 3}
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    # FTS hits outrank trigram-only hits for the same query
    build2 = search.search(conn, ["TIE"], mode="keyword")
    assert build2[0].messages[0].message_id >> SHIFT == 4


# --- A11 and ingest ---------------------------------------------------------------------------


def test_non_server_ids_are_skipped_at_ingest_and_never_linked(
    conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    good = {"id": 2, "date": NOW.isoformat(), "text": "ВНЖ ok", "sender_name": "x"}
    local = {
        "message_id": (3 << 20) + 5,
        "date": NOW.isoformat(),
        "text": "ВНЖ local",
        "sender_name": "x",
    }
    td = FakeTdClient({"chats": [chat(A, [good, local])], "runs": [{}]})
    enable(conn, td, A)
    caplog.set_level("DEBUG")
    sync(conn, td)
    assert conn.execute("SELECT count(*) FROM message").fetchone() == (1,)
    assert "ВНЖ local" not in caplog.text
    h = search.search(conn, ["ВНЖ"], mode="keyword")
    assert len(h) == 1 and h[0].messages[0].link == "https://t.me/c/1000000101/2"


def test_links_for_every_chat_kind_from_the_database(conn: Conn) -> None:
    kinds = {-1001111111111: ("supergroup", None), -1002222222222: ("channel", "chan"),
             -1003333333333: ("supergroup", "grp"), -555: ("basic_group", None), 4242: ("private", "someone")}  # fmt: skip
    chs = [
        chat(cid, [{"id": 7, "date": NOW.isoformat(), "text": "ВНЖ link", "sender_name": "x"}])
        for cid in kinds
    ]
    td = FakeTdClient({"chats": chs, "runs": [{}]})
    enable(conn, td, *kinds)
    for cid, (kind, user) in kinds.items():
        conn.execute(
            "UPDATE chat SET kind = %s, username = %s WHERE chat_id = %s", (kind, user, cid)
        )
    sync(conn, td)
    got = {h.chat_id: h.messages[0].link for h in search.search(conn, ["ВНЖ"], mode="keyword")}
    assert got == {-1001111111111: "https://t.me/c/1111111111/7", -1002222222222: "https://t.me/chan/7",
                   -1003333333333: "https://t.me/grp/7", -555: None, 4242: None}  # fmt: skip
    assert all(
        m.message_id == 7 << 20
        for h in search.search(conn, ["ВНЖ"], mode="keyword")
        for m in h.messages
    )


# --- CLI, logs --------------------------------------------------------------------------------


@pytest.fixture
def cli(db_url: str, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("DATABASE_URL", db_url)
    return CliRunner()


def test_cli_json_shape_exit_codes_and_no_text_in_logs(
    conn: Conn, cli: CliRunner, caplog: pytest.LogCaptureFixture
) -> None:
    close(conn, [TEXTS[4], "SECRETWORD in a window", "ещё"])
    chats.confirm_tag(conn, A, "visa")
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="psycopg")
    out = cli.invoke(app, ["search", "внж", "SECRETWORD", "--json"])
    assert out.exit_code == 0, out.stderr
    data = json.loads(out.stdout)
    assert len(data) == 1
    h = data[0]
    assert list(h) == [
        "window_id",
        "chat_id",
        "chat_title",
        "tags",
        "start_date",
        "end_date",
        "score",
        "messages",
        "highlights",
    ]
    assert list(h["messages"][0]) == ["message_id", "sender_name", "date", "text", "link"]
    assert (
        h["tags"] == ["visa"] and isinstance(h["window_id"], int) and isinstance(h["score"], float)
    )
    assert h["start_date"].endswith("+00:00") and len(h["messages"]) == 3
    assert any("SECRETWORD" in x for x in h["highlights"])
    human = cli.invoke(app, ["search", "SECRETWORD"])
    assert (
        human.exit_code == 0
        and "SECRETWORD" in human.stdout
        and "https://t.me/c/1000000101/" in human.stdout
    )
    assert "SECRETWORD" not in caplog.text and "внж" not in caplog.text.lower()
    assert "SECRETWORD" not in out.stderr + human.stderr
    for args in (
        ["search"],
        ["search", "x", "--limit", "0"],
        ["search", "  "],
        ["search", "x", "--until", "2026-13-01"],
    ):
        assert cli.invoke(app, args).exit_code != 0, args
    assert cli.invoke(app, ["search", "zzzz"]).exit_code == 0  # no hits is not an error


def test_cli_tags_edge_cases(conn: Conn, cli: CliRunner) -> None:
    close(conn, ["x"])
    cli.invoke(app, ["tags", "propose", str(A), "a", "--by", "agent"])
    assert conn.execute("SELECT status, proposed_by FROM chat_tag").fetchall() == [
        ("proposed", "agent")
    ]
    assert (
        cli.invoke(app, ["tags", "confirm", str(A), "b"]).exit_code == 0
    )  # confirm without propose adds it
    assert cli.invoke(app, ["tags", "propose", str(A), "b"]).exit_code == 0
    assert conn.execute("SELECT status FROM chat_tag WHERE tag = 'b'").fetchone() == ("confirmed",)
    assert cli.invoke(app, ["tags", "remove", str(A), "a"]).exit_code == 0
    assert (
        cli.invoke(app, ["tags", "propose", "-1009", "a"]).exit_code == 1
    )  # negative chat id parsed
    assert cli.invoke(app, ["tags", "propose", str(A), "x" * 5000]).exit_code in (0, 1, 2)


def test_trigram_plan_uses_the_gin_index_for_two_tokens(conn: Conn) -> None:
    close(conn, ["ВНЖ"])
    stmt, params = search.leg_query(conn, "ВНЖ TIE", None, None, None, False)
    from psycopg import sql

    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        conn.execute("SET LOCAL pg_trgm.strict_word_similarity_threshold = 0.5")
        plan = "\n".join(str(r[0]) for r in conn.execute(sql.SQL("EXPLAIN ") + stmt, params))
    assert "message_text_trgm_idx" in plan and "<<%" in plan


def test_full_text_hits_rank_above_trigram_only_hits(conn: Conn) -> None:
    build(conn, ["he tied it", "ties", "tiesx foo", "tiez"], gap_h=3)
    legs = search.keyword_leg(conn, "ties", None, None, None, False)
    assert [(m.message_id >> SHIFT, m.fts is not None, m.trgm is not None) for m in legs] == [
        (2, True, True), (1, True, False), (3, False, True)]  # fmt: skip
    assert [
        h.messages[0].message_id >> SHIFT for h in search.search(conn, ["ties"], mode="keyword")
    ] == [2, 1, 3]


# --- #16: stale windows (a run failed between the sweep and the window rebuild) -------------


def stale_window(conn: Conn) -> list[int]:
    close(
        conn, ["сообщение про ВНЖ", "SECRET-DELETED text", "ещё одно ВНЖ"]
    )  # 4, 2, 0 min before NOW
    conn.execute("UPDATE message SET deleted_at = now() WHERE server_message_id = 2")
    return [r[0] for r in conn.execute('SELECT window_id FROM "window"')]


def test_16_live_neighbours_hide_the_deleted_message_by_default(conn: Conn) -> None:
    assert len(stale_window(conn)) == 1
    (hit,) = search.search(conn, ["ВНЖ"], mode="keyword")
    assert [m.message_id >> SHIFT for m in hit.messages] == [1, 3]
    assert "SECRET" not in json.dumps(hit.as_dict(), ensure_ascii=False)
    assert (hit.start_date, hit.end_date) == (hit.messages[0].date, hit.messages[-1].date)
    assert hit.end_date - hit.start_date == timedelta(minutes=4)


def test_16_include_deleted_shows_it_in_the_window(conn: Conn) -> None:
    stale_window(conn)
    hits = search.search(conn, ["ВНЖ"], include_deleted=True, mode="keyword")
    assert len(hits) == 1 and hits[0].window_id is not None
    assert [m.message_id >> SHIFT for m in hits[0].messages] == [1, 2, 3]
    assert hits[0].end_date - hits[0].start_date == timedelta(minutes=4)


def test_16_window_dates_follow_shown_messages_at_the_edges(conn: Conn) -> None:
    stale_window(conn)
    conn.execute("UPDATE message SET deleted_at = now() WHERE server_message_id = 3")
    (hit,) = search.search(conn, ["ВНЖ"], mode="keyword")
    assert [m.message_id >> SHIFT for m in hit.messages] == [1]
    assert hit.start_date == hit.end_date == hit.messages[0].date  # only the shown message
    conn.execute("UPDATE message SET deleted_at = NULL WHERE server_message_id = 3")
    conn.execute("UPDATE message SET deleted_at = now() WHERE server_message_id = 1")
    (hit,) = search.search(conn, ["ВНЖ"], mode="keyword")
    assert [m.message_id >> SHIFT for m in hit.messages] == [3] and hit.start_date == hit.end_date


def test_16_hit_tombstoned_between_the_leg_and_the_fetch(
    conn: Conn, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Window hit and no-window hit whose matched messages are tombstoned after the leg ran."""
    build(conn, ["ВНЖ одиночка", "другое", "ВНЖ второй"], gap_h=3)
    conn.execute(
        'DELETE FROM "window" WHERE first_message_id = %s', (3 << 20,)
    )  # message 3 has no window yet
    real = search.keyword_leg

    def racing(*a: Any, **kw: Any) -> Any:
        out = real(*a, **kw)
        conn.execute("UPDATE message SET deleted_at = now() WHERE server_message_id IN (1, 3)")
        return out

    monkeypatch.setattr(search, "keyword_leg", racing)
    assert search.search(conn, ["ВНЖ"], mode="keyword") == []  # both vanish, no error, no text
    monkeypatch.setattr(search, "keyword_leg", real)
    got = search.search(conn, ["ВНЖ"], include_deleted=True, mode="keyword")
    assert sorted(m.message_id >> SHIFT for h in got for m in h.messages) == [1, 3]
    assert all(h.start_date == h.end_date for h in got)
