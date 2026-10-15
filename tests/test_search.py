"""T9: keyword search (SPEC §5.5, A6), chat tags and `tgi search` / `tgi tags`."""

import json
from datetime import date, datetime, timedelta
from typing import Any

import psycopg
import pytest
from psycopg import sql
from test_embed import sync
from test_ingest import NOW, A, B, chat, enable
from typer.testing import CliRunner

from tgi import chats, search
from tgi.cli import app
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20

FORTY = (
    "Сегодня ходили в полицию по поводу документов и там сказали что ВНЖ, который "
    "мы подавали весной, пока не готов и нужно прийти через две недели с паспортом "
    "фотографиями справкой о прописке и квитанцией об оплате пошлины иначе ничего не примут"
)
TEXTS = {
    1: "Вопрос о продлении карты",
    2: "Hubo muchas renovaciones este mes",
    3: "Two renewals were filed today",
    4: "у меня внж истекает в марте",
    5: FORTY,
    6: "La empresa tiene una oficina nueva",
    7: "paella recipe from grandma",
    8: "thanks, will try it",
}
B_DATE = datetime(2026, 9, 20, 9, 0, tzinfo=NOW.tzinfo)


def a_messages() -> list[dict[str, Any]]:
    """Each message an hour apart (its own window), except 8: ten minutes after 7 (one window)."""
    msgs = [{"id": i, "date": (NOW - timedelta(hours=10 - i)).isoformat(), "text": t,
             "sender_name": "Ana"} for i, t in TEXTS.items()]  # fmt: skip
    msgs[7]["date"] = (NOW - timedelta(hours=2, minutes=50)).isoformat()
    return msgs


def scenario(runs: list[dict[str, Any]] | None = None) -> FakeTdClient:
    b = [{"id": 1, "date": B_DATE.isoformat(), "text": "renewal of the lease", "sender_name": "Bo"}]
    return FakeTdClient({"chats": [chat(A, a_messages()), chat(B, b)], "runs": runs or []})


@pytest.fixture
def archive(conn: Conn) -> FakeTdClient:
    td = scenario([{"delete": [{"chat_id": A, "id": 4}]}])
    enable(conn, td, A, B)
    sync(conn, td)
    return td


def found(hits: list[search.Hit]) -> list[tuple[int, int]]:
    """(chat_id, server id) of every message of every hit, in hit order."""
    return [(h.chat_id, m.message_id >> SHIFT) for h in hits for m in h.messages]


def first_ids(hits: list[search.Hit]) -> set[tuple[int, int]]:
    return {(h.chat_id, h.messages[0].message_id >> SHIFT) for h in hits}


# --- A6 -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("продление", {(A, 1)}),
        ("renovación", {(A, 2)}),
        ("renewal", {(A, 3), (B, 1)}),
        ("ВНЖ", {(A, 4), (A, 5)}),
    ],
)
def test_a6_stems_and_case(
    archive: FakeTdClient, conn: Conn, query: str, expected: set[tuple[int, int]]
) -> None:
    assert first_ids(search.search(conn, [query], mode="keyword")) == expected


def test_a6_forty_word_message_matches_via_the_trigram_leg(
    archive: FakeTdClient, conn: Conn
) -> None:
    assert len(FORTY.split()) == 40
    legs = {
        m.message_id >> SHIFT: m for m in search.keyword_leg(conn, "ВНЖ", None, None, None, False)
    }
    assert legs[5].trgm is not None and legs[5].trgm >= search.TRGM_THRESHOLD
    assert legs[4].trgm is not None


def test_a6_tie_does_not_match_tiene(archive: FakeTdClient, conn: Conn) -> None:
    assert search.keyword_leg(conn, "TIE", None, None, None, False) == []
    assert search.search(conn, ["TIE"], mode="keyword") == []
    tiene = search.search(conn, ["tiene"], mode="keyword")
    assert first_ids(tiene) == {(A, 6)}  # the message is searchable


def test_stop_words_do_not_go_to_the_trigram_leg(conn: Conn) -> None:
    assert search.short_tokens(conn, "ВНЖ de the и TIE renovación ВНЖ") == ["ВНЖ", "TIE"]


def test_negated_words_and_phrases_do_not_go_to_the_trigram_leg(conn: Conn) -> None:
    assert search.short_tokens(conn, 'ВНЖ -карта "TIE cita" NIE-X -"tiene hijos') == [
        "ВНЖ",
        "NIE",
        "X",
    ]
    assert search.negated('ВНЖ -карта "a -b" x-y -"tiene hijos') == 'карта OR "tiene hijos'
    assert search.negated("ВНЖ x-y") is None


def test_negated_hyphenated_compound_is_one_term() -> None:
    # Issue #20: `-a-b` negates the compound, not `a` with `-b` left over.
    assert search.negated("renewal -permit-extension") == "permit-extension"
    assert search.positive("renewal -permit-extension") == "renewal"


def test_negated_hyphenated_compound_excludes_only_the_compound(conn: Conn) -> None:
    texts = ["renewal extension", "renewal permit-extension"]
    msgs = [{"id": i, "date": (NOW - timedelta(hours=5 - i)).isoformat(), "text": t,
             "sender_name": "Ana"} for i, t in enumerate(texts, 1)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": []})
    enable(conn, td, A)
    sync(conn, td)
    hits = search.search(conn, ["renewal -permit-extension"], mode="keyword")
    assert first_ids(hits) == {(A, 1)}


def test_trigram_leg_honours_negation(conn: Conn) -> None:
    texts = ["ВНЖ карта готова", "ВНЖ отказ", "TIE y cita", "TIE y otra"]
    msgs = [{"id": i, "date": (NOW - timedelta(hours=5 - i)).isoformat(), "text": t,
             "sender_name": "Ana"} for i, t in enumerate(texts, 1)]  # fmt: skip
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": []})
    enable(conn, td, A)
    sync(conn, td)
    assert first_ids(search.search(conn, ["ВНЖ -карта"], mode="keyword")) == {(A, 2)}
    assert first_ids(search.search(conn, ['ВНЖ -"карта готова"'], mode="keyword")) == {(A, 2)}
    assert first_ids(search.search(conn, ["TIE -cita"], mode="keyword")) == {(A, 4)}
    assert first_ids(search.search(conn, ["TIE -отказ"], mode="keyword")) == {(A, 3), (A, 4)}


def test_a6_tag_filter_uses_confirmed_tags_only(archive: FakeTdClient, conn: Conn) -> None:
    assert chats.confirm_tag(conn, B, "housing")
    assert chats.propose_tag(conn, A, "docs", "agent")
    assert first_ids(search.search(conn, ["renewal"], tags=["housing"], mode="keyword")) == {(B, 1)}
    assert (
        search.search(conn, ["renewal"], tags=["docs"], mode="keyword") == []
    )  # proposed does not filter
    assert first_ids(
        search.search(conn, ["renewal"], tags=["docs", "housing"], mode="keyword")
    ) == {(B, 1)}
    assert search.search(conn, ["renewal"], tags=["housing"], chat_ids=[A], mode="keyword") == []
    assert first_ids(search.search(conn, ["renewal"], chat_ids=[A], mode="keyword")) == {(A, 3)}
    hit = search.search(conn, ["renewal"], tags=["housing"], mode="keyword")[0]
    assert hit.tags == ["housing"]  # hits list confirmed tags only
    assert search.search(conn, ["renewal"], chat_ids=[A], mode="keyword")[0].tags == []

    assert chats.propose_tag(conn, B, "housing", "agent")  # does not downgrade
    assert first_ids(search.search(conn, ["renewal"], tags=["housing"], mode="keyword")) == {(B, 1)}
    assert chats.remove_tag(conn, B, "housing")
    assert not chats.remove_tag(conn, B, "housing")
    assert search.search(conn, ["renewal"], tags=["housing"], mode="keyword") == []
    assert not chats.propose_tag(conn, -1009, "x", "agent")  # unknown chat
    with pytest.raises(ValueError, match="tag"):
        chats.confirm_tag(conn, A, "  ")


def test_a6_date_filter_by_utc_days(archive: FakeTdClient, conn: Conn) -> None:
    def ids(**kw: Any) -> set[tuple[int, int]]:
        return first_ids(search.search(conn, ["renewal"], **kw, mode="keyword"))

    assert ids(since=date(2026, 9, 21)) == {(A, 3)}
    assert ids(until=date(2026, 9, 20)) == {(B, 1)}  # inclusive: the whole day
    assert ids(since=date(2026, 9, 20), until=date(2026, 9, 20)) == {(B, 1)}
    assert ids(until=date(2026, 9, 19)) == set()
    assert ids(since=B_DATE + timedelta(minutes=1)) == {(A, 3)}  # a datetime is exact
    with pytest.raises(search.BadRange):
        search.search(
            conn, ["renewal"], since=date(2026, 9, 21), until=date(2026, 9, 20), mode="keyword"
        )


def test_a6_tombstoned_messages_only_with_include_deleted(
    archive: FakeTdClient, conn: Conn
) -> None:
    archive.next_run()
    sync(conn, archive)  # the sweep tombstones message 4
    gone = conn.execute(
        "SELECT deleted_at IS NOT NULL FROM message WHERE chat_id = %s AND server_message_id = 4",
        (A,),
    ).fetchone()
    assert gone == (True,)
    assert first_ids(search.search(conn, ["ВНЖ"], mode="keyword")) == {(A, 5)}
    hits = search.search(conn, ["ВНЖ"], include_deleted=True, mode="keyword")
    assert first_ids(hits) == {(A, 4), (A, 5)}
    single = next(h for h in hits if h.messages[0].message_id == 4 << SHIFT)
    assert single.window_id is None and len(single.messages) == 1
    assert single.start_date == single.end_date == single.messages[0].date
    assert single.messages[0].text == TEXTS[4]


def test_a_tombstoned_message_still_in_a_window_is_a_single_message_hit(
    archive: FakeTdClient, conn: Conn
) -> None:
    """Between the sweep and the window step (or during a concurrent sync)."""
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND server_message_id = 7", (A,)
    )
    assert search.search(conn, ["paella"], mode="keyword") == []
    (hit,) = search.search(conn, ["paella"], include_deleted=True, mode="keyword")
    assert hit.window_id is None and found([hit]) == [(A, 7)]


def test_a_tombstoned_neighbour_in_a_stale_window_is_left_out_of_the_window_hit(
    archive: FakeTdClient, conn: Conn
) -> None:
    """#16: window [7, 8], 7 tombstoned by a sweep whose run failed before the window step;
    a search matching live 8 must not bring back deleted 7."""
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND server_message_id = 7", (A,)
    )
    d8 = (NOW - timedelta(hours=2, minutes=50)).isoformat()
    (hit,) = search.search(conn, ["thanks"], mode="keyword")
    assert hit.window_id is not None and found([hit]) == [(A, 8)]
    assert hit.as_dict()["start_date"] == hit.as_dict()["end_date"] == d8
    (hit,) = search.search(conn, ["thanks"], include_deleted=True, mode="keyword")
    assert hit.window_id is not None and found([hit]) == [(A, 7), (A, 8)]


def test_a_single_message_hit_tombstoned_after_the_leg_is_dropped(
    archive: FakeTdClient, conn: Conn
) -> None:
    """#16: a sweep tombstones the message between the keyword leg and the hit fetch."""
    conn.execute(
        "UPDATE message SET deleted_at = now() WHERE chat_id = %s AND server_message_id = 6", (A,)
    )
    key: search.Key = ("m", A, 6 << SHIFT)
    assert search._hit(conn, key, 1.0, [(A, 6 << SHIFT)], ["tiene"], False) is None
    hit = search._hit(conn, key, 1.0, [(A, 6 << SHIFT)], ["tiene"], True)
    assert hit is not None and found([hit]) == [(A, 6)]


# --- hits -----------------------------------------------------------------------------------


def test_hit_shape_maps_messages_to_their_window(archive: FakeTdClient, conn: Conn) -> None:
    hits = search.search(conn, ["paella"], mode="keyword")
    assert len(hits) == 1
    hit = hits[0]
    window = conn.execute(
        'SELECT window_id, start_date, end_date FROM "window" WHERE first_message_id = %s',
        (7 << SHIFT,),
    ).fetchone()
    assert window is not None
    d7, d8 = NOW - timedelta(hours=3), NOW - timedelta(hours=2, minutes=50)
    assert hit.as_dict() == {
        "window_id": window[0],
        "chat_id": A,
        "chat_title": f"chat {A}",
        "tags": [],
        "start_date": d7.isoformat(),
        "end_date": d8.isoformat(),
        "score": 1 / 61,
        "messages": [
            {
                "message_id": 7 << SHIFT,
                "sender_name": "Ana",
                "date": d7.isoformat(),
                "text": TEXTS[7],
                "link": "https://t.me/c/1000000101/7",
            },
            {
                "message_id": 8 << SHIFT,
                "sender_name": "Ana",
                "date": d8.isoformat(),
                "text": TEXTS[8],
                "link": "https://t.me/c/1000000101/8",
            },
        ],
        "highlights": [TEXTS[7]],
        "matched_by": ["keyword", "trigram"],
        "matched_message_ids": [7 << SHIFT],
        "matched_queries": [0],
        "semantic_distance": None,
        "omitted_messages": 0,
    }
    assert (window[1], window[2]) == (hit.start_date, hit.end_date)


def test_long_highlight_is_cut_around_the_match(archive: FakeTdClient, conn: Conn) -> None:
    hit = next(
        h
        for h in search.search(conn, ["ВНЖ"], mode="keyword")
        if h.messages[0].message_id == 5 << SHIFT
    )
    (highlight,) = hit.highlights
    assert "ВНЖ," in highlight and len(highlight) <= search.HIGHLIGHT_CHARS + 2
    assert highlight.endswith("…") and highlight.strip("…") in FORTY


def test_queries_are_fused_by_reciprocal_rank(archive: FakeTdClient, conn: Conn) -> None:
    hits = search.search(conn, ["renewal", "paella", "thanks"], mode="keyword")
    assert found(hits)[:2] == [(A, 7), (A, 8)]  # the window matched by two queries leads
    assert hits[0].score == pytest.approx(2 / 61)
    assert sorted(hits[0].highlights) == sorted([TEXTS[7], TEXTS[8]])
    assert [h.score for h in hits[1:]] == pytest.approx([1 / 61, 1 / 62])
    assert len(search.search(conn, ["renewal", "paella", "thanks"], limit=1, mode="keyword")) == 1


def test_a_matching_message_without_a_window_is_a_single_message_hit(
    archive: FakeTdClient, conn: Conn
) -> None:
    conn.execute('DELETE FROM "window" WHERE first_message_id = %s', (6 << SHIFT,))
    (hit,) = search.search(conn, ["tiene"], mode="keyword")
    assert hit.window_id is None and found([hit]) == [(A, 6)]


@pytest.mark.parametrize(
    ("kw", "error"),
    [
        ({"limit": 0}, ValueError),
        ({"limit": search.MAX_LIMIT + 1}, ValueError),
        ({"mode": "fuzzy"}, ValueError),
    ],
)
def test_bad_arguments(conn: Conn, kw: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        search.search(conn, ["x"], **kw)
    with pytest.raises(ValueError, match="query"):
        search.search(conn, ["  ", ""])


def test_trigram_leg_uses_the_gin_index(archive: FakeTdClient, conn: Conn) -> None:
    stmt, params = search.leg_query(conn, "ВНЖ", None, None, None, False)
    with conn.transaction():
        conn.execute("SET LOCAL enable_seqscan = off")
        conn.execute("SET LOCAL pg_trgm.strict_word_similarity_threshold = 0.5")
        plan = "\n".join(str(r[0]) for r in conn.execute(sql.SQL("EXPLAIN ") + stmt, params))
    assert "message_text_trgm_idx" in plan and "<<%" in plan


# --- CLI ------------------------------------------------------------------------------------


@pytest.fixture
def cli(db_url: str, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    monkeypatch.setenv("DATABASE_URL", db_url)
    return CliRunner()


def test_cli_search(archive: FakeTdClient, conn: Conn, cli: CliRunner) -> None:
    out = cli.invoke(app, ["search", "--mode", "keyword", "paella"])
    assert out.exit_code == 0, out.stderr
    lines = out.stdout.splitlines()
    assert lines[0].startswith("0.0164  chat -1001000000101  2026-10-01 09:00  (window ")
    assert lines[1:] == [
        f"  2026-10-01 09:00 Ana: {TEXTS[7]}", "    https://t.me/c/1000000101/7",
        f"  2026-10-01 09:10 Ana: {TEXTS[8]}", "    https://t.me/c/1000000101/8",
    ]  # fmt: skip

    as_json = cli.invoke(
        app, ["search", "--mode", "keyword", "renewal", "paella", "--json", "--limit", "2"]
    )
    assert as_json.exit_code == 0, as_json.stderr
    expected = search.search(conn, ["renewal", "paella"], limit=2, mode="keyword")
    assert json.loads(as_json.stdout) == [h.as_dict() for h in expected]

    chats.confirm_tag(conn, B, "housing")
    tagged = cli.invoke(
        app, ["search", "--mode", "keyword", "renewal", "--tag", "housing", "--json"]
    )
    assert [h["chat_id"] for h in json.loads(tagged.stdout)] == [B]
    dated = cli.invoke(
        app, ["search", "--mode", "keyword", "renewal", "--until", "2026-09-20", "--json"]
    )
    assert [h["chat_id"] for h in json.loads(dated.stdout)] == [B]
    by_chat = cli.invoke(
        app, ["search", "--mode", "keyword", "renewal", "--chat", str(A), "--json"]
    )
    assert [h["chat_id"] for h in json.loads(by_chat.stdout)] == [A]
    assert cli.invoke(app, ["search", "--mode", "keyword", "nothingmatches"]).stdout == ""

    bad_range = cli.invoke(app, ["search", "x", "--since", "2026-09-21", "--until", "2026-09-20"])
    assert bad_range.exit_code == 1 and "since is after until" in bad_range.stderr
    assert cli.invoke(app, ["search", "x", "--since", "21/09/2026"]).exit_code == 1
    assert cli.invoke(app, ["search", "x", "--limit", "51"]).exit_code == 1


def test_cli_tags(archive: FakeTdClient, conn: Conn, cli: CliRunner) -> None:
    def tags() -> list[tuple[Any, ...]]:
        return conn.execute("SELECT chat_id, tag, status FROM chat_tag ORDER BY 1, 2").fetchall()

    out = cli.invoke(app, ["tags", "propose", str(B), " housing "])
    assert (out.exit_code, out.stdout.strip()) == (0, f"proposed housing for {B}")
    assert tags() == [(B, "housing", "proposed")]
    out = cli.invoke(app, ["tags", "confirm", str(B), "housing"])
    assert (out.exit_code, out.stdout.strip()) == (0, f"confirmed housing for {B}")
    assert tags() == [(B, "housing", "confirmed")]
    out = cli.invoke(app, ["tags", "remove", str(B), "housing"])
    assert (out.exit_code, out.stdout.strip()) == (0, f"removed housing from {B}")
    assert tags() == []

    missing = cli.invoke(app, ["tags", "remove", str(B), "housing"])
    assert missing.exit_code == 1 and f"chat {B} has no tag housing" in missing.stderr
    unknown = cli.invoke(app, ["tags", "confirm", "-1009", "x"])
    assert unknown.exit_code == 1 and "unknown chat -1009" in unknown.stderr
    assert cli.invoke(app, ["tags", "propose", str(B), " "]).exit_code == 1
    assert tags() == []
