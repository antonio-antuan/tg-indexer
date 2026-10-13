"""`recent` ("what's new" without a topic) and date-time bounds on `search`."""

import base64
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_embed import sync
from test_ingest import NOW, A, B, C, chat, enable
from test_mcp import in_process
from typer.testing import CliRunner

from tgi import mcp, search
from tgi.cli import app
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
H = timedelta(hours=1)
M = timedelta(minutes=1)


def at(i: int, when: datetime, text: str = "news") -> dict[str, Any]:
    return {"id": i, "date": when.isoformat(), "text": f"{text} {i}", "sender_name": "S"}


# Windows (30-min gap): A [1] [2] [3,4] [5,6] [7,8], B [1] [2], C [1]. A3/A4 straddle NOW-24h.
A_MSGS = [at(1, NOW - 72 * H), at(2, NOW - 30 * H), at(3, NOW - 24 * H - 10 * M),
          at(4, NOW - 24 * H + 10 * M), at(5, NOW - 5 * H - 10 * M), at(6, NOW - 5 * H, "rent"),
          at(7, NOW - H, "rent"), at(8, NOW - 50 * M)]  # fmt: skip
B_MSGS = [at(1, NOW - 25 * H), at(2, NOW - 2 * H)]
C_MSGS = [at(1, NOW - 30 * M), at(2, NOW + 10 * M)]  # C2 is after `now` in these tests


@pytest.fixture
def archive(conn: Conn) -> None:
    td = FakeTdClient({
        "chats": [chat(A, A_MSGS, username="news_a"), chat(B, B_MSGS), chat(C, C_MSGS)],
        "runs": [],
    })  # fmt: skip
    enable(conn, td, A, B, C)
    assert sync(conn, td).status == "ok"
    for chat_id, tag in ((A, "valencia"), (B, "valencia"), (C, "madrid")):
        conn.execute(
            "INSERT INTO chat_tag (chat_id, tag, status) VALUES (%s, %s, 'confirmed')",
            (chat_id, tag),
        )


def ids(hits: list[search.Hit]) -> list[tuple[int, list[int]]]:
    return [(h.chat_id, [m.message_id >> SHIFT for m in h.messages]) for h in hits]


def dict_ids(hits: list[dict[str, Any]]) -> list[tuple[int, list[int]]]:
    return [(h["chat_id"], [m["message_id"] >> SHIFT for m in h["messages"]]) for h in hits]


DAY = [(C, [1]), (A, [7, 8]), (B, [2]), (A, [5, 6]), (A, [4])]  # the last 24 h, newest first


# --- search.recent ------------------------------------------------------------------------------


def test_recent_lists_the_range_newest_first_with_only_messages_inside(
    archive: None, conn: Conn
) -> None:
    answer = search.recent(conn, since=NOW - 24 * H, now=NOW)
    assert ids(answer.hits) == DAY  # A3 is outside the range although its window is not
    assert answer.next_cursor is None
    straddling = answer.hits[-1]
    assert straddling.start_date == straddling.end_date == NOW - 24 * H + 10 * M
    assert all(m.link for h in answer.hits if h.chat_id == A for m in h.messages)
    assert answer.as_dict()["now"] == NOW.isoformat()


def test_recent_filters_by_confirmed_tag_and_chat(archive: None, conn: Conn) -> None:
    valencia = search.recent(conn, since=NOW - 24 * H, tags=["valencia"], now=NOW)
    assert ids(valencia.hits) == [h for h in DAY if h[0] != C]
    only_b = search.recent(conn, since=NOW - 24 * H, chat_ids=[B], now=NOW)
    assert ids(only_b.hits) == [(B, [2])]
    assert search.recent(conn, since=NOW - 24 * H, tags=["nowhere"], now=NOW).hits == []


def test_recent_pages_with_a_cursor_without_gaps_or_repeats(archive: None, conn: Conn) -> None:
    pages, cursor = [], None
    while True:
        answer = search.recent(conn, since=NOW - 24 * H, limit=2, cursor=cursor, now=NOW)
        pages.append(ids(answer.hits))
        cursor = answer.next_cursor
        if cursor is None:
            break
    assert pages == [DAY[:2], DAY[2:4], DAY[4:]]


def test_recent_until_and_dates(archive: None, conn: Conn) -> None:
    window = search.recent(conn, since=NOW - 26 * H, until=NOW - 4 * H, now=NOW)
    assert ids(window.hits) == [(A, [5, 6]), (A, [3, 4]), (B, [1])]
    day = search.recent(conn, since=date(2026, 9, 30), until=date(2026, 9, 30), now=NOW)
    assert ids(day.hits) == [(A, [3, 4]), (B, [1]), (A, [2])]  # all of 30 September (UTC)
    assert day.as_dict()["until"] == "2026-10-01T00:00:00+00:00"  # exclusive end


def test_recent_reads_a_naive_datetime_as_utc(archive: None, conn: Conn) -> None:
    naive = (NOW - 24 * H).replace(tzinfo=None)
    assert ids(search.recent(conn, since=naive, now=NOW).hits) == DAY


def test_recent_rejects_bad_arguments(archive: None, conn: Conn) -> None:
    with pytest.raises(search.BadRange):
        search.recent(conn, since=NOW, until=NOW - H, now=NOW)
    with pytest.raises(ValueError, match="limit"):
        search.recent(conn, since=NOW - H, limit=0, now=NOW)
    for cursor in ("nonsense", "2026-10-01T00:00:00/5", "eyJzaW5jZSI6IjEifQ=="):
        with pytest.raises(ValueError, match="cursor"):
            search.recent(conn, cursor=cursor, now=NOW)
    with pytest.raises(ValueError, match="since"):
        search.recent(conn, now=NOW)


# --- the MCP tool -------------------------------------------------------------------------------


def test_recent_tool_defaults_to_the_last_24_hours_of_the_server_clock(
    archive: None, conn: Conn
) -> None:
    answer = mcp.run_recent(conn, now=NOW)
    assert dict_ids(answer["hits"]) == DAY
    assert (answer["now"], answer["since"]) == (NOW.isoformat(), (NOW - 24 * H).isoformat())
    assert answer["until"] == NOW.isoformat() and answer["next_cursor"] is None
    assert dict_ids(mcp.run_recent(conn, hours=3, now=NOW)["hits"]) == DAY[:3]


def test_recent_tool_over_mcp(archive: None, db_url: str) -> None:
    span = {"since": (NOW - 24 * H).isoformat(), "until": NOW.isoformat()}
    first, second, tagged = in_process(db_url, [
        ("recent", {**span, "limit": 3}),
        ("recent", {"limit": 3, "cursor": "PLACEHOLDER"}),
        ("recent", {**span, "tags": ["madrid"]}),
    ])  # fmt: skip
    assert dict_ids(first["hits"]) == DAY[:3]
    assert first["next_cursor"] is not None
    assert second["error"]["code"] == "bad_argument"  # not a cursor the server returned
    (rest,) = in_process(db_url, [("recent", {"limit": 3, "cursor": first["next_cursor"]})])
    assert dict_ids(rest["hits"]) == DAY[3:] and rest["next_cursor"] is None
    assert dict_ids(tagged["hits"]) == [(C, [1])]


@pytest.mark.parametrize(
    ("args", "code"),
    [
        ({"hours": 3, "since": "2026-10-01"}, "bad_argument"),
        ({"until": "2026-10-01"}, "bad_argument"),
        ({"hours": 0}, "bad_range"),
        ({"hours": 745}, "bad_range"),
        ({"since": "2026-10-02", "until": "2026-10-01"}, "bad_range"),
        ({"limit": 51}, "bad_range"),
        ({"tags": ["nowhere"]}, "unknown_tag"),
        ({"chat_ids": [123]}, "unknown_chat"),
        ({"hours": True}, "bad_argument"),
    ],
)
def test_recent_tool_errors(archive: None, db_url: str, args: dict[str, Any], code: str) -> None:
    (result,) = in_process(db_url, [("recent", args)])
    assert result["error"]["code"] == code


# --- date-time bounds on search -----------------------------------------------------------------


def test_search_takes_a_date_time_as_an_exact_bound(archive: None, db_url: str) -> None:
    day, exact, naive = in_process(db_url, [
        ("search", {"queries": ["rent"], "mode": "keyword", "since": "2026-10-01"}),
        ("search", {"queries": ["rent"], "mode": "keyword", "since": "2026-10-01T10:00:00Z"}),
        ("search", {"queries": ["rent"], "mode": "keyword", "since": "2026-10-01T10:00:00"}),
    ])  # fmt: skip
    assert sorted(dict_ids(day)) == [(A, [5, 6]), (A, [7, 8])]  # A6 07:00 and A7 11:00 UTC
    assert dict_ids(exact) == [(A, [7, 8])]
    assert naive == exact  # no zone: UTC


# --- tgi recent ---------------------------------------------------------------------------------


def test_cli_recent(
    archive: None, db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    since, until = (NOW - 24 * H).isoformat(), NOW.isoformat()
    out = runner.invoke(app, ["recent", "--since", since, "--until", until, "--json"])
    assert out.exit_code == 0, out.output
    assert dict_ids(json.loads(out.output)["hits"]) == DAY
    paged = runner.invoke(app, ["recent", "--since", since, "--until", until, "--limit", "2"])
    assert paged.exit_code == 0 and "more: tgi recent --cursor " in paged.output
    cursor = paged.output.rsplit("--cursor ", 1)[1].strip()
    rest = runner.invoke(app, ["recent", "--cursor", cursor, "--hours", "1", "--json"])
    assert dict_ids(json.loads(rest.output)["hits"]) == DAY[2:]  # the cursor's range wins
    tagged = runner.invoke(app, ["recent", "--since", since, "--tag", "madrid"])
    assert f"chat {C}" in tagged.output and f"chat {A}" not in tagged.output
    for bad in (["--hours", "2", "--since", since], ["--until", since],
                ["--since", "yesterday"], ["--cursor", "x"]):  # fmt: skip
        assert runner.invoke(app, ["recent", *bad]).exit_code == 1, bad


# --- range edges (review round 1) -------------------------------------------------------------


def test_until_defaults_to_now_and_is_exclusive(archive: None, conn: Conn) -> None:
    assert (C, [2]) not in ids(search.recent(conn, since=NOW - H, now=NOW).hits)
    later = search.recent(conn, since=NOW - H, now=NOW + H)
    assert ids(later.hits)[0] == (C, [2])
    a8 = NOW - 50 * M
    assert ids(search.recent(conn, since=NOW - 2 * H, until=a8, now=NOW).hits) == [
        (A, [7]), (B, [2])
    ]  # fmt: skip
    just_after = a8 + timedelta(microseconds=1)
    assert ids(search.recent(conn, since=NOW - 2 * H, until=just_after, now=NOW).hits)[0] == (
        A, [7, 8]
    )  # fmt: skip


def test_consecutive_ranges_meet_without_gap_or_overlap(archive: None, conn: Conn) -> None:
    def messages(since: datetime, until: datetime) -> list[tuple[int, int]]:
        hits = search.recent(conn, since=since, until=until, now=NOW).hits
        return [(h.chat_id, m.message_id >> SHIFT) for h in hits for m in h.messages]

    split = NOW - 5 * H  # A6 is exactly here: it belongs to the second range only
    first, second = messages(NOW - 24 * H, split), messages(split, NOW)
    assert (A, 6) in second and (A, 6) not in first
    assert sorted(first + second) == sorted((c, i) for c, ms in DAY for i in ms)


def test_a_cursor_keeps_its_range_and_filters(archive: None, conn: Conn) -> None:
    first = mcp.run_recent(conn, tags=["valencia"], limit=2, now=NOW)
    later = mcp.run_recent(conn, cursor=first["next_cursor"], hours=1, tags=["madrid"],
                           limit=2, now=NOW + 3 * H)  # fmt: skip
    assert dict_ids(first["hits"]) == [h for h in DAY if h[0] != C][:2]
    assert dict_ids(later["hits"]) == [h for h in DAY if h[0] != C][2:]
    assert (later["since"], later["until"]) == (first["since"], first["until"])


def test_paging_ends_with_an_empty_page_when_the_last_one_is_full(
    archive: None, conn: Conn
) -> None:
    full = search.recent(conn, since=NOW - 24 * H, limit=len(DAY), now=NOW)
    assert ids(full.hits) == DAY and full.next_cursor is not None
    end = search.recent(conn, cursor=full.next_cursor, now=NOW)
    assert end.hits == [] and end.next_cursor is None


def test_a_page_of_windows_without_in_range_messages_is_empty_but_continues(
    archive: None, conn: Conn
) -> None:
    edge = NOW - 24 * H  # A3 and A4 sit 10 minutes either side; their window overlaps
    page = search.recent(conn, since=edge - 5 * M, until=edge + 5 * M, limit=1, now=NOW)
    assert page.hits == [] and page.next_cursor is not None
    assert search.recent(conn, cursor=page.next_cursor, now=NOW).next_cursor is None


def test_tied_end_dates_page_without_repeats(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, [at(1, NOW - H)]), chat(B, [at(1, NOW - H)]),
                                 chat(C, [at(1, NOW - H)])], "runs": []})  # fmt: skip
    enable(conn, td, A, B, C)
    assert sync(conn, td).status == "ok"
    seen, cursor = [], None
    while True:
        page = search.recent(conn, since=NOW - 2 * H, limit=1, cursor=cursor, now=NOW)
        seen += [h.chat_id for h in page.hits]
        if (cursor := page.next_cursor) is None:
            break
    assert sorted(seen) == sorted([A, B, C])


def test_naive_bounds_are_utc_whatever_the_session_time_zone(archive: None, conn: Conn) -> None:
    conn.execute("SET TIME ZONE 'Asia/Tokyo'")
    naive = (NOW - 24 * H).replace(tzinfo=None)
    assert ids(search.recent(conn, since=naive, now=NOW).hits) == DAY
    hits = search.search(conn, ["rent"], mode="keyword", since=(NOW - 2 * H).replace(tzinfo=None))
    assert ids(hits) == [(A, [7, 8])]


def test_a_midnight_date_time_is_an_instant_not_a_day(archive: None, db_url: str) -> None:
    whole_day, midnight, recent_midnight = in_process(db_url, [
        ("search", {"queries": ["news"], "mode": "keyword", "until": "2026-09-30"}),
        ("search", {"queries": ["news"], "mode": "keyword", "until": "2026-09-30T00:00:00"}),
        ("recent", {"since": "2026-09-28T00:00:00Z", "until": "2026-10-01T00:00:00Z"}),
    ])  # fmt: skip
    assert {(A, 2), (A, 3), (B, 1)} <= {
        (h["chat_id"], m["message_id"] >> SHIFT) for h in whole_day for m in h["messages"]
    }  # 30 September counts with a date
    assert dict_ids(midnight) == [(A, [1])]  # only 28 September, before that instant
    assert recent_midnight["until"] == "2026-10-01T00:00:00+00:00"
    assert dict_ids(recent_midnight["hits"]) == [(A, [3, 4]), (B, [1]), (A, [2]), (A, [1])]


# --- review round 2 ------------------------------------------------------------------------------


def forged(**fields: Any) -> str:
    state = {"since": "2026-09-30T12:00:00+00:00", "until": "2026-10-01T12:00:00+00:00",
             "tags": None, "chat_ids": None, "end": "2026-10-01T11:00:00+00:00",
             "window": 5, **fields}  # fmt: skip
    return base64.urlsafe_b64encode(json.dumps(state).encode()).decode()


@pytest.mark.parametrize(
    "cursor",
    [
        forged(tags={"a": 1}),
        forged(tags="valencia"),
        forged(tags=[1]),
        forged(chat_ids={"a": 1}),
        forged(chat_ids=[True]),
        forged(window=True),
        forged(window="5"),
        forged(end="2026-10-01T11:00:00"),  # no zone
        base64.urlsafe_b64encode(b"[1, 2]").decode(),
        base64.urlsafe_b64encode(b"\xff").decode(),
    ],
)
def test_a_forged_cursor_is_refused_before_the_database(
    archive: None, conn: Conn, db_url: str, cursor: str
) -> None:
    with pytest.raises(ValueError, match="cursor"):
        search.recent(conn, cursor=cursor, now=NOW)
    (result,) = in_process(db_url, [("recent", {"cursor": cursor})])
    assert result["error"]["code"] == "bad_argument"


def test_a_well_formed_cursor_is_accepted(archive: None, conn: Conn) -> None:
    assert search.recent(conn, cursor=forged(tags=["valencia"], chat_ids=[A]), now=NOW).hits


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-10-01", date(2026, 10, 1)),
        (" 2026-10-01 ", date(2026, 10, 1)),
        ("2026-10-01T00:00:00", datetime(2026, 10, 1)),
        ("2026-10-01 10:00", datetime(2026, 10, 1, 10, 0)),
        ("2026-10-01T10:00:00Z", datetime(2026, 10, 1, 10, 0, tzinfo=NOW.tzinfo)),
    ],
)
def test_parse_when(text: str, expected: date) -> None:
    value = search.parse_when(text)
    assert value == expected and type(value) is type(expected)


@pytest.mark.parametrize("bad", ["", "   ", "yesterday", "2026-13-01", "2026-10-01T25:00"])
def test_parse_when_refuses(bad: str) -> None:
    with pytest.raises(ValueError):
        search.parse_when(bad)


@pytest.mark.parametrize("since", [1727784000, 1727784000.5, True, ["2026-10-01"]])
def test_a_non_string_date_is_bad_argument(archive: None, db_url: str, since: Any) -> None:
    search_result, recent_result = in_process(db_url, [
        ("search", {"queries": ["news"], "since": since}),
        ("recent", {"since": since}),
    ])  # fmt: skip
    assert search_result["error"]["code"] == recent_result["error"]["code"] == "bad_argument"


def test_cli_recent_cursor_ignores_bounds_and_takes_a_space_date_time(
    archive: None, db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)
    runner = CliRunner()
    first = runner.invoke(
        app, ["recent", "--since", "2026-09-30 12:00", "--until", "2026-10-01 12:00",
              "--limit", "2", "--json"],
    )  # fmt: skip
    assert first.exit_code == 0, first.output
    cursor = json.loads(first.output)["next_cursor"]
    rest = runner.invoke(app, ["recent", "--cursor", cursor, "--until", "garbage", "--json"])
    assert rest.exit_code == 0, rest.output
    assert dict_ids(json.loads(rest.output)["hits"]) == DAY[2:]
