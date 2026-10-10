# ruff: noqa: E501
"""QA T10: negation edge cases for the keyword search (the trigram leg and the FTS leg)."""

import contextlib

import psycopg
import pytest
from test_qa_t9 import build, ids

Conn = psycopg.Connection[tuple[object, ...]]

TEXTS = [  # server ids 1..13
    "ВНЖ карта готова", "ВНЖ отказ", "TIE y cita", "TIE y otra", "tiene hijos y TIE",
    "карта-готова ВНЖ", "продление карты", "продлении ВНЖ", "renewal of TIE card", "renewals only",
    "NIE cita previa", "ВНЖ", "x",
]  # fmt: skip


def ok(conn: Conn, q: str) -> set[int]:
    return set(ids(conn, q, limit=50))


@pytest.fixture
def corpus(conn: Conn) -> None:
    build(conn, TEXTS, gap_h=2)


def test_trigram_leg_negation_edge_cases(conn: Conn, corpus: None) -> None:
    assert ok(conn, "ВНЖ -карта") == {2, 8, 12}  # 1 and the hyphenated 6 are excluded
    assert ok(conn, 'ВНЖ -"карта готова"') == {2, 8, 12}  # the phrase also matches the hyphenated 6
    assert ok(conn, 'ВНЖ -"карта готова" -отказ') == {8, 12}  # a mix
    assert ok(conn, "TIE -cita -otra") >= {5, 9} and 3 not in ok(
        conn, "TIE -cita -otra"
    )  # 4 stays: otra is an es stop word (known limit)
    assert (
        ok(conn, "ВНЖ -ВНЖ") == set() and ok(conn, "TIE -TIE") == set()
    )  # negating the short query token
    assert ok(conn, "NIE -TIE") == {11}
    assert ok(conn, "ВНЖ - карта") >= {1, 2}  # a lone dash negates nothing
    assert 1 in ok(conn, 'ВНЖ "карта готова"') and 2 in ok(conn, 'ВНЖ "карта готова"')
    for q in [
        "-",
        "--",
        "- -",
        '-"',
        '-""',
        "ВНЖ -",
        "ВНЖ --карта",
        'ВНЖ -"unclosed',
        "ВНЖ -карта-готова",
    ]:
        with contextlib.suppress(ValueError):  # an empty query is rejected; anything else is a bug
            ok(conn, q)


def test_negation_only_query(conn: Conn, corpus: None) -> None:
    got = ok(conn, "-отказ")
    assert 2 not in got and {1, 3, 12} <= got  # everything without the word


def test_stop_word_limitation_is_as_documented(conn: Conn, corpus: None) -> None:
    """`TIE -tiene`: `tiene` is a Spanish stop word, but the Russian and English configs keep it,
    and a negated term excludes through any config (#18 fix): message 5 is excluded.
    (Coder, T10-fix2: was `5 in`, the limitation before the fix.)"""
    assert 5 not in ok(conn, "TIE -tiene")
    assert 5 not in ok(conn, "-tiene")
    assert 5 in ok(conn, "TIE -a")  # `a` is dropped by all three configs: it negates nothing


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("renewal -renewals", set()),  # english stem renew is in 9 and 10
        ("ВНЖ -карты", {2, 8, 12}),  # ru stem карт is in 1 and 6
        ("продление -продлении", set()),  # same russian stem
    ],
)
def test_fts_leg_negation_excludes_by_stem(
    conn: Conn, corpus: None, query: str, expected: set[int]
) -> None:
    assert ok(conn, query) == expected
