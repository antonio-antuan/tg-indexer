# ruff: noqa: S311
# mypy: disable-error-code="unused-ignore,index,arg-type,misc,type-arg,no-untyped-def,return-value"
# ruff: noqa: E501
"""QA P2.T2: filtered recall of semantic_leg on clustered data with HNSW in place (chats correlate with clusters)."""

import math
import os
import random
from datetime import timedelta

import psycopg
import psycopg.sql
import pytest
from test_ingest import NOW

from tgi import embed, search

Conn = psycopg.Connection[tuple[object, ...]]
DIM, NCHAT, PER = 1024, 60, 60


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@pytest.fixture
def corpus(conn: Conn) -> list[list[float]]:
    rnd = random.Random(7)
    centers = [unit([rnd.gauss(0, 1) for _ in range(DIM)]) for _ in range(NCHAT)]
    for c in range(NCHAT):
        cid = -1000000000000 - c
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind, enabled) VALUES (%s,'+0','c','supergroup',true)",
            (cid,),
        )
        if c % 6 == 0:
            conn.execute(
                "INSERT INTO chat_tag (chat_id, tag, status, proposed_by) VALUES (%s,'t','confirmed','user')",
                (cid,),
            )
        with conn.transaction(), conn.cursor() as cur:
            for i in range(1, PER + 1):
                when = NOW - timedelta(days=c * 3, hours=PER - i)
                v = unit([a + rnd.gauss(0, 0.12) for a in centers[c]])
                cur.execute(
                    "INSERT INTO message (chat_id,message_id,server_message_id,date,text) VALUES (%s,%s,%s,%s,'x')",
                    (cid, i << 20, i, when),
                )
                cur.execute(
                    'INSERT INTO "window" (chat_id,first_message_id,start_date,end_date,message_ids,text,stale) VALUES (%s,%s,%s,%s,%s,%s,false) RETURNING window_id',
                    (cid, i << 20, when, when, [i << 20], "x"),
                )
                wid = cur.fetchone()[0]  # type: ignore[index]
                cur.execute(
                    "INSERT INTO window_embedding (window_id,model,dim,embedding) VALUES (%s,'fake',1024,%s::vector)",
                    (wid, embed._literal(v)),
                )
    conn.execute("ANALYZE")
    return centers


def exact(conn: Conn, vec: list[float], chats: list[int] | None, since, until, n: int) -> list[int]:
    stmt, params = search.semantic_query(vec, chats, since, until)
    with conn.transaction():
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")
        conn.execute("SET LOCAL enable_seqscan = on")
        rows = conn.execute(stmt, params).fetchall()
    return [r[0] for r in sorted(rows, key=lambda r: (r[1], r[0]))][:n]  # type: ignore[type-var]


@pytest.mark.skipif(
    not os.environ.get("QA_SCALE"), reason="slow (~40 s): QA_SCALE=1; QA_FORCE=1 forces HNSW plans"
)
def test_filtered_recall(
    conn: Conn, corpus: list[list[float]], capsys: pytest.CaptureFixture[str]
) -> None:
    if os.environ.get("QA_FORCE"):
        conn.execute("SET enable_seqscan = off")
        conn.execute("SET enable_sort = off")
    rnd = random.Random(3)
    rows = []
    bad = 0
    cases = []
    for k in range(25):
        q = unit([a + rnd.gauss(0, 0.12) for a in corpus[k]])
        far = [-1000000000000 - ((k + 17) % NCHAT)]  # a chat far from the query cluster
        near = [-1000000000000 - k]
        many = [-1000000000000 - c for c in range(0, NCHAT, 3)]
        cases += [
            (q, far, None, None),
            (q, near, None, None),
            (q, many, None, None),
            (q, None, NOW.date() - timedelta(days=60), NOW.date() - timedelta(days=30)),
            (q, None, None, NOW.date() - timedelta(days=170)),
        ]
    for q, chats, since, until in cases:
        for limit in (10, 50):
            with conn.transaction():
                got, ex = search.semantic_leg(conn, q, chats, since, until, limit=limit)
            want = exact(conn, q, chats, since, until, 100)
            gids = [g for g, _ in got]
            n = min(limit, len(want))
            rec = len(set(gids[:limit]) & set(want[:limit])) / max(1, n)
            ordered = gids == [g for g, _ in sorted(got, key=lambda r: (r[1], r[0]))]
            rows.append((rec, ex, len(got) >= n, ordered))
            if len(gids) < n or not ordered:
                bad += 1
    avg = sum(r[0] for r in rows) / len(rows)
    print(
        f"cases={len(rows)} mean recall@limit={avg:.3f} min={min(r[0] for r in rows):.2f} rerun={sum(r[1] for r in rows)} short={bad}"
    )
    print("recall<1:", [(round(r[0], 2), r[1]) for r in rows if r[0] < 1][:20])
    assert bad == 0
    assert avg > 0.9
