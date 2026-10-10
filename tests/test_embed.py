"""T8: the embed step of `tgi sync` (SPEC §5.3 step 6), the fake provider and `tgi embed`."""

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta
from typing import Any

import psycopg
import pytest
from test_ingest import NOW, SETTINGS, A, FakeClock, chat, enable
from test_sweep import DELETED_IN, EDITED, a3_changes, history
from typer.testing import CliRunner

from tgi import embed, ingest
from tgi.cli import app
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20


def sync(conn: Conn, td: FakeTdClient, embedder: embed.Embedder | None = None) -> Any:
    settings = replace(SETTINGS, page_delay_ms=0)
    return ingest.sync(conn, lambda _: td, settings, FakeClock(), embedder=embedder)


def hourly(n: int, text: str = "message {i}") -> list[dict[str, Any]]:
    """n messages an hour apart: n single-message windows."""
    return [{"id": i, "date": (NOW - timedelta(hours=n - i)).isoformat(),
             "text": text.format(i=i)} for i in range(1, n + 1)]  # fmt: skip


def embeddings(conn: Conn) -> dict[int, tuple[Any, ...]]:
    """first server id → (model, dim, embedding text, embedded_at)."""
    rows = conn.execute(
        "SELECT w.first_message_id, e.model, e.dim, e.embedding::text, e.embedded_at"
        ' FROM "window" w JOIN window_embedding e USING (window_id)'
    )
    return {r[0] >> SHIFT: tuple(r[1:]) for r in rows}


def stale(conn: Conn) -> list[int]:
    rows = conn.execute('SELECT first_message_id FROM "window" WHERE stale ORDER BY 1')
    return [r[0] >> SHIFT for r in rows]


def count(conn: Conn, table: str) -> int:
    return conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]  # type: ignore[index,no-any-return]


class Recording(FakeEmbedder):
    """The fake provider, recording each batch size."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.batches: list[int] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(len(texts))
        return super().embed_documents(texts)


# --- the fake provider ---------------------------------------------------------------------


def test_fake_vectors_are_deterministic_unit_and_token_based() -> None:
    a, b = FakeEmbedder(), FakeEmbedder()
    texts = ["Valencia paella recipe", "paella in Valencia", "train timetable", ""]
    va, vb = a.embed_documents(texts), b.embed_queries(texts)
    assert va == vb and (a.model, a.dim) == ("fake", 1024)
    assert all(len(v) == 1024 and math.isclose(sum(x * x for x in v), 1.0) for v in va)

    def cos(x: list[float], y: list[float]) -> float:
        return sum(p * q for p, q in zip(x, y, strict=True))

    assert cos(va[0], va[1]) > 0.5 > abs(cos(va[0], va[2]))


# --- the embed step in sync ----------------------------------------------------------------


def test_sync_embeds_every_window_in_batches_of_64(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, hourly(150))], "runs": [{}]})
    enable(conn, td, A)
    rec = Recording()
    result = sync(conn, td, rec)
    assert (result.status, result.stats["windows_rebuilt"], result.stats["embedded"]) == (
        "ok", 150, 150)  # fmt: skip
    assert rec.batches == [64, 64, 22]
    rows = embeddings(conn)
    assert len(rows) == 150 and {r[:2] for r in rows.values()} == {("fake", 1024)}
    assert stale(conn) == []
    row = conn.execute('SELECT text FROM "window" WHERE first_message_id = %s', (7 << SHIFT,))
    expected = FakeEmbedder().embed_documents([row.fetchone()[0]])[0]  # type: ignore[index]
    assert [float(x) for x in rows[7][2][1:-1].split(",")] == pytest.approx(expected, abs=1e-6)

    td.next_run()
    rec.batches.clear()
    assert sync(conn, td, rec).stats["embedded"] == 0 and rec.batches == []  # nothing stale


def test_failed_batch_stays_stale_run_is_ok_and_next_run_embeds_it(
    conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    msgs = hourly(150, "SECRET {i}")
    msgs[99]["text"] = "SECRET boom"  # message 100: window 100, in the second batch (65-128)
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{}]})
    enable(conn, td, A)
    caplog.set_level("DEBUG")
    result = sync(conn, td, FakeEmbedder(fail=lambda t: "boom" in t))
    assert (result.status, result.stats["embedded"], result.stats["embed_failed"]) == (
        "ok", 86, 64)  # fmt: skip
    assert stale(conn) == list(range(65, 129))
    assert sorted(embeddings(conn)) == list(range(1, 65)) + list(range(129, 151))
    assert "SECRET" not in caplog.text and "boom" not in caplog.text
    assert any("embed batch" in r.getMessage() and "RuntimeError" in r.getMessage()
               for r in caplog.records)  # fmt: skip
    status = conn.execute("SELECT status, error FROM sync_run ORDER BY run_id DESC").fetchone()
    assert status == ("ok", None)

    td.next_run()
    assert sync(conn, td).stats["embedded"] == 64
    assert stale(conn) == [] and len(embeddings(conn)) == 150


def test_a3_reembeds_only_the_affected_windows(conn: Conn) -> None:
    """A3 end to end: the 7 changed or new windows are embedded, the other 628 kept as is."""
    td = FakeTdClient({"chats": [chat(A, history())], "runs": [a3_changes()]})
    enable(conn, td, A)
    assert sync(conn, td).stats["embedded"] == 640
    before = embeddings(conn)
    td.next_run()
    result = sync(conn, td)
    assert (result.stats["windows_rebuilt"], result.stats["embedded"]) == (7, 7)
    after = embeddings(conn)
    assert sorted(before.keys() - after.keys()) == DELETED_IN  # cascaded with their windows
    redone = sorted(k for k in after if after[k] != before.get(k))
    assert redone == EDITED + [641, 681]
    assert len(after) - len(redone) == 628 and stale(conn) == []
    assert count(conn, "window_embedding") == count(conn, "window")


@pytest.mark.parametrize("other", [FakeEmbedder(model="other"), FakeEmbedder(dim=512)])
def test_model_mismatch_embeds_nothing_and_the_run_is_an_error(
    conn: Conn, other: FakeEmbedder
) -> None:
    edit = {"chat_id": A, "id": 3, "text": "edited", "edit_date": NOW.isoformat()}
    td = FakeTdClient({"chats": [chat(A, hourly(5))], "runs": [{"edit": [edit]}]})
    enable(conn, td, A)
    sync(conn, td)
    before = embeddings(conn)
    td.next_run()
    with pytest.raises(embed.ModelMismatch):
        sync(conn, td, other)
    row = conn.execute("SELECT status, error, stats FROM sync_run ORDER BY run_id DESC")
    status, error, stats = row.fetchone()  # type: ignore[misc]
    assert (status, error, stats["edited"], stats["windows_rebuilt"]) == (
        "error", "model_mismatch", 1, 1)  # fmt: skip
    assert stats["embedded"] == 0 and embeddings(conn) == before
    assert stale(conn) == [3]  # ingestion and windows are committed; the window waits
    text = conn.execute("SELECT text FROM message WHERE server_message_id = 3").fetchone()
    assert text == ("edited",)


def test_embed_step_stops_at_the_time_budget(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, hourly(150))], "runs": [{}]})
    enable(conn, td, A)
    ingest.sync(conn, lambda _: td, SETTINGS, FakeClock(), embedder=FakeEmbedder())
    conn.execute('UPDATE "window" SET stale = true')
    calls = iter([False, True])
    counts = embed.embed_stale(conn, FakeEmbedder(), lambda: next(calls))
    assert counts == {"embedded": 64, "embed_failed": 0, "embed_stopped": 1}
    assert stale(conn) == list(range(65, 151))


# --- tgi embed -----------------------------------------------------------------------------


def test_cli_embed_reindex_after_a_provider_change(
    conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    td = FakeTdClient({"chats": [chat(A, hourly(70))], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)
    cli = CliRunner()
    monkeypatch.setenv("DATABASE_URL", db_url)
    monkeypatch.setenv("EMBED_MODEL", "fake-2")

    mismatch = cli.invoke(app, ["embed"])
    assert mismatch.exit_code == 3 and "model_mismatch" in mismatch.stderr
    assert {r[0] for r in embeddings(conn).values()} == {"fake"}

    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        assert cli.invoke(app, ["embed", "--reindex"]).exit_code == 2  # a sync is running
    assert {r[0] for r in embeddings(conn).values()} == {"fake"}

    done = cli.invoke(app, ["embed", "--reindex"])
    assert done.exit_code == 0, done.stderr
    assert done.stdout.strip() == "dropped=70 embedded=70 embed_failed=0 tokens=0"
    assert {r[0] for r in embeddings(conn).values()} == {"fake-2"} and stale(conn) == []
    again = cli.invoke(app, ["embed"])
    assert (again.exit_code, again.stdout.strip()) == (0, "embedded=0 embed_failed=0 tokens=0")

    monkeypatch.setenv("EMBED_PROVIDER", "voyage")  # no VOYAGE_API_KEY: a usage error
    assert cli.invoke(app, ["embed"]).exit_code == 1


def test_reindex_with_an_unusable_dim_deletes_nothing(conn: Conn) -> None:
    td = FakeTdClient({"chats": [chat(A, hourly(5))], "runs": [{}]})
    enable(conn, td, A)
    sync(conn, td)
    conn.execute('UPDATE "window" SET stale = true WHERE first_message_id = %s', (3 << SHIFT,))
    before = embeddings(conn)
    with pytest.raises(embed.ModelMismatch):
        embed.reindex(conn, FakeEmbedder(model="fake-512", dim=512))
    assert embeddings(conn) == before and len(before) == 5
    assert stale(conn) == [3]
