"""QA probes for T8 (29f206c): embedding pipeline, mismatch, reindex, time limit, logs."""
# ruff: noqa: E501, S311

from dataclasses import replace
from typing import Any

import psycopg
import pytest
from test_embed import Recording, count, embeddings, hourly, stale
from test_ingest import NOW, SETTINGS, A, FakeClock, chat, enable
from typer.testing import CliRunner

from tgi import embed, ingest
from tgi.cli import app
from tgi.embed import FakeEmbedder
from tgi.td_fake import FakeTdClient

Conn = psycopg.Connection[tuple[object, ...]]
SHIFT = 20
cli = CliRunner()


def sync(conn: Conn, td: FakeTdClient, emb: Any = None, clock: Any = None, **kw: Any) -> Any:
    s = replace(SETTINGS, page_delay_ms=0)
    return ingest.sync(conn, lambda _: td, s, clock or FakeClock(), embedder=emb, **kw)


def vec(conn: Conn, first: int) -> list[float]:
    r = conn.execute(
        'SELECT e.embedding::text FROM "window" w JOIN window_embedding e USING (window_id)'
        " WHERE w.first_message_id = %s", (first << SHIFT,)).fetchone()  # fmt: skip
    return [float(x) for x in r[0][1:-1].split(",")]  # type: ignore[index]


def snap(conn: Conn) -> dict[int, tuple[Any, ...]]:
    return {k: (v[2], v[3]) for k, v in embeddings(conn).items()}


def mk(conn: Conn, n: int = 150, runs: Any = None) -> FakeTdClient:
    td = FakeTdClient({"chats": [chat(A, hourly(n))], "runs": runs or [{}, {}]})
    enable(conn, td, A)
    return td


def test_every_window_holds_the_vector_of_its_own_text(conn: Conn) -> None:
    td = mk(conn, 150)
    rec = Recording()
    r = sync(conn, td, rec)
    assert r.status == "ok" and rec.batches == [64, 64, 22]
    f = FakeEmbedder()
    for wid, first, text in conn.execute(
        'SELECT window_id, first_message_id, text FROM "window"'
    ).fetchall():
        got = vec(conn, first >> SHIFT)
        assert got == pytest.approx(f.embed_documents([text])[0], abs=1e-6), wid
    assert conn.execute("SELECT count(*) FROM window_embedding WHERE vector_dims(embedding) <> 1024 OR dim <> 1024 OR model <> 'fake'").fetchone() == (0,)  # fmt: skip


def test_one_bad_window_fails_only_its_batch_and_run_stays_ok(
    conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    td = mk(conn, 200)
    caplog.set_level("DEBUG")
    # newest first: batches 200..137, 136..73, 72..9, 8..1; message 70 → window 70 is in the
    # third; text must not leak
    conn.execute("SELECT 1")
    emb = FakeEmbedder(fail=lambda t: "message 70" in t.split("\n")[1] + "\n")
    r = sync(conn, td, emb)
    assert r.status == "ok" and r.stats["embed_failed"] == 64 and r.stats["embedded"] == 136
    assert stale(conn) == list(range(9, 73))
    assert "message 70" not in caplog.text
    td.next_run()
    r2 = sync(conn, td, FakeEmbedder())
    assert r2.status == "ok" and r2.stats["embedded"] == 64 and stale(conn) == []
    # the failed batch's windows never got a partial row in between
    assert count(conn, "window_embedding") == 200


def test_dim_and_model_mismatch_embed_nothing_but_commit_ingest_and_windows(conn: Conn) -> None:
    td = mk(conn, 70)
    sync(conn, td, FakeEmbedder())
    before = snap(conn)
    td.next_run()
    # new messages arrive and a message is edited
    td._runs.insert(
        0,
        {
            "add": [{"chat_id": A, "id": 71, "date": NOW.isoformat(), "text": "late arrival"}],
            "edit": [{"chat_id": A, "id": 5, "text": "changed five", "edit_date": NOW.isoformat()}],
        },
    )  # noqa: SLF001
    td.next_run()
    for bad in (
        FakeEmbedder(model="other"),
        FakeEmbedder(model="fake", dim=512),
        FakeEmbedder(model="x", dim=512),
    ):
        with pytest.raises(embed.ModelMismatch):
            sync(conn, td, bad)
        row = conn.execute("SELECT status, error FROM sync_run ORDER BY run_id DESC").fetchone()
        assert row == ("error", "model_mismatch")
        assert snap(conn) == before  # nothing re-embedded, nothing overwritten
    assert conn.execute("SELECT text FROM message WHERE server_message_id = 71").fetchone() == (
        "late arrival",
    )
    assert conn.execute("SELECT text FROM message WHERE server_message_id = 5").fetchone() == (
        "changed five",
    )
    assert 5 in stale(conn) or 71 in stale(conn)  # windows rebuilt and waiting
    assert conn.execute("SELECT windows_dirty_from FROM chat").fetchone() == (None,)
    # the same data with the right model catches up
    r = sync(conn, td, FakeEmbedder())
    assert r.status == "ok" and stale(conn) == []


def test_edit_delete_and_rename_reembed_only_what_changed(conn: Conn) -> None:
    td = mk(conn, 100)
    sync(conn, td, FakeEmbedder())
    before = snap(conn)
    td._runs.insert(
        0,
        {
            "edit": [{"chat_id": A, "id": 10, "text": "edited ten", "edit_date": NOW.isoformat()}],
            "delete": [{"chat_id": A, "id": 50}, {"chat_id": A, "id": 51}],
        },
    )  # noqa: SLF001
    td.next_run()
    r = sync(conn, td, FakeEmbedder())
    assert r.status == "ok" and r.stats["embedded"] == 1, r.stats  # window 10 only
    after = snap(conn)
    assert set(before) - set(after) == {50, 51}
    changed = {k for k in after if k in before and after[k] != before[k]}
    assert changed == {10}
    # unchanged windows keep the same embedded_at (not merely an equal vector)
    # rename → every window of the chat is re-embedded (accepted by SPEC §5.4)
    from tgi import chats
    from tgi.td_port import Chat

    chats.enable(conn, Chat(chat_id=A, title="Renamed", kind="supergroup"), "+34600000001")
    r = sync(conn, FakeTdClient({"chats": [chat(A, [])], "runs": [{}]}), FakeEmbedder())
    assert r.stats["embedded"] == len(after)
    assert stale(conn) == []


def test_embedded_at_untouched_for_unchanged_windows(conn: Conn) -> None:
    td = mk(conn, 80)
    sync(conn, td, FakeEmbedder())
    t = {k: v[3] for k, v in embeddings(conn).items()}
    td.next_run()
    sync(conn, td, FakeEmbedder())
    assert {k: v[3] for k, v in embeddings(conn).items()} == t


def test_time_limit_stops_embedding_and_next_run_finishes(conn: Conn) -> None:
    td = mk(conn, 200)

    class Slow(FakeEmbedder):
        def __init__(self, clock: FakeClock) -> None:
            super().__init__()
            self.clock = clock

        def embed_documents(self, texts: Any) -> Any:
            self.clock.sleep(60)  # each batch costs a minute
            return super().embed_documents(texts)

    clock = FakeClock()
    r = ingest.sync(
        conn,
        lambda _: td,
        replace(SETTINGS, page_delay_ms=0),
        clock,
        max_minutes=1.5,
        embedder=Slow(clock),
    )
    assert r.status == "time_limited"
    assert 0 < r.stats["embedded"] < 200 and r.stats["embedded"] % 64 == 0
    assert len(stale(conn)) == 200 - r.stats["embedded"]
    td.next_run()
    r2 = sync(conn, td, FakeEmbedder())
    assert r2.status == "ok" and stale(conn) == []


def test_reindex_variants(conn: Conn, db_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    td = mk(conn, 70)
    sync(conn, td, FakeEmbedder())
    monkeypatch.setenv("DATABASE_URL", db_url)
    # re-run reindex with the same model: drops and re-embeds all, identical vectors
    before = {k: v[2] for k, v in embeddings(conn).items()}
    r = cli.invoke(app, ["embed", "--reindex"])
    assert r.exit_code == 0 and r.stdout.strip() == "dropped=70 embedded=70 embed_failed=0 tokens=0"
    assert {k: v[2] for k, v in embeddings(conn).items()} == before
    # embed with nothing stale
    assert cli.invoke(app, ["embed"]).stdout.strip() == "embedded=0 embed_failed=0 tokens=0"
    # unknown provider: refuses before touching anything
    monkeypatch.setenv("EMBED_PROVIDER", "voyage")
    out = cli.invoke(app, ["embed", "--reindex"])
    assert out.exit_code != 0 and count(conn, "window_embedding") == 70
    print("provider-unavailable exit code", out.exit_code)
    monkeypatch.setenv("EMBED_PROVIDER", "fake")  # unset now means voyage (P2.T7)
    # lock held → exit 2 and nothing deleted, for both forms
    with psycopg.connect(db_url, autocommit=True) as other:
        other.execute("SELECT pg_advisory_lock(%s)", (ingest.SYNC_LOCK_KEY,))
        for args in (["embed"], ["embed", "--reindex"]):
            res = cli.invoke(app, args)
            assert res.exit_code == 2
        assert count(conn, "window_embedding") == 70
    # lock is released after a CLI run and after SyncLocked
    assert cli.invoke(app, ["embed"]).exit_code == 0


def test_reindex_with_unusable_provider_dim_does_not_lose_embeddings(conn: Conn) -> None:
    """Probe: reindex deletes first; a provider with the wrong dim then raises."""
    td = mk(conn, 70)
    sync(conn, td, FakeEmbedder())
    with pytest.raises(embed.ModelMismatch):
        embed.reindex(conn, FakeEmbedder(model="x", dim=512))
    # QA finding (minor, latent): reindex commits the delete before the dim check fails.
    assert (count(conn, "window_embedding"), len(stale(conn))) == (70, 0), "embeddings lost"


def test_reindex_with_failing_batch_leaves_that_batch_stale(conn: Conn) -> None:
    td = mk(conn, 130)
    sync(conn, td, FakeEmbedder())
    res = embed.reindex(conn, FakeEmbedder(fail=lambda t: "message 100" in t))
    # newest first: window 100 is in the first batch, 130..67
    assert res["embed_failed"] == 64 and stale(conn) == list(range(67, 131))
    assert count(conn, "window_embedding") == 130 - 64


def test_logs_never_carry_window_text_or_exception_text(
    conn: Conn, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("DEBUG")
    msgs = hourly(70, "TOPSECRET {i}")
    td = FakeTdClient({"chats": [chat(A, msgs)], "runs": [{}]})
    enable(conn, td, A)

    def boom(t: str) -> bool:
        return "TOPSECRET 3 " in t + " "

    class Leaky(FakeEmbedder):
        def embed_documents(self, texts: Any) -> Any:
            if any(boom(t) for t in texts):
                raise RuntimeError("provider says: " + texts[0])
            return super().embed_documents(texts)

    sync(conn, td, Leaky())
    with pytest.raises(embed.ModelMismatch):
        sync(conn, td, FakeEmbedder(model="other"))
    assert "TOPSECRET" not in caplog.text and "provider says" not in caplog.text
    assert conn.execute("SELECT count(*) FROM sync_run WHERE stats::text LIKE '%TOPSECRET%' OR error LIKE '%TOPSECRET%'").fetchone() == (0,)  # fmt: skip
