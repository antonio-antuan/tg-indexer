"""A1: `tgi db migrate` creates the SPEC §4 schema; re-running is a no-op."""

from collections.abc import Callable

import psycopg
import pytest
from typer.testing import CliRunner

from tgi import db
from tgi.cli import app

TABLES = {
    "chat",
    "chat_tag",
    "message",
    "window",
    "window_embedding",
    "sync_run",
    "schema_migrations",
}

# index name -> fragment its definition must contain
INDEXES = {
    "message_tsv_idx": "USING gin (tsv)",
    "message_text_trgm_idx": "USING gin (text gin_trgm_ops)",
    "message_chat_date_idx": "USING btree (chat_id, date)",
    "message_chat_reply_idx": "USING btree (chat_id, reply_to_message_id)",
    "window_chat_start_idx": "USING btree (chat_id, start_date)",
    "window_message_ids_idx": "USING gin (message_ids)",
    "window_embedding_hnsw_idx": "USING hnsw (embedding vector_cosine_ops) WITH (m='16', "
    "ef_construction='128')",
    "window_chat_id_first_message_id_key": "(chat_id, first_message_id)",
}


def _schema_snapshot(conn: psycopg.Connection[tuple[object, ...]]) -> list[tuple[object, ...]]:
    return conn.execute(
        "SELECT table_name, column_name, data_type, is_nullable, column_default"
        " FROM information_schema.columns WHERE table_schema = 'public'"
        " UNION ALL SELECT tablename, indexname, indexdef, NULL, NULL"
        " FROM pg_indexes WHERE schemaname = 'public' ORDER BY 1, 2"
    ).fetchall()


@pytest.fixture
def empty_url(create_database: Callable[[], str]) -> str:
    return create_database()


def test_migrate_creates_schema_and_rerun_is_noop(empty_url: str) -> None:
    with psycopg.connect(empty_url) as conn:
        assert db.migrate(conn) == [1, 2, 3]
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
        }
        extensions = {r[0] for r in conn.execute("SELECT extname FROM pg_extension")}
        indexdefs = dict(
            conn.execute("SELECT indexname, indexdef FROM pg_indexes WHERE schemaname = 'public'")
        )
        dim = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute"
            " WHERE attrelid = 'window_embedding'::regclass AND attname = 'embedding'"
        ).fetchone()
        before = _schema_snapshot(conn)

        assert db.migrate(conn) == []
        after = _schema_snapshot(conn)
        versions = conn.execute("SELECT version FROM schema_migrations").fetchall()

    assert tables == TABLES
    assert {"vector", "pg_trgm"} <= extensions
    for name, fragment in INDEXES.items():
        assert fragment in str(indexdefs.get(name)), name
    assert dim == ("vector(1024)",)
    assert before == after
    assert versions == [(1,), (2,), (3,)]
    assert ("chat", "windows_dirty_from", "bigint", "YES", None) in after
    assert ("chat", "backfill_cutoff", "timestamp with time zone", "YES", None) in after


def test_cli_migrate(empty_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", empty_url)
    runner = CliRunner()
    first = runner.invoke(app, ["db", "migrate"])
    second = runner.invoke(app, ["db", "migrate"])
    assert (first.exit_code, first.output) == (0, "applied: 0001, 0002, 0003\n")
    assert (second.exit_code, second.output) == (0, "schema is up to date\n")


def test_tsv_stems_three_languages_and_utf8_database(empty_url: str) -> None:
    with psycopg.connect(empty_url) as conn:
        db.migrate(conn)
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind)"
            " VALUES (-1001, 'acct', 't', 'supergroup')"
        )
        conn.execute(
            "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
            " VALUES (-1001, 1048576, 1, now(), 'продлении renovaciones renewals')"
        )
        tsv = conn.execute("SELECT tsv::text FROM message").fetchone()
        encoding, collate, ctype = conn.execute(
            "SELECT pg_encoding_to_char(encoding), datcollate, datctype"
            " FROM pg_database WHERE datname = current_database()"
        ).fetchone() or (None, None, None)
        cyrillic_trigrams = conn.execute("SELECT show_trgm('внж')").fetchone()

    assert tsv is not None
    for lexeme in ("'продлен'", "'renov'", "'renew'"):
        assert lexeme in tsv[0]
    # PG16 dropped the lc_collate setting; the database's own locale is the source of truth.
    assert encoding == "UTF8"
    for value in (collate, ctype):
        assert str(value).lower().replace("-", "").endswith(".utf8"), value
    assert cyrillic_trigrams is not None and len(cyrillic_trigrams[0]) > 0


def test_message_id_must_be_tdlib_server_shaped(empty_url: str) -> None:
    with psycopg.connect(empty_url) as conn:
        db.migrate(conn)
        conn.execute(
            "INSERT INTO chat (chat_id, account_phone, title, kind)"
            " VALUES (-1001, 'acct', 't', 'supergroup')"
        )
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO message (chat_id, message_id, server_message_id, date, text)"
                " VALUES (-1001, 1048577, 1, now(), '')"
            )
