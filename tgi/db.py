"""Postgres connection and the numbered-migration runner (SPEC §3, §4)."""

import logging
import re
from pathlib import Path

import psycopg

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d{4})_[a-z0-9_]+\.sql$")
# Arbitrary constant; serialises concurrent `tgi db migrate` runs.
_MIGRATE_LOCK_KEY = 0x7467_6930


def connect(url: str) -> psycopg.Connection[tuple[object, ...]]:
    return psycopg.connect(url)


def migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, Path]]:
    found = []
    for path in sorted(directory.iterdir()):
        match = _MIGRATION_NAME.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return found


def migrate(
    conn: psycopg.Connection[tuple[object, ...]], directory: Path = MIGRATIONS_DIR
) -> list[int]:
    """Apply pending migrations in one transaction under an advisory lock.

    Returns the versions applied; an empty list means the schema was already current.
    """
    applied: list[int] = []
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATE_LOCK_KEY,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version integer PRIMARY KEY,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, path in migrations(directory):
            if version in done:
                continue
            conn.execute(path.read_bytes())
            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
            log.info("applied migration %04d", version)
            applied.append(version)
    return applied
