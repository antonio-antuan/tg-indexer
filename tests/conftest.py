"""Shared fixtures: every test database is created fresh and dropped afterwards.

DATABASE_URL must point at a Postgres 16 with pgvector and pg_trgm (the compose `db`
service by default) and a role that may create databases. Database tests never skip.
"""

import os
import secrets
from collections.abc import Callable, Iterator

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from selftest_dbs import Owned, drop, exists, owning

from tgi import cli, db, embed, selftest, windows


@pytest.fixture(autouse=True)
def no_env_file(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI checks a `.env` that does not exist, whatever the checkout holds."""
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path_factory.mktemp("cwd") / ".env")


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake provider and no provider keys: no test reaches Voyage or OpenAI.

    `make` exports `.env`, which may hold a real key and `EMBED_PROVIDER=voyage` (the
    default); a test that needs a provider or a key sets it itself.
    """
    monkeypatch.setenv("EMBED_PROVIDER", "fake")
    for key in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "EMBED_MODEL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.fail("DATABASE_URL is not set; start the db (`make db`) and load .env")
    return url


@pytest.fixture(scope="session")
def create_database(admin_url: str) -> Iterator[Callable[[], str]]:
    """Factory returning the URL of a new empty database; all are dropped at session end."""
    created: list[str] = []

    def create() -> str:
        name = f"tgi_test_{secrets.token_hex(4)}"
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        created.append(name)
        return make_conninfo(admin_url, dbname=name)

    yield create
    with psycopg.connect(admin_url, autocommit=True) as conn:
        for name in created:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


@pytest.fixture(scope="session")
def database_url(create_database: Callable[[], str]) -> str:
    """One fresh database per test session."""
    return create_database()


@pytest.fixture
def db_url(create_database: Callable[[], str]) -> str:
    """A fresh, migrated database for one test."""
    url = create_database()
    with db.connect(url) as c:
        db.migrate(c)
    return url


@pytest.fixture
def conn(db_url: str) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    with psycopg.connect(db_url, autocommit=True) as c:
        yield c


@pytest.fixture
def no_window_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip `tgi sync`'s window step so ingest tests can read the marker it leaves behind."""
    monkeypatch.setattr(windows, "rebuild_dirty", lambda conn, settings: {})


@pytest.fixture
def no_embed_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip `tgi sync`'s embed step so window tests can read the `stale` flags it consumes."""
    none = {"embedded": 0, "embed_failed": 0, "embed_stopped": 0}
    monkeypatch.setattr(embed, "embed_stale", lambda *args, **kwargs: dict(none))


@pytest.fixture
def owned(db_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Owned]:
    with owning(db_url, monkeypatch) as o:
        yield o


@pytest.fixture
def taken(db_url: str) -> Iterator[str]:
    """A `tgi_selftest_*` database made by hand; checked after `owned_over_taken`'s teardown."""
    suffix = secrets.token_hex(8)
    name = selftest.PREFIX + suffix
    with psycopg.connect(db_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield suffix
        assert exists(db_url, name), "the fixture's teardown dropped a database it did not name"
    finally:
        drop(db_url, name)


@pytest.fixture
def owned_over_taken(taken: str, db_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Owned]:
    """`owned`, set up after (so torn down before) `taken`."""
    with owning(db_url, monkeypatch) as o:
        yield o
