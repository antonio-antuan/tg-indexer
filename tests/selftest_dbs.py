"""Selftest databases a test owns: assign names, drop only those (m-T7-1, #35).

The fixtures built on these live in conftest.py (`owned`, `taken`, `owned_over_taken`).
"""

import os
import secrets  # the real module, unaffected by the monkeypatch of `selftest.secrets` below
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql

from tgi import selftest

# A child selftest names its database with the suffix the parent test reserved.
PIN = """
import os
from types import SimpleNamespace
import tgi.selftest as s
s.secrets = SimpleNamespace(token_hex=lambda _n: os.environ["TGI_QA_SELFTEST_SUFFIX"])
"""


def exists(url: str, name: str) -> bool:
    with psycopg.connect(url) as conn:
        return (
            conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
            is not None
        )


def drop(url: str, name: str) -> None:
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )


@dataclass
class Owned:
    """Selftest database names this test (and its child) may create; the only ones torn down.

    Each suffix is random and checked absent on the server before `selftest._database` sees
    it, which registers a name before its CREATE and drops it on failure: this avoids the
    known collision with a database it did not create; a small check-to-CREATE race remains.
    """

    db_url: str
    names: list[str] = field(default_factory=list)
    fresh: Callable[[], str] = field(default=lambda: secrets.token_hex(8))  # a test may override

    def assign(self) -> str:
        suffix = self.fresh()
        while exists(self.db_url, selftest.PREFIX + suffix):  # never hand out a taken name
            suffix = self.fresh()
        self.names.append(selftest.PREFIX + suffix)
        return suffix


@contextmanager
def owning(db_url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[Owned]:
    o = Owned(db_url)
    monkeypatch.setattr(selftest, "secrets", SimpleNamespace(token_hex=lambda _n: o.assign()))
    yield o
    # Drop first, then assert: a failing test still leaves none of its own databases behind.
    left = [n for n in o.names if exists(db_url, n)]
    for name in left:
        drop(db_url, name)
    assert left == [], f"test left its selftest databases behind: {left}"


def child_env(owned: Owned) -> dict[str, str]:
    return {**os.environ, "DATABASE_URL": owned.db_url, "TGI_QA_SELFTEST_SUFFIX": owned.assign()}
