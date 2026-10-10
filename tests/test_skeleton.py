import psycopg
from typer.testing import CliRunner

from tgi import __version__
from tgi.cli import app


def test_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"tgi {__version__}"


def test_session_database_is_fresh(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        row = conn.execute("SELECT current_database()").fetchone()
        tables = conn.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchone()
    assert row is not None and row[0].startswith("tgi_test_")
    assert tables == (0,)
