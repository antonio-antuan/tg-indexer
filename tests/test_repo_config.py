"""Static checks of repo configuration (A10 compose/.env part).

This parses the files; it does not run `docker compose config`.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_compose_publishes_postgres_on_loopback_only() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    db = compose["services"]["db"]
    assert db["ports"] == ["127.0.0.1:5432:5432"]
    assert db["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
    assert "en_US.utf8" in db["environment"]["POSTGRES_INITDB_ARGS"]


def test_env_example_has_no_default_password() -> None:
    values = dict(
        line.split("=", 1)
        for line in (ROOT / ".env.example").read_text().splitlines()
        if line and not line.startswith("#")
    )
    assert values["POSTGRES_PASSWORD"] == ""


def test_secrets_and_data_are_gitignored() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in ignored
    assert "data/" in ignored
