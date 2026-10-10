# Local CI. GitHub Actions are disabled on this repo; `make ci` output is the CI evidence.
# Reads .env (POSTGRES_PASSWORD, DATABASE_URL) when present; its values override exported
# environment variables, so pass overrides on the make command line instead. Tests create
# and drop their own databases through DATABASE_URL.
-include .env
export

# Starts the compose Postgres. To use another local Postgres 16 with pgvector:
#   make ci DB_UP=true DATABASE_URL=postgresql://...
DB_UP ?= docker compose up -d --wait db

.PHONY: ci lint types test actionlint db

ci: lint types actionlint test

lint:
	uv run ruff check .
	uv run ruff format --check .

types:
	uv run mypy

actionlint:
	actionlint

db:
	$(DB_UP)

test: db
	uv run pytest
