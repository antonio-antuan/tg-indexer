"""The CLI reads what the MCP read tools return: `tgi tags list`, `tgi chats archived`,
`tgi context`, `tgi thread` (with `tgi search` and `tgi recent`, every read tool)."""

import json
from pathlib import Path
from typing import Any

import pytest
from test_mcp import A, archive, in_process, mid  # noqa: F401 - the fixture
from typer.testing import CliRunner

from tgi.cli import app


@pytest.fixture
def cli(db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", db_url)

    def run(*args: str) -> Any:
        return CliRunner().invoke(app, list(args))

    return run


def as_json(result: Any) -> Any:
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.mark.usefixtures("archive")
def test_each_command_returns_what_its_tool_returns(db_url: str, cli: Any) -> None:
    tags, chats, spain, context, thread = in_process(db_url, [
        ("list_tags", {}),
        ("list_chats", {}),
        ("list_chats", {"tag": "spain"}),
        ("get_context", {"chat_id": A, "message_id": mid(4), "before": 2, "after": 2}),
        ("get_thread", {"chat_id": A, "message_id": mid(3)}),
    ])  # fmt: skip
    assert as_json(cli("tags", "list", "--json")) == tags
    assert as_json(cli("chats", "archived", "--json")) == chats
    assert as_json(cli("chats", "archived", "--tag", "spain", "--json")) == spain
    got = cli("context", str(A), str(mid(4)), "--before", "2", "--after", "2", "--json")
    assert as_json(got) == context
    assert as_json(cli("thread", str(A), str(mid(3)), "--json")) == thread


@pytest.mark.usefixtures("archive")
def test_a_link_number_works_as_the_message(cli: Any) -> None:
    by_id = as_json(cli("context", str(A), str(mid(4)), "--json"))
    by_link = as_json(cli("context", str(A), "4", "--json"))  # https://t.me/…/4
    assert by_id == by_link and by_id


@pytest.mark.usefixtures("archive")
def test_tool_errors_exit_1_with_their_code(cli: Any) -> None:
    for args, code in (
        (("chats", "archived", "--tag", "nowhere"), "unknown_tag"),
        (("context", "123", "4"), "unknown_chat"),
        (("thread", "123", "4"), "unknown_chat"),
        (("context", str(A), "4", "--before", "90", "--after", "90"), "bad_range"),
    ):
        result = cli(*args)
        assert result.exit_code == 1 and code in result.output, (args, result.output)


@pytest.mark.usefixtures("archive")
def test_text_output(cli: Any) -> None:
    listed = cli("tags", "list")
    assert listed.exit_code == 0 and "spain\tconfirmed\t1 chats" in listed.output
    chats = cli("chats", "archived")
    assert chats.exit_code == 0 and str(A) in chats.output and "tags spain" in chats.output
    context = cli("context", str(A), "4")
    assert context.exit_code == 0 and "https://t.me/nie_valencia/4" in context.output
