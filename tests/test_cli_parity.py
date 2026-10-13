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
@pytest.mark.parametrize(
    "link",
    ["https://t.me/nie_valencia/4", "t.me/nie_valencia/4?single", "https://t.me/c/1000000101/4",
     "https://t.me/c/1000000101/99/4", " https://t.me/nie_valencia/4 ",
     "https://t.me/s/nie_valencia/4", "https://t.me/NIE_valencia/4/",
     "t.me/nie_valencia/4/?thread=2"],
)  # fmt: skip
def test_a_link_works_as_the_message(cli: Any, link: str) -> None:
    by_id = as_json(cli("context", str(A), str(mid(4)), "--json"))
    assert by_id and as_json(cli("context", str(A), link, "--json")) == by_id


@pytest.mark.usefixtures("archive")
def test_a_bare_number_is_always_a_message_id(cli: Any) -> None:
    # no guessing: 4 is message_id 4 (no such message), not server message 4
    assert as_json(cli("context", str(A), "4", "--json")) == []
    assert as_json(cli("context", str(A), str(1 << 20), "--json"))  # server message 1
    for bad in ("four", "https://example.com/4", "t.me/nie_valencia", "t.me/c/123", "t.me/s/123",
                "t.me/+hash/5"):  # fmt: skip
        result = cli("context", str(A), bad)
        assert result.exit_code == 1 and "MESSAGE" in result.output, bad
    # well formed, but not this chat's: t.me's own paths and other chats
    for other in ("t.me/joinchat/12345678", "t.me/call/123", "t.me/someone_else/4",
                  "https://t.me/c/1000000999/4"):  # fmt: skip
        result = cli("context", str(A), other)
        assert result.exit_code == 1 and "bad_argument" in result.output, other


@pytest.mark.usefixtures("archive")
def test_search_recent_and_status_refuse_unknown_filters_like_the_tools(cli: Any) -> None:
    for args, code in (
        (("search", "x", "--tag", "nowhere"), "unknown_tag"),
        (("search", "x", "--chat", "123"), "unknown_chat"),
        (("recent", "--tag", "nowhere"), "unknown_tag"),
        (("recent", "--chat", "123"), "unknown_chat"),
        (("status", "--chat", "123"), "unknown_chat"),
    ):
        result = cli(*args)
        assert result.exit_code == 1 and code in result.output, (args, result.output)


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
    context = cli("context", str(A), "https://t.me/nie_valencia/4")
    assert context.exit_code == 0 and "https://t.me/nie_valencia/4" in context.output


@pytest.mark.usefixtures("archive")
def test_search_and_recent_report_the_tools_codes(cli: Any) -> None:
    for args, code in (
        (("search", "x", "--since", "2026-10-02", "--until", "2026-10-01"), "bad_range"),
        (("search", "  "), "bad_argument"),
        (("recent", "--since", "2026-10-02", "--until", "2026-10-01"), "bad_range"),
        (("recent", "--cursor", "nonsense"), "bad_argument"),
        (("recent", "--hours", "2", "--since", "2026-10-01"), "bad_argument"),
        (("recent", "--until", "2026-10-01"), "bad_argument"),
    ):
        result = cli(*args)
        assert result.exit_code == 1 and f"{code}:" in result.output, (args, result.output)
