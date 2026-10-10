"""P2.T7 phase gate: the P2 flow end to end through the CLI, on throwaway data.

`tgi sync` loads the reference scenario (fake TDLib, fake embedder), the chats are tagged,
`tgi mcp --http` serves the database on loopback in a subprocess, `tgi selftest` queries it
as the Nano agent would (a real MCP client over HTTP), a message deleted in Telegram is
tombstoned by the next sync's sweep and `tgi purge` hard-deletes it, and the server still
answers afterwards. A10 over the whole flow: no message text, query, token or phone appears
in any output or log line, the server's included at DEBUG.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
from test_bind_guard import free_port, listening
from typer.testing import CliRunner

from tgi import selftest
from tgi.cli import app

Conn = psycopg.Connection[tuple[object, ...]]
SCENARIO = json.loads((Path(__file__).parent / "scenarios" / "reference.json").read_text())
TEXTS = [m["text"] for c in SCENARIO["chats"] for m in c["messages"]]
TOKEN = "e2e-" + "k" * 39
VALENCIA = -1001700000001
GONE = 3  # "https://t.me/rus_valencia/3", an unrelated message deleted in Telegram

# `tgi mcp --http` with every logger at DEBUG on stdout, so the A10 check sees the most.
SERVE = (
    "import logging, sys; logging.basicConfig(level=logging.DEBUG, stream=sys.stdout); "
    "from tgi.cli import app; sys.argv = ['tgi', 'mcp', '--http']; app()"
)


def write(path: Path, scenario: dict[str, Any]) -> Path:
    path.write_text(json.dumps(scenario, ensure_ascii=False))
    return path


def test_p2_flow_end_to_end(
    db_url: str,
    conn: Conn,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("DATABASE_URL", db_url)
    outputs: list[str] = []

    def tgi(*args: str) -> str:
        result = CliRunner().invoke(app, list(args))
        outputs.append(result.output)
        assert result.exit_code == 0, result.output
        return result.stdout

    # 1. Sync the reference scenario, dated so that its `reference.now` is now.
    day1 = selftest._shifted(SCENARIO, datetime.now(UTC))
    first = write(tmp_path / "day1.json", day1)
    for chat in day1["chats"]:
        tgi("chats", "enable", str(chat["chat_id"]), "--account", selftest.ACCOUNT,
            "--scenario", str(first))  # fmt: skip
    tgi("sync", "--scenario", str(first))
    for tag, chat_ids in day1["reference"]["tags"].items():
        for chat_id in chat_ids:
            tgi("tags", "propose", str(chat_id), tag)
            tgi("tags", "confirm", str(chat_id), tag)
    assert conn.execute("SELECT count(*) FROM window_embedding").fetchone() == (9,)

    # 2. Serve it on loopback, as `tgi mcp --http` runs for the Nano agent.
    port = free_port()
    log = tmp_path / "server.log"
    env = {**os.environ, "MCP_BIND": f"127.0.0.1:{port}", "MCP_TOKEN": TOKEN}
    with log.open("w") as out:
        server = subprocess.Popen(  # noqa: S603 - our own entry point
            [sys.executable, "-c", SERVE], cwd=tmp_path, env=env, stdout=out,
            stderr=subprocess.STDOUT,
        )  # fmt: skip
    try:
        deadline = time.monotonic() + 20
        while not listening(port):
            assert server.poll() is None and time.monotonic() < deadline, log.read_text()
            time.sleep(0.1)
        assert listening(port) == {"127.0.0.1"}
        monkeypatch.setenv("MCP_BIND", f"127.0.0.1:{port}")
        monkeypatch.setenv("MCP_TOKEN", TOKEN)

        # 3. The reference query through a real MCP client: tags, search, context, links.
        assert tgi("selftest") == "selftest ok: tags=2 hits=7 context=4\n"

        # 4. A message deleted in Telegram is tombstoned by the next sweep, then purged.
        day2 = json.loads(json.dumps(day1))
        day2["chats"][0]["messages"] = [m for m in day2["chats"][0]["messages"]
                                        if m["id"] != GONE]  # fmt: skip
        monkeypatch.setenv("SWEEP_DAYS", "30")
        tgi("sync", "--scenario", str(write(tmp_path / "day2.json", day2)))
        tombstoned = "SELECT chat_id, server_message_id FROM message WHERE deleted_at IS NOT NULL"
        assert conn.execute(tombstoned).fetchall() == [(VALENCIA, GONE)]
        assert tgi("purge", "--deleted", "--older-than", "0d", "--yes") == (
            f"purged chat {VALENCIA}: 1\npurged 1\n"
        )
        gone = "SELECT count(*) FROM message WHERE chat_id = %s AND server_message_id = %s"
        assert conn.execute(gone, (VALENCIA, GONE)).fetchone() == (0,)

        # 5. The server answers after the purge: the deleted message is neither a hit nor in
        # the top hit's context.
        assert tgi("selftest") == "selftest ok: tags=2 hits=6 context=3\n"
    finally:
        server.terminate()
        server.wait(timeout=10)
    served = log.read_text()

    # A10 over the whole flow: counts and ids only.
    assert "search mode=hybrid queries=4" in served  # the server's logs were captured
    assert "Uvicorn running" in served and "DEBUG:mcp.server" in served
    for text in (*outputs, caplog.text, served):
        assert [t for t in TEXTS if t in text] == []
        assert [q for q in selftest.QUERIES if q in text] == []
        assert TOKEN not in text
        assert selftest.ACCOUNT not in text
