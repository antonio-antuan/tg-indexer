# ruff: noqa: E501, S603, ANN001, ANN202
"""QA P2.T5: selftest findings. Tests 1-2 fail on 0feb1f0 (see the report); the rest pass."""

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from selftest_dbs import PIN, Owned, child_env, exists
from typer.testing import CliRunner

from tgi import selftest
from tgi.cli import app

REFERENCE = Path(__file__).parent / "scenarios" / "reference.json"
SCENARIO = json.loads(REFERENCE.read_text())
START = PIN + "import sys; from tgi.cli import app; app(sys.argv[1:])"


def leftovers(owned: Owned) -> list[str]:
    """The selftest databases this test was assigned that still exist; `owned` drops only these."""
    return [n for n in owned.names if exists(owned.db_url, n)]


def scenario_file(tmp_path: Path, edit) -> Path:  # type: ignore[no-untyped-def]
    data = json.loads(json.dumps(SCENARIO))
    edit(data)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(data))
    return path


def test_1_a_missing_valencia_tag_is_a_failed_check_not_an_error(
    owned: Owned, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """README: exit 1 when a check fails, 3 on any other error. `Failed` raised inside the
    client's task group reaches the CLI as an ExceptionGroup: exit 3, message `ExceptionGroup`."""
    path = scenario_file(tmp_path, lambda d: d["reference"]["tags"].pop("valencia"))
    monkeypatch.setenv("DATABASE_URL", owned.db_url)
    result = CliRunner().invoke(app, ["selftest", "--scenario", str(path)])
    assert result.exit_code == 1, (result.exit_code, result.stderr)
    assert "valencia" in result.stderr
    assert owned.names and leftovers(owned) == []


def test_2_sigterm_drops_the_throwaway_database(owned: Owned) -> None:
    """SIGINT cleans up (finally runs); SIGTERM kills Python without running it."""
    env = child_env(owned)
    p = subprocess.Popen(
        [sys.executable, "-c", START, "selftest", "--scenario", str(REFERENCE)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not leftovers(owned):
        time.sleep(0.02)
    assert leftovers(owned), "the selftest database never appeared"
    p.send_signal(signal.SIGTERM)
    p.wait(30)
    assert leftovers(owned) == []


def test_sigint_drops_the_throwaway_database(owned: Owned) -> None:
    env = child_env(owned)
    p = subprocess.Popen(
        [sys.executable, "-c", START, "selftest", "--scenario", str(REFERENCE)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )  # fmt: skip
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not leftovers(owned):
        time.sleep(0.02)
    assert leftovers(owned), "the selftest database never appeared"
    p.send_signal(signal.SIGINT)
    p.wait(30)
    assert leftovers(owned) == []


@pytest.mark.parametrize(
    ("edit", "code", "text"),
    [
        (lambda d: d["reference"]["tags"].update(ghost=[-5]), 1, "which it does not have"),
        (lambda d: d["reference"]["unrelated"].append(d["reference"]["seeded"][0]), 1, "unrelated messages rank"),
        (lambda d: d["chats"][0]["messages"][0].update(date="2026-09-29T09:00:00+00:00"), 1, "1 excluded"),
        (lambda d: d["reference"].pop("seeded"), 3, "KeyError"),
    ],
)  # fmt: skip
def test_scenario_variants_exit_codes_and_cleanup(
    owned: Owned,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    edit,
    code: int,
    text: str,  # type: ignore[no-untyped-def]
) -> None:
    path = scenario_file(tmp_path, edit)
    monkeypatch.setenv("DATABASE_URL", owned.db_url)
    result = CliRunner().invoke(app, ["selftest", "--scenario", str(path)])
    assert result.exit_code == code
    assert text in result.stderr
    assert owned.names and leftovers(owned) == []


def test_a_foreign_selftest_database_survives_this_file(
    owned_over_taken: Owned, taken: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#35: a failing selftest here once dropped every `tgi_selftest_*` database. `taken`
    checks its database after `owned_over_taken`'s teardown has run."""
    path = scenario_file(tmp_path, lambda d: d["reference"]["tags"].pop("valencia"))
    monkeypatch.setenv("DATABASE_URL", owned_over_taken.db_url)
    result = CliRunner().invoke(app, ["selftest", "--scenario", str(path)])
    assert result.exit_code == 1, (result.exit_code, result.stderr)
    assert owned_over_taken.names and leftovers(owned_over_taken) == []
    assert exists(owned_over_taken.db_url, selftest.PREFIX + taken)
