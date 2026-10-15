"""The systemd user units for `tgi sync` (deploy/tgi-sync.service, deploy/tgi-sync.timer)."""

import configparser
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SERVICE, TIMER = ROOT / "deploy" / "tgi-sync.service", ROOT / "deploy" / "tgi-sync.timer"


def units() -> tuple[configparser.ConfigParser, configparser.ConfigParser]:
    parsed = []
    for path in (SERVICE, TIMER):
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str  # type: ignore[assignment, method-assign]
        cp.read(path)
        parsed.append(cp)
    return parsed[0], parsed[1]


def test_the_service_runs_one_budgeted_sync_and_a_held_lock_is_no_failure() -> None:
    service, timer = units()
    s = service["Service"]
    assert s["Type"] == "oneshot"
    assert s["ExecStart"].endswith("/.venv/bin/tgi sync")  # budget: SYNC_MAX_MINUTES
    assert s["EnvironmentFile"] == "%h/.config/tgi/.env"
    assert s["SuccessExitStatus"] == "2"  # SPEC §5.3: exit 2 = another run holds the lock
    assert int(s["TimeoutStartSec"].removesuffix("min")) > 150  # the budget is soft
    assert timer["Timer"]["OnCalendar"] == "*-*-* 00/3:00:00 UTC"  # as tgi.eta estimates
    assert timer["Timer"]["Persistent"] == "true"
    assert timer["Install"]["WantedBy"] == "timers.target"


def test_the_units_hold_no_secrets_and_are_not_writable_by_others() -> None:
    for path in (SERVICE, TIMER):
        text = path.read_text()
        for needle in ("MCP_TOKEN=", "API_KEY", "PASSWORD", "TDLIB", "DATABASE_URL="):
            assert needle not in text, (path.name, needle)
        assert stat.S_IMODE(path.stat().st_mode) & 0o022 == 0


def test_systemd_analyze_verify_accepts_the_units(tmp_path: Path) -> None:
    analyze = shutil.which("systemd-analyze")
    if analyze is None:
        pytest.skip("no systemd-analyze")
    for path in (SERVICE, TIMER):
        (tmp_path / path.name).write_text(path.read_text())
    r = subprocess.run(  # noqa: S603 - fixed arguments, files written above
        [analyze, "verify", str(tmp_path / TIMER.name), str(tmp_path / SERVICE.name)],
        capture_output=True, text=True, timeout=30, env={**os.environ, "LC_ALL": "C"},
    )  # fmt: skip
    problems = [
        line for line in (r.stdout + r.stderr).splitlines()
        if line.strip() and "is not executable" not in line
    ]  # fmt: skip
    assert problems == [], problems
