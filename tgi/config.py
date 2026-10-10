"""Settings read from the environment with the P1 defaults (PLAN P1 "Config keys").

Secrets are kept out of `repr()` so a logged or printed `Settings` never shows them.
`POSTGRES_PASSWORD` is read by docker compose only and reaches Python inside `DATABASE_URL`.
"""

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

OnDelete = Literal["tombstone", "purge"]


def env_file_warning(path: Path) -> str | None:
    """SPEC §5.1: a warning when the `.env` at `path` is more open than 0600.

    `tgi` does not read `.env` (the shell or make loads it); it only checks the mode.
    """
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:  # no .env: the variables come from elsewhere
        return None
    if not mode & 0o077:
        return None
    return f"warning: {path.absolute()} is mode {mode:04o}, more open than 0600; chmod 600 it"


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer") from None
    if value < 0:
        raise ValueError(f"{key} must not be negative")
    return value


# A backfill lookback above this is a typo; it also keeps `years_before` past year 1 and the
# value inside the int4 `backfill_years` column.
MAX_BACKFILL_YEARS = 100


def check_years(years: int, name: str = "years") -> int:
    """`years` if it is a valid backfill lookback, else ValueError naming `name`."""
    if not 1 <= years <= MAX_BACKFILL_YEARS:
        raise ValueError(f"{name} must be between 1 and {MAX_BACKFILL_YEARS}")
    return years


# A sync schedule longer than a week is a typo; it also keeps `eta.next_run` in range.
MAX_SYNC_EVERY_MINUTES = 7 * 24 * 60


def _str(env: Mapping[str, str], key: str) -> str | None:
    return env.get(key, "").strip() or None


def _sync_every(minutes: int) -> int:
    if minutes > MAX_SYNC_EVERY_MINUTES:
        raise ValueError(f"SYNC_EVERY_MINUTES must be at most {MAX_SYNC_EVERY_MINUTES}")
    return minutes


@dataclass(frozen=True)
class Settings:
    database_url: str | None = field(default=None, repr=False)
    data_dir: Path | None = None
    tdjson_path: Path | None = None
    tdlib_db_key: str | None = field(default=None, repr=False)
    tg_api_id: str | None = field(default=None, repr=False)
    tg_api_hash: str | None = field(default=None, repr=False)
    backfill_years: int = 3
    sweep_days: int = 7
    page_delay_ms: int = 300
    on_delete: OnDelete = "tombstone"
    window_gap_min: int = 30
    window_max_chars: int = 1500
    window_max_msgs: int = 40
    embed_provider: str = "voyage"  # SPEC §3; tests use "fake"
    embed_model: str | None = None
    voyage_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    mcp_bind: str = "127.0.0.1:8765"
    mcp_token: str | None = field(default=None, repr=False)
    mcp_allowed_hosts: tuple[str, ...] = ()
    mcp_account: str | None = None  # the account `add_chat` joins with; None: the only one
    sync_every_minutes: int = 0  # the host's sync schedule, for `add_chat` estimates; 0: none

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        on_delete = env.get("ON_DELETE", "").strip() or "tombstone"
        if on_delete not in ("tombstone", "purge"):
            raise ValueError("ON_DELETE must be 'tombstone' or 'purge'")
        backfill_years = check_years(_int(env, "BACKFILL_YEARS", 3), "BACKFILL_YEARS")
        data_dir = _str(env, "DATA_DIR")
        tdjson_path = _str(env, "TDJSON_PATH")
        return cls(
            database_url=_str(env, "DATABASE_URL"),
            data_dir=Path(data_dir) if data_dir else None,
            tdjson_path=Path(tdjson_path) if tdjson_path else None,
            tdlib_db_key=_str(env, "TDLIB_DB_KEY"),
            tg_api_id=_str(env, "TG_API_ID"),
            tg_api_hash=_str(env, "TG_API_HASH"),
            backfill_years=backfill_years,
            sweep_days=_int(env, "SWEEP_DAYS", 7),
            page_delay_ms=_int(env, "PAGE_DELAY_MS", 300),
            on_delete=cast(OnDelete, on_delete),
            window_gap_min=_int(env, "WINDOW_GAP_MIN", 30),
            window_max_chars=_int(env, "WINDOW_MAX_CHARS", 1500),
            window_max_msgs=_int(env, "WINDOW_MAX_MSGS", 40),
            embed_provider=_str(env, "EMBED_PROVIDER") or "voyage",
            embed_model=_str(env, "EMBED_MODEL"),
            voyage_api_key=_str(env, "VOYAGE_API_KEY"),
            openai_api_key=_str(env, "OPENAI_API_KEY"),
            mcp_bind=_str(env, "MCP_BIND") or "127.0.0.1:8765",
            mcp_token=_str(env, "MCP_TOKEN"),
            mcp_allowed_hosts=tuple(
                h.strip() for h in env.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
            ),
            mcp_account=_str(env, "MCP_ACCOUNT"),
            sync_every_minutes=_sync_every(_int(env, "SYNC_EVERY_MINUTES", 0)),
        )
