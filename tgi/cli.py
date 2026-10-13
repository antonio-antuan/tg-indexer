"""`tgi` command-line entry point."""

import json
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import typer
from typer._click.exceptions import UsageError  # typer vendors click; no public alias
from typer.core import TyperGroup

from tgi import __version__, chats, db, embed, ingest, search, selftest, status, td
from tgi import mcp as mcp_server
from tgi.cli_td import td_app
from tgi.config import Settings, check_years, env_file_warning
from tgi.td_fake import FakeTdClient
from tgi.td_port import FloodWait, JoinPending, NotJoinable, TdClient, TdError, parse_link


class _Group(TyperGroup):
    """Usage errors exit 1, not click's 2, which `tgi sync` keeps for a held lock (SPEC §5.3)."""

    def make_context(self, *args: Any, **kwargs: Any) -> Any:
        with _usage_exits_1():
            return super().make_context(*args, **kwargs)

    def invoke(self, ctx: Any) -> Any:  # subcommands parse their arguments in here
        with _usage_exits_1():
            return super().invoke(ctx)


@contextmanager
def _usage_exits_1() -> Iterator[None]:
    try:
        yield
    except UsageError as exc:
        exc.exit_code = 1
        raise


# No locals in tracebacks: they would print TDLib parameters, phones and message text.
app = typer.Typer(
    no_args_is_help=True, add_completion=False, cls=_Group, pretty_exceptions_show_locals=False
)
ENV_FILE = Path(".env")  # the one `make` and the README's `set -a; . ./.env` load


def _version(value: bool) -> None:
    if value:
        typer.echo(f"tgi {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", callback=_version, is_eager=True, help="Show the version and exit."
    ),
) -> None:
    """Telegram archive: sync, search and serve to an agent."""
    warning = env_file_warning(ENV_FILE)
    if warning:
        typer.echo(warning, err=True)


app.add_typer(td_app)  # its groups join the root: `tgi accounts …` and `tgi td …`

db_app = typer.Typer(no_args_is_help=True, help="Database maintenance.")
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending schema migrations to DATABASE_URL."""
    with db.connect(_database_url()) as conn:
        applied = db.migrate(conn)
    if applied:
        typer.echo("applied: " + ", ".join(f"{v:04d}" for v in applied))
    else:
        typer.echo("schema is up to date")


def _settings() -> Settings:
    """The environment's settings; a bad value is a usage error (exit 1), not a traceback."""
    try:
        return Settings.from_env()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise typer.BadParameter("DATABASE_URL is not set")
    return url


# Hidden test seam: drive the commands from a FakeTdClient scenario (SPEC §8) instead of
# TDLib.
_SCENARIO = typer.Option(None, "--scenario", hidden=True, exists=True, dir_okay=False)


class _Clients:
    """One TdClient per account, opened on first use and closed together."""

    def __init__(self, scenario: Path | None) -> None:
        self._scenario = scenario
        self._open: dict[str, TdClient] = {}

    def __call__(self, account_phone: str) -> TdClient:
        if account_phone not in self._open:
            if self._scenario is not None:
                self._open[account_phone] = FakeTdClient.from_file(self._scenario)
            else:
                try:
                    self._open[account_phone] = td.open_client(_settings(), account_phone)
                except td.SetupError as exc:
                    raise typer.BadParameter(str(exc)) from None
        return self._open[account_phone]

    def close(self) -> None:
        """Close every client; a failure is reported by type and code only."""
        for client in self._open.values():
            try:
                client.close()
            except Exception as exc:
                typer.echo(f"closing a TDLib client failed: {ingest.describe_error(exc)}", err=True)
        self._open.clear()


@contextmanager
def _tdlib_errors(what: str) -> Iterator[None]:
    """A TDLib error ends the command with exit 3, reported by type and code only."""
    try:
        yield
    except TdError as exc:
        typer.echo(f"{what} failed: {ingest.describe_error(exc)}", err=True)
        raise typer.Exit(3) from None


@contextmanager
def _sync_lock_or_exit(conn: Any, what: str) -> Iterator[None]:
    """Hold the sync lock while a command uses an account's TDLib session, which a running
    `tgi sync` holds too; exit 2 when it is taken, as `sync` does."""
    try:
        with ingest.sync_lock(conn):
            yield
    except ingest.SyncLocked:
        typer.echo(f"{what}: a tgi sync, embed, purge or forget is running; try again", err=True)
        raise typer.Exit(2) from None


# Supergroup ids are negative (-100…); without this click reads them as unknown options.
_CHAT_ID_ARGS = {"ignore_unknown_options": True}


@app.command("sync", context_settings=_CHAT_ID_ARGS)
def sync_command(
    chat_id: int | None = typer.Argument(None, help="The chat for --resweep-days."),
    max_minutes: float = typer.Option(150, "--max-minutes", min=0, help="Run time budget."),
    resweep_days: int | None = typer.Option(
        None, "--resweep-days", min=1, help="Only re-run CHAT_ID's sweep over N days."
    ),
    scenario: Path | None = _SCENARIO,
) -> None:
    """Catch up, backfill and sweep every enabled chat."""
    if (chat_id is None) != (resweep_days is None):
        raise typer.BadParameter("--resweep-days and CHAT_ID go together")
    resweep = None if chat_id is None or resweep_days is None else (chat_id, resweep_days)
    settings = _settings()
    try:
        embedder = embed.provider(settings)  # a missing key fails before any fetch
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    clients = _Clients(scenario)
    try:
        with db.connect(_database_url()) as conn:
            try:
                result = ingest.sync(
                    conn, clients, settings, max_minutes=max_minutes, resweep=resweep,
                    embedder=embedder, close_clients=clients.close,
                )  # fmt: skip
            except ingest.SyncLocked as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(2) from None
            except ingest.UnknownChat as exc:
                typer.echo(str(exc), err=True)
                raise typer.Exit(1) from None
    except (typer.Exit, typer.BadParameter):  # our own exits and messages
        raise
    except Exception as exc:
        # The message may quote TDLib responses or message text: print the type only.
        typer.echo(f"sync failed: {ingest.describe_error(exc)}", err=True)
        raise typer.Exit(3) from None
    finally:
        clients.close()
    stats = " ".join(f"{k}={v}" for k, v in result.stats.items())
    # ok, time_limited and flood_limited are normal runs (exit 0); errors raised above.
    typer.echo(f"run {result.run_id}: {result.status} {stats}")


@app.command("embed")
def embed_command(
    reindex: bool = typer.Option(
        False, "--reindex", help="Delete all stored embeddings and re-embed every window."
    ),
    max_windows: int | None = typer.Option(
        None, "--max-windows", min=1, help="Embed at most N stale windows this run."
    ),
) -> None:
    """Embed stale windows now (`tgi sync` does this too); --reindex after a provider change."""
    try:
        embedder = embed.provider(_settings())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    try:
        with db.connect(_database_url()) as conn, ingest.sync_lock(conn):
            if reindex:
                counts = embed.reindex(conn, embedder, max_windows=max_windows)
            else:
                counts = embed.embed_stale(conn, embedder, max_windows=max_windows)
    except ingest.SyncLocked as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    except embed.ModelMismatch:
        typer.echo("model_mismatch: stored embeddings come from another model;"
                   " run `tgi embed --reindex`", err=True)  # fmt: skip
        raise typer.Exit(3) from None
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"embed failed: {ingest.describe_error(exc)}", err=True)
        raise typer.Exit(3) from None
    counts.pop("embed_stopped", None)
    counts["tokens"] = embedder.tokens_used
    typer.echo(" ".join(f"{k}={v}" for k, v in counts.items()))


def _days(value: str) -> int:
    """`--older-than`: a whole number of days written `<N>d`; the value is never echoed."""
    match = re.fullmatch(r"([0-9]+)d", value)
    try:
        return int(match.group(1)) if match else -1
    except ValueError:  # more digits than int() accepts
        return -1


@app.command("purge")
def purge_command(
    deleted: bool = typer.Option(
        False, "--deleted", help="Purge tombstoned messages (required: the only scope)."
    ),
    older_than: str = typer.Option(
        "90d", "--older-than", help="Only those tombstoned more than N days ago, as Nd."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
) -> None:
    """Hard-delete messages tombstoned more than N days ago (D-TGI-3); asks first."""
    if not deleted:
        raise typer.BadParameter("--deleted is required (tombstoned messages are the only scope)")
    days = _days(older_than)
    if days < 0:
        raise typer.BadParameter("--older-than takes a whole number of days, such as 90d")
    before = ingest.purge_cutoff(datetime.now(UTC), days)
    try:
        # The lock is held from the count through the prompt to the delete, so the counts
        # shown are the rows deleted; a sync started meanwhile exits 2.
        with db.connect(_database_url()) as conn, ingest.sync_lock(conn):
            counts = ingest.tombstones_before(conn, before)
            if not counts:
                typer.echo("nothing to purge")
                return
            if not yes:
                for chat_id, n in counts.items():
                    typer.echo(f"chat {chat_id}: {n}")
                total = sum(counts.values())
                typer.confirm(f"Delete these {total} tombstoned messages for good?", abort=True)
            purged = ingest.purge_tombstones(conn, before)
    except ingest.SyncLocked as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    except (typer.Abort, typer.Exit, typer.BadParameter):
        raise
    except Exception as exc:
        typer.echo(f"purge failed: {ingest.describe_error(exc)}", err=True)
        raise typer.Exit(3) from None
    for chat_id, n in purged.items():
        typer.echo(f"purged chat {chat_id}: {n}")
    typer.echo(f"purged {sum(purged.values())}")


chats_app = typer.Typer(no_args_is_help=True, help="The chat allowlist.")
app.add_typer(chats_app, name="chats")
_ACCOUNT = typer.Option(..., "--account", help="Account phone, e.g. +34...")


@chats_app.command("list")
def chats_list(account: str = _ACCOUNT, scenario: Path | None = _SCENARIO) -> None:
    """Show the account's dialogs; '*' marks enabled chats. Exit 2 while a sync runs."""
    with db.connect(_database_url()) as conn:
        with _sync_lock_or_exit(conn, "chats list"):
            clients = _Clients(scenario)
            try:
                with _tdlib_errors("chats list"):
                    dialogs = clients(account).get_chats()
            finally:
                clients.close()
        enabled = chats.enabled_ids(conn, account)
    for chat in dialogs:
        mark = "*" if chat.chat_id in enabled else " "
        typer.echo(f"{mark} {chat.chat_id}\t{chat.kind}\t{chat.title}")


@chats_app.command("enable", context_settings=_CHAT_ID_ARGS)
def chats_enable(
    chat_id: int,
    account: str = _ACCOUNT,
    years: int | None = typer.Option(None, "--years", help="Backfill lookback."),
    scenario: Path | None = _SCENARIO,
) -> None:
    """Add a chat to the allowlist. Exit 2 while a sync runs."""
    try:  # before TDLib is opened or anything is written
        if years is not None:
            check_years(years)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--years") from None
    default_years = _settings().backfill_years
    with db.connect(_database_url()) as conn:
        with _sync_lock_or_exit(conn, "chats enable"):
            clients = _Clients(scenario)
            try:
                with _tdlib_errors("chats enable"):
                    chat = clients(account).get_chat(chat_id)
            finally:
                clients.close()
        try:
            depth = chats.enable(conn, chat, account, years, default_years=default_years)
        except chats.ChatConflict as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from None
    more = f", backfill continues to {depth.years} years on the next sync" if depth.reopens else ""
    typer.echo(f"enabled {chat_id}{more}")


@chats_app.command("add", context_settings=_CHAT_ID_ARGS)
def chats_add(
    link: str,
    account: str = _ACCOUNT,
    years: int | None = typer.Option(None, "--years", help="Backfill lookback."),
    scenario: Path | None = _SCENARIO,
) -> None:
    """Join a supergroup or channel from a t.me link and add it to the allowlist.

    LINK is https://t.me/name, an invite link https://t.me/+..., @name, or a chat id. The
    same join as the MCP add_chat tool; it waits for no sync (exit 2 while one runs)."""
    try:  # before TDLib is opened or anything is written
        parsed = parse_link(link)
    except ValueError:
        raise typer.BadParameter(
            "must be a t.me link, an @username or a chat id", param_hint="LINK"
        ) from None
    try:
        if years is not None:
            check_years(years)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--years") from None
    settings = _settings()

    def telegram() -> tuple[str, TdClient]:
        if scenario is not None:
            return account, FakeTdClient.from_file(scenario)
        return account, td.open_client(settings, account)

    def refuse(message: str, code: int = 1) -> typer.Exit:
        typer.echo(message, err=True)
        return typer.Exit(code)

    with db.connect(_database_url()) as conn:
        try:
            joined = mcp_server.join_chat(conn, telegram, parsed, years, settings.backfill_years)
        except ingest.SyncLocked:
            raise refuse("a tgi sync, embed, purge or forget is running; try again", 2) from None
        except td.SetupError as exc:
            raise typer.BadParameter(str(exc)) from None
        except NotJoinable as exc:
            kind = exc.kind
            raise refuse(f"only supergroups and channels can be added, not a {kind} chat") from None
        except chats.ChatConflict as exc:
            raise refuse(str(exc)) from None
        except JoinPending:
            raise refuse("the link sent a join request; run this again once approved") from None
        except FloodWait as exc:
            raise refuse(f"Telegram asks to wait {exc.retry_after}s", 3) from None
        except TdError as exc:
            if mcp_server.not_found(exc):
                raise refuse("no chat this account can join behind the link") from None
            raise refuse(f"chats add failed: {ingest.describe_error(exc)}", 3) from None
    chat = joined.chat
    more = ", backfill continues on the next sync" if joined.depth.reopens else ""
    typer.echo(
        f"enabled {chat.chat_id} ({chat.kind}, {joined.depth.years} years{more}): {chat.title}"
    )


@chats_app.command("disable", context_settings=_CHAT_ID_ARGS)
def chats_disable(chat_id: int) -> None:
    """Stop syncing a chat; its data is kept."""
    with db.connect(_database_url()) as conn:
        if not chats.disable(conn, chat_id):
            typer.echo(f"unknown chat {chat_id}", err=True)
            raise typer.Exit(1)
    typer.echo(f"disabled {chat_id}")


@chats_app.command("forget", context_settings=_CHAT_ID_ARGS)
def chats_forget(
    chat_id: int, yes: bool = typer.Option(False, "--yes", help="Skip the confirmation.")
) -> None:
    """Delete a chat's messages, windows and embeddings."""
    if not yes:
        typer.confirm(f"Delete chat {chat_id} and all its stored data?", abort=True)
    # Under the sync lock, so it cannot change the counts a waiting `tgi purge` showed.
    try:
        with db.connect(_database_url()) as conn, ingest.sync_lock(conn):
            forgotten = chats.forget(conn, chat_id)
    except ingest.SyncLocked as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(2) from None
    if not forgotten:
        typer.echo(f"unknown chat {chat_id}", err=True)
        raise typer.Exit(1)
    typer.echo(f"forgot {chat_id}")


tags_app = typer.Typer(no_args_is_help=True, help="Chat tags; search filters use confirmed ones.")
app.add_typer(tags_app, name="tags")


def _tag_result(done: bool, missing: str) -> None:
    if not done:
        typer.echo(missing, err=True)
        raise typer.Exit(1)


@tags_app.command("propose", context_settings=_CHAT_ID_ARGS)
def tags_propose(
    chat_id: int,
    tag: str,
    by: str = typer.Option("owner", "--by", help="Who proposes it."),
) -> None:
    """Propose a tag for a chat (does not filter searches until confirmed)."""
    with db.connect(_database_url()) as conn:
        try:
            done = chats.propose_tag(conn, chat_id, tag, by)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
    _tag_result(done, f"unknown chat {chat_id}")
    typer.echo(f"proposed {tag.strip()} for {chat_id}")


@tags_app.command("confirm", context_settings=_CHAT_ID_ARGS)
def tags_confirm(chat_id: int, tag: str) -> None:
    """Confirm a chat's tag, adding it if it was not proposed."""
    with db.connect(_database_url()) as conn:
        try:
            done = chats.confirm_tag(conn, chat_id, tag)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from None
    _tag_result(done, f"unknown chat {chat_id}")
    typer.echo(f"confirmed {tag.strip()} for {chat_id}")


@tags_app.command("remove", context_settings=_CHAT_ID_ARGS)
def tags_remove(chat_id: int, tag: str) -> None:
    """Remove a chat's tag, proposed or confirmed."""
    with db.connect(_database_url()) as conn:
        done = chats.remove_tag(conn, chat_id, tag)
    _tag_result(done, f"chat {chat_id} has no tag {tag.strip()}")
    typer.echo(f"removed {tag.strip()} from {chat_id}")


_QUERIES = typer.Argument(..., help="One or more queries; hits are fused.")
_TAGS = typer.Option(None, "--tag", help="Confirmed tag (repeatable).")
_CHATS = typer.Option(None, "--chat", help="Chat id (repeatable).")


@app.command("search")
def search_command(
    queries: list[str] = _QUERIES,
    tag: list[str] | None = _TAGS,
    chat: list[int] | None = _CHATS,
    since: str | None = typer.Option(
        None, "--since", help="First day YYYY-MM-DD, or an ISO date-time (UTC)."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Last day YYYY-MM-DD, or an ISO date-time (UTC)."
    ),
    limit: int = typer.Option(20, "--limit", min=1, max=search.MAX_LIMIT),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
    as_json: bool = typer.Option(False, "--json", help="Print the hits as JSON."),
    mode: str = typer.Option(
        "hybrid", "--mode", help="hybrid, keyword or semantic.", show_default=True
    ),
) -> None:
    """Search the archive; prints windows with message links."""
    if mode not in ("hybrid", "keyword", "semantic"):
        raise typer.BadParameter("--mode must be hybrid, keyword or semantic")
    try:
        with db.connect(_database_url()) as conn:
            hits = search.search(
                conn,
                queries,
                tags=tag or None,
                chat_ids=chat or None,
                since=_when(since, "--since"),
                until=_when(until, "--until"),
                mode=cast(search.Mode, mode),
                limit=limit,
                include_deleted=include_deleted,
            )
    except (search.EmbedderUnavailable, search.ModelMismatch) as exc:
        typer.echo(f"search failed: {exc}", err=True)
        raise typer.Exit(3) from None
    except ValueError as exc:  # bad range, empty queries
        raise typer.BadParameter(str(exc)) from None
    if as_json:
        typer.echo(json.dumps([h.as_dict() for h in hits], ensure_ascii=False, indent=2))
        return
    for h in hits:
        where = "message" if h.window_id is None else f"window {h.window_id}"
        typer.echo(f"{h.score:.4f}  {h.chat_title}  {h.start_date:%Y-%m-%d %H:%M}  ({where})")
        for m in h.messages:
            typer.echo(f"  {m.date:%Y-%m-%d %H:%M} {m.sender_name or '?'}: {m.text}")
            typer.echo(f"    {m.link or '(no link)'}")


def _when(value: str | None, name: str) -> date | None:
    """A date (a whole UTC day) or an ISO date-time (UTC without a zone)."""
    if value is None:
        return None
    try:
        return search.parse_when(value)
    except ValueError:
        raise typer.BadParameter(f"{name} must be YYYY-MM-DD or an ISO date-time") from None


@app.command("recent")
def recent_command(
    tag: list[str] | None = _TAGS,
    chat: list[int] | None = _CHATS,
    hours: int | None = typer.Option(
        None, "--hours", min=1, max=search.MAX_RECENT_HOURS, help="Back from now (default 24)."
    ),
    since: str | None = typer.Option(None, "--since", help="YYYY-MM-DD or ISO date-time (UTC)."),
    until: str | None = typer.Option(
        None, "--until", help="Exclusive end: YYYY-MM-DD (that whole day) or ISO date-time."
    ),
    limit: int = typer.Option(20, "--limit", min=1, max=search.MAX_LIMIT),
    cursor: str | None = typer.Option(None, "--cursor", help="next_cursor of a previous page."),
    as_json: bool = typer.Option(False, "--json", help="Print the answer as JSON."),
) -> None:
    """What's new: windows with messages in a time range, newest first, no query.

    --until is exclusive and defaults to now. --cursor continues a listing: it keeps the
    range and filters, so the other options but --limit are ignored with it."""
    if cursor is None and hours is not None and since is not None:
        raise typer.BadParameter("give --hours or --since, not both")
    if cursor is None and until is not None and since is None:
        raise typer.BadParameter("--until needs --since")
    now = datetime.now(UTC)
    if cursor is not None:  # the cursor carries the range: --since/--until are not read
        since = until = None
    start = None if cursor else _when(since, "--since") or now - timedelta(hours=hours or 24)
    try:
        with db.connect(_database_url()) as conn:
            answer = search.recent(
                conn,
                since=start,
                until=_when(until, "--until"),
                tags=tag or None,
                chat_ids=chat or None,
                limit=limit,
                cursor=cursor,
                now=now,
            )
    except ValueError as exc:  # bad range, a cursor this tool did not print
        raise typer.BadParameter(str(exc)) from None
    if as_json:
        typer.echo(json.dumps(answer.as_dict(), ensure_ascii=False, indent=2))
        return
    for h in answer.hits:
        typer.echo(f"{h.chat_title}  {h.start_date:%Y-%m-%d %H:%M}  (window {h.window_id})")
        for m in h.messages:
            typer.echo(f"  {m.date:%Y-%m-%d %H:%M} {m.sender_name or '?'}: {m.text}")
            typer.echo(f"    {m.link or '(no link)'}")
    if answer.next_cursor is not None:
        typer.echo(f"more: tgi recent --cursor {answer.next_cursor}")


@app.command("status")
def status_command(
    chat: list[int] | None = _CHATS,
    as_json: bool = typer.Option(False, "--json", help="Print the answer as JSON."),
) -> None:
    """How current the archive is: runs, embedding backlog, per chat history and last sync."""
    settings = _settings()
    with db.connect(_database_url()) as conn:
        answer = status.status(conn, chat or None, default_years=settings.backfill_years,
                               sync_every_minutes=settings.sync_every_minutes)  # fmt: skip
    if as_json:
        typer.echo(json.dumps(answer, ensure_ascii=False, indent=2))
        return
    last, ok = answer["last_run"], answer["last_ok_run"]
    typer.echo("sync running now" if answer["sync_running"] else "no sync running")
    if (upcoming := answer["schedule"]["next_run_at"]) is not None:
        typer.echo(f"next scheduled run (estimate): {upcoming}")
    if last is None:
        typer.echo("last run: none yet")
    else:
        typer.echo(f"last run: {last['status']}, finished {last['finished_at'] or '-'}")
        typer.echo(f"  {' '.join(f'{k}={v}' for k, v in last['stats'].items())}")
    if ok is not None and ok != last:
        typer.echo(f"last ok run: finished {ok['finished_at']}")
    w = answer["windows"]
    typer.echo(f"windows: {w['total']}, waiting for embedding: {w['waiting_for_embedding']}")
    typer.echo("chats (* = enabled):")
    for c in answer["chats"]:
        history = "complete" if c["history_complete"] else "incomplete"
        rebuild = ", messages wait for the window step" if c["windows_need_rebuild"] else ""
        typer.echo(f"{'*' if c['enabled'] else ' '} {c['chat_id']}  {c['title']}")
        typer.echo(
            f"    history {history} ({c['backfill_years']}y, oldest"
            f" {c['oldest_message_at'] or '-'}), newest {c['newest_message_at'] or '-'}"
        )
        typer.echo(
            f"    last sync turn {c['last_sync_at'] or 'never'}, waiting for embedding"
            f" {c['windows_waiting_for_embedding']}{rebuild}"
        )


_REFERENCE = typer.Option(
    None,
    "--scenario",
    exists=True,
    dir_okay=False,
    help="Load this scenario into a throwaway database and check its expectations.",
)


@app.command("selftest")
def selftest_command(scenario: Path | None = _REFERENCE) -> None:
    """Run the reference query (SPEC §5.7) through an MCP client over HTTP: against the
    server at MCP_BIND with MCP_TOKEN, or with --scenario against a test server."""
    settings = _settings()
    try:
        if scenario is not None:
            data = json.loads(scenario.read_text())
            outcome, counts = selftest.run_scenario(_database_url(), data, settings)
        else:
            if not settings.mcp_token:
                raise typer.BadParameter("MCP_TOKEN is not set")
            url = f"http://{settings.mcp_bind}/mcp"
            outcome, counts = selftest.run(url, settings.mcp_token), {}
    except selftest.Failed as exc:
        typer.echo(f"selftest failed: {exc}", err=True)
        raise typer.Exit(1) from None
    except selftest.Unreachable as exc:
        typer.echo(f"selftest failed: {exc}", err=True)
        raise typer.Exit(3) from None
    except typer.BadParameter:
        raise
    except Exception as exc:
        typer.echo(f"selftest failed: {ingest.describe_error(exc)}", err=True)
        raise typer.Exit(3) from None
    found = {"tags": outcome.tags, "hits": len(outcome.hits), **counts}
    found["context"] = len(outcome.context)
    typer.echo("selftest ok: " + " ".join(f"{k}={v}" for k, v in found.items()))


mcp_app = typer.Typer(help="The MCP server for an agent (SPEC §5.6).")
app.add_typer(mcp_app, name="mcp")

# A 32-byte token in URL-safe base64; a hex one is longer.
MIN_TOKEN_CHARS = 43


def _mcp_telegram(settings: Settings) -> mcp_server.Telegram:
    """Opens the account the MCP `add_chat` tool joins with: MCP_ACCOUNT, or the only
    logged-in one. Called per tool call, under the sync lock."""

    def open_account() -> tuple[str, TdClient]:
        phones = [settings.mcp_account] if settings.mcp_account else td.logged_in_accounts(settings)
        if len(phones) != 1:
            raise td.SetupError("set MCP_ACCOUNT: there is no logged-in account or several")
        return phones[0], td.open_client(settings, phones[0])

    return open_account


@mcp_app.callback(invoke_without_command=True)
def mcp_main(
    ctx: typer.Context,
    http: bool = typer.Option(False, "--http", help="Streamable HTTP on MCP_BIND, MCP_TOKEN."),
    stdio: bool = typer.Option(False, "--stdio", help="stdin/stdout, for local debugging."),
    allow_any_bind: bool = typer.Option(
        False, "--allow-any-bind", help="Let --http listen beyond loopback and Docker networks."
    ),
) -> None:
    """Serve the search tools: `tgi mcp --http` or `tgi mcp --stdio`."""
    if ctx.invoked_subcommand is not None:
        if http or stdio or allow_any_bind:
            raise typer.BadParameter("--http and --stdio do not go with a subcommand")
        return
    if http == stdio:
        raise typer.BadParameter("give exactly one of --http and --stdio")
    if allow_any_bind and not http:
        raise typer.BadParameter("--allow-any-bind goes with --http")
    settings = _settings()
    server = mcp_server.build_server(
        _database_url(),
        # read-only: no tool can reach Telegram, so the server never opens the account
        telegram=None if settings.mcp_read_only else _mcp_telegram(settings),
        default_years=settings.backfill_years,
        sync_every_minutes=settings.sync_every_minutes,
        read_only=settings.mcp_read_only,
    )
    if settings.mcp_read_only:
        typer.echo("tgi mcp: MCP_READ_ONLY: suggest_tags and add_chat are off", err=True)
    if stdio:
        server.run("stdio")
        return
    token = settings.mcp_token
    if token is None:
        raise typer.BadParameter("MCP_TOKEN is not set; generate one with `tgi mcp token`")
    if len(token) < MIN_TOKEN_CHARS:
        raise typer.BadParameter("MCP_TOKEN is too short; generate one with `tgi mcp token`")
    try:
        host, port = mcp_server.parse_bind(settings.mcp_bind)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    interfaces = mcp_server.read_interfaces()
    refusal = mcp_server.check_bind(host, interfaces, allow_any_bind)
    if refusal is not None:
        raise typer.BadParameter(refusal)
    if mcp_server.check_bind(host, interfaces) is not None:
        typer.echo(f"tgi mcp: --allow-any-bind: listening on {host}", err=True)
    security = mcp_server.transport_security(host, port, interfaces, settings.mcp_allowed_hosts)
    import uvicorn  # only the HTTP server needs it

    typer.echo(f"tgi mcp: serving http://{settings.mcp_bind}/mcp", err=True)
    # No access log: a request target can carry anything a client put there (#29).
    app = mcp_server.http_app(server, token, security)
    uvicorn.run(app, host=host, port=port, access_log=False)


@mcp_app.command("token")
def mcp_token() -> None:
    """Print a fresh MCP_TOKEN (32 random bytes) for .env."""
    typer.echo(mcp_server.new_token())
