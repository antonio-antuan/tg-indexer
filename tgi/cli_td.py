"""`tgi accounts add` and `tgi td spike`: the commands that talk to real TDLib (P1.T3).

Both print ids, counts and TDLib error codes only: never the phone, a code, a password,
a key, message text or a TDLib payload (A10).
"""

import time

import typer

from tgi import ingest, search, td
from tgi.config import Settings
from tgi.td_port import FloodWait, TdError, server_message_id

td_app = typer.Typer()
accounts_app = typer.Typer(no_args_is_help=True, help="Telegram accounts (TDLib sessions).")
td_app.add_typer(accounts_app, name="accounts")
spike_app = typer.Typer(no_args_is_help=True, help="Checks against real TDLib.")
td_app.add_typer(spike_app, name="td")

_PHONE = typer.Option(..., "--phone", help="The account's phone, e.g. +34600000000.")
_ACCOUNT = typer.Option(..., "--account", help="Account phone, e.g. +34...")


class TerminalPrompt:
    """Login input from the terminal; the code and the password are not echoed."""

    def code(self) -> str:
        return str(typer.prompt("Login code (sent to your Telegram apps)", hide_input=True))

    def password(self, hint: str) -> str:
        label = "2FA password" + (f" (hint: {hint})" if hint else "")
        return str(typer.prompt(label, hide_input=True))

    def rejected(self, what: str, error: TdError) -> None:
        typer.echo(f"the {what} was rejected (TDLib {error.code}); try again", err=True)


def _failed(what: str, exc: BaseException) -> typer.Exit:
    detail = ingest.describe_error(exc)
    if isinstance(exc, FloodWait):
        detail += f", retry after {exc.retry_after}s"
    typer.echo(f"{what} failed: {detail}", err=True)
    return typer.Exit(3)


def _open(phone: str, prompt: td.Prompt | None) -> td.TdlibClient:
    try:
        return td.open_client(Settings.from_env(), phone, prompt)
    except (td.SetupError, ValueError) as exc:  # ValueError: Settings rejects a config value
        raise typer.BadParameter(str(exc)) from None
    except TdError as exc:
        raise _failed("login" if prompt else "opening the account", exc) from None


def _close(client: td.TdlibClient) -> None:
    try:
        client.close()
    except Exception as exc:
        typer.echo(f"closing the TDLib client failed: {ingest.describe_error(exc)}", err=True)


@accounts_app.command("add")
def accounts_add(phone: str = _PHONE) -> None:
    """Log an account in (phone, code, 2FA password); the session lands in DATA_DIR/accounts."""
    client = _open(phone, TerminalPrompt())
    _close(client)
    typer.echo("account ready")


@spike_app.command("spike", context_settings={"ignore_unknown_options": True})
def td_spike(
    chat_id: int,
    account: str = _ACCOUNT,
    pages: int = typer.Option(1, "--pages", min=1, max=10, help="getChatHistory pages."),
) -> None:
    """One cold-start getChatHistory run on CHAT_ID: prints ids and timings only."""
    started = time.monotonic()
    client = _open(account, None)  # validates the settings first
    settings = Settings.from_env()
    try:
        typer.echo(f"tdlib {td.hub(settings).version()}")
        chat = client.get_chat(chat_id)
        typer.echo(f"chat {chat_id} kind={chat.kind} username={'yes' if chat.username else 'no'}")
        from_id, first = 0, None
        for n in range(1, pages + 1):
            if n > 1:
                time.sleep(settings.page_delay_ms / 1000)
            t0 = time.monotonic()
            page = client.get_chat_history(chat_id, from_id, ingest.PAGE_LIMIT)
            ms = (time.monotonic() - t0) * 1000
            typer.echo(f"page {n}: from_message_id={from_id} messages={len(page)} {ms:.0f}ms")
            if from_id and page:
                typer.echo(f"  from_inclusive={page[0].message_id == from_id}")
            for m in page:
                try:
                    server = str(server_message_id(m.message_id))
                except ValueError:
                    server = "none (not a server message id)"
                typer.echo(f"  message_id={m.message_id} server_message_id={server}")
            if not page:
                typer.echo("  end of history")
                break
            first = first or page[0]
            from_id = page[-1].message_id
        if first is not None and first.message_id % (1 << 20) == 0:
            sid = server_message_id(first.message_id)
            url = search.link(chat.kind, chat_id, chat.username, sid)
            typer.echo(f"link for message_id={first.message_id}: {url or '(no link)'}")
    except TdError as exc:
        raise _failed("spike", exc) from None
    finally:
        _close(client)
    typer.echo(f"done in {time.monotonic() - started:.1f}s")
