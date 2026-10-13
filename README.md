# tg-indexer
Personal Telegram archive (TDLib) with keyword + LLM search

## Development

Requirements: Docker (compose), [uv](https://docs.astral.sh/uv/), `make`,
[actionlint](https://github.com/rhysd/actionlint).

```sh
cp .env.example .env && chmod 600 .env   # set POSTGRES_PASSWORD (letters and digits)
uv sync
make ci                                  # starts the compose db, then ruff, mypy, actionlint, pytest
uv run tgi db migrate                    # with .env loaded: set -a; . ./.env; set +a
```

GitHub Actions are disabled on this repo; `.github/workflows/ci.yml` mirrors `make ci`
and the pasted local `make ci` output is the CI evidence.
What the tests cannot verify (real TDLib, real links, the compose database) is listed in
[docs/MANUAL-CHECKS.md](docs/MANUAL-CHECKS.md).

Tests need a Postgres 16 with `pgvector` and `pg_trgm` at `DATABASE_URL` and a role that
may create databases; each session creates and drops its own `tgi_test_*` databases. To
use a Postgres other than the compose one, pass both on the make command line (values in
`.env` override exported variables, command-line values override `.env`):

```sh
make ci DB_UP=true DATABASE_URL=postgresql://user@127.0.0.1:55432/postgres
```

## TDLib and accounts

```sh
scripts/tdlib.sh            # libtdjson 1.8.67 into DATA_DIR/lib (or TDJSON_PATH), from PyPI
scripts/tdlib.sh build      # or build it from the TDLib sources (cmake, 30-60 min)
uv run tgi accounts add --phone +34...             # log in: code, then the 2FA password
uv run tgi td spike <chat_id> --account +34... [--pages 2]  # history paging: ids only
```

`scripts/tdlib.sh` downloads the `tdjson` 1.8.67 wheel (Linux x86_64; TDLib is BSL-1.0,
the wheel's packaging MIT), checks its pinned sha256, installs the library and checks that
it loads and reports 1.8.67. `build` compiles the same TDLib commit instead; its header
lists the packages it needs. Never use a library built for another system (e.g. in a
container).

`tgi accounts add` keeps the session in `DATA_DIR/accounts/<phone>/` (mode 0700, encrypted
with `TDLIB_DB_KEY`); the code and the password are not echoed. The other commands only
open a logged-in account and never prompt. TDLib's own log (errors only) goes to
`DATA_DIR/accounts/tdlib.log`, not to the terminal; if TDLib cannot open that file, the
command stops (exit 1) before it creates a client.

`tgi` does not read `.env` itself, but every command warns on stderr when the `.env` in the
current directory is more open than 0600 (the line names the file and its mode only).
A TDLib error in `tgi chats list` or `tgi chats enable` exits 3 with the error type and
code, as `sync` does. `tgi chats add` takes the same links as the MCP `add_chat` tool and
refuses the same chats: a private chat, a basic group, a chat enabled under another account,
a link that sends a join request (run it again once approved) or leads nowhere all exit 1
with nothing enabled; it exits 2 while a sync (or `embed`, `purge`, `forget`) holds the lock,
and 3 on another TDLib error or a flood wait.

## Sync

```sh
uv run tgi chats list --account +34...             # the account's dialogs; * = enabled by it
uv run tgi chats enable <chat_id> --account +34... [--years N]  # --years (1–100) is kept until changed; a larger value continues the backfill on the next sync; a smaller one deletes nothing
uv run tgi chats add <link> --account +34... [--years N]  # join a supergroup or channel from a t.me link (or @name) and enable it, like the MCP add_chat
uv run tgi sync [--max-minutes 150]                # catch-up, interleaved backfill, sweep
uv run tgi sync --resweep-days N <chat_id>         # only that chat's sweep, N days deep
uv run tgi chats disable <chat_id>                 # stop syncing, keep data
uv run tgi chats forget <chat_id>                  # delete the chat's data (asks first)
uv run tgi embed [--reindex] [--max-windows N]     # embed stale windows; --reindex redoes all
uv run tgi purge --deleted [--older-than 90d] [--yes]  # hard-delete old tombstones (asks first)
```

`tgi sync` exits 0 when the run ends `ok`, `time_limited` or `flood_limited` (partial runs
are normal: a flood wait longer than the remaining budget ends the run at once, and status
and stats are printed), 2 if another sync holds the lock (no `sync_run` row), 1 on a usage
error or a `--resweep-days` chat that is not in the archive, and 3 when the run ends in
`error`, printing only the error type (the run is recorded as `error` in `sync_run`);
progress is checkpointed per page either way. A run killed before it could
record its end leaves its row `running`; the next run marks it `interrupted`.

The sweep re-pages the last `SWEEP_DAYS` (7) days of each chat: edits are applied, and
messages missing from a fully paged range are tombstoned (`ON_DELETE=tombstone`, hidden
from search) or deleted (`ON_DELETE=purge`). A sweep cut short deletes nothing, and so
does one that gets an empty page while the archive still holds live window messages that
page should have covered (logged as `incomplete: untrusted empty page`; the next run sweeps
again). A chat emptied entirely while enabled therefore keeps its rows: `tgi chats forget`.
Last, every run (partial ones too) rebuilds the conversation windows of each chat whose
messages changed, from the window holding the lowest changed message onward
(`chat.windows_dirty_from`); unchanged windows keep their id and embedding, changed ones
are marked `stale` for embedding. Window caps: `WINDOW_GAP_MIN` (30), `WINDOW_MAX_CHARS`
(1500), `WINDOW_MAX_MSGS` (40).

Then, within the same time budget, `stale` windows are embedded in batches of 64 with the
`EMBED_PROVIDER`: `voyage` (the default; `VOYAGE_API_KEY`, `EMBED_MODEL` default
`voyage-4`), `openai` (`OPENAI_API_KEY`, default `text-embedding-3-small`) or `fake`
(deterministic token-hash vectors, what the tests use), all 1024 dims. A missing or empty key is a usage error (exit 1,
`VOYAGE_API_KEY is not set`) before anything runs. The HTTP providers use the environment's proxy and `SSL_CERT_FILE`
settings; a 429 with `Retry-After` up to 30 s is waited out once, anything else fails the
batch. `tgi embed --max-windows N` embeds at most N windows (a cheap first try with a paid
provider) and prints the `tokens=` the provider billed. A
failed batch stays `stale` for the next run and does not fail it (`embed_failed` in the
stats). If stored embeddings come from another `EMBED_MODEL` or dimension, nothing is
embedded and the run ends `error` (`model_mismatch`, exit 3): run `tgi embed --reindex`,
which drops every embedding and embeds all windows again (with a paid provider this costs
a full re-embedding). `tgi embed` takes the sync lock (exit 2 if a sync is running).

`tgi chats forget` takes the sync lock too: it exits 2 while a sync, `tgi embed` or
`tgi purge` holds it, and deletes nothing.

`tgi purge --deleted` hard-deletes tombstoned messages whose `deleted_at` is more than
`--older-than` days ago (`<N>d`, default `90d`; `0d` purges every tombstone); live messages
are never touched, in enabled and disabled chats alike. `--deleted` is required, and any
other age format is a usage error (exit 1). It prints the count per chat and asks before
deleting. The prompt (`typer.confirm`) accepts `y` or `yes` in any case; `n`, `no`, an
empty answer or no terminal deletes nothing (exit 1), and any other answer is asked again,
so only a yes deletes. `--yes` skips the question. With nothing to purge it says so and exits 0. Each affected chat's
`windows_dirty_from` is lowered to its lowest purged id before the delete, all in one
transaction (the next sync's rebuild normally changes nothing: tombstones belong to no
window). Purge holds the sync lock from the count through the question to the delete, so
the counts shown are the rows deleted; it exits 2 if a sync is running, and a scheduled sync
that starts while the question is open exits 2 and retries on its next run.

These commands use the real TDLib through the account's session; tests pass `--scenario`
(a JSON file) to run them against a fake TDLib instead.

## Search

```sh
uv run tgi search "продление ВНЖ" "renovación TIE" [--json]   # several queries are fused
uv run tgi search QUERY --mode hybrid|keyword|semantic        # default hybrid
uv run tgi search QUERY --tag TAG --chat CHAT_ID --since 2026-01-01 --until 2026-03-31
uv run tgi search QUERY --include-deleted --limit 50          # tombstoned messages too
uv run tgi recent [--hours 24 | --since … [--until …]] [--tag TAG] [--chat ID] [--json]  # what's new, no query
uv run tgi tags propose|confirm|remove <chat_id> <tag>      # only confirmed tags filter
```

`--mode keyword` is full-text in russian, spanish and english, plus a trigram match for short tokens such as `ВНЖ` or `TIE` (strict word
similarity ≥ 0.5, so `TIE` does not match `tiene`). Queries use web-search syntax:
`"a phrase"` matches the phrase, and `-word` or `-"a phrase"` excludes on both legs (`-a-b` excludes the hyphenated compound);
phrases and excluded words are not trigram-matched. An excluded term drops every message it
matches in any of the three languages (so `-tiene`, a Spanish stop word, still excludes
through the russian and english configs); a word that all three drop (`-a`) excludes
nothing. Matching messages are returned as their
conversation windows, ranked by reciprocal-rank fusion across the queries, each message with
a `t.me` link where Telegram has one (supergroups and channels; not basic groups or private
chats). `--since`/`--until` take a date, a whole UTC day, or an ISO date-time, an exact
inclusive instant (UTC when it has no zone); `--tag` and `--chat` may be repeated.

`tgi recent` answers "what's new" without a topic: the conversation windows with messages
posted in a time range, newest activity first (by each window's last message), each showing
only its messages inside the range. The range is `--hours` back from now (default 24, at most
744), or `--since` and optionally `--until`: `since` ≤ date < `until`. Unlike `search`, `--until`
is exclusive (a date means up to the end of that UTC day) and defaults to now, so consecutive
ranges meet without gap or overlap. Pages hold `--limit` windows (at most 50); when more follow
it prints `more: tgi recent --cursor …` (with `--json`, `next_cursor`). The cursor keeps the
range and filters, so `--cursor` alone continues (other options but `--limit` are ignored with
it), and a short or even empty page can still have more. Paging runs over live data: a sync
between pages can move a window, which then repeats or drops out. Messages a sync has stored but
not yet put in a window (the window step runs at the end of every sync) and tombstoned messages
are not listed.

`--mode semantic` embeds each query with the `EMBED_PROVIDER` and takes the 100 nearest
embedded windows (cosine) under the same tag, chat and date filters (a window counts when
it overlaps the dates); when a selective filter leaves the HNSW index short of `--limit`,
the leg reruns as an exact scan. `hybrid` (the default) fuses both legs in the same
reciprocal-rank fusion, so a window found by both ranks above one found by one. The
semantic leg ignores negation: excluded terms are left out of the embedded text, but a
window can still come back through that leg, so use `--mode keyword` when an exclusion must
hold; a negation-only query (`-word`) runs on the keyword legs only. A provider that fails
or has no key, or stored embeddings from another model (run `tgi embed --reindex`), exit 3
in `hybrid` and `semantic`; `keyword` never calls the provider. Until anything is
embedded, `hybrid` returns the keyword result and embeds nothing. How an agent should use
all this over MCP (queries, negation, filters, citing, errors) is in
[docs/AGENT-NOTES.md](docs/AGENT-NOTES.md).

## MCP server

```sh
uv run tgi mcp token        # a fresh MCP_TOKEN (32 random bytes) for .env
uv run tgi mcp --http       # streamable HTTP at http://MCP_BIND/mcp, Bearer MCP_TOKEN
uv run tgi mcp --stdio      # the same tools over stdin/stdout, for local debugging
```

Eight tools (SPEC §5.6): `list_chats`, `list_tags`, `search` (as `tgi search --json`, `limit`
≤ 50), `recent` (as `tgi recent --json`; `hours` or `since`/`until`, a `cursor` to page), `get_thread`, `get_context` (`before + after` ≤ 100, at most 100 messages with the message itself), `suggest_tags`, which only
adds `proposed` tags (`proposed_by` `mcp`), and `add_chat`, which joins a supergroup or
channel from a `t.me` link with `MCP_ACCOUNT` (default: the only logged-in account) and
enables it; private chats and basic groups are refused, and during a `tgi sync` it answers
`busy`. The server then needs the TDLib settings of `.env` (`DATA_DIR`, `TDLIB_DB_KEY`,
`TG_API_ID`, `TG_API_HASH`) as well.

With `MCP_READ_ONLY=1` (also `true`, `yes`, `on`; unset, empty, `0`, `false`, `no`, `off` mean
off, anything else is a usage error) the server offers only the six read tools:
`suggest_tags` and `add_chat` are not registered at all, a call to them gets `unknown_tool`,
and the server never opens the Telegram account, so it needs no TDLib settings. Add chats with
`tgi chats add` then. Every tool carries MCP annotations so a client can tell reads from
writes: the read tools `readOnlyHint: true`, `suggest_tags` and `add_chat` `readOnlyHint:
false` (`add_chat` also `openWorldHint: true`). A client can then run reads unattended and ask
the user before a write; in Hermes that is `trust: untrusted` on the server. The annotations
are hints for well-behaved clients; `MCP_READ_ONLY` is what the server enforces.

A tool error's text is the SDK's prefix
`Error executing tool <name>: ` followed by a JSON object `{"code": ..., "message": ...}`;
`code` is `unknown_chat`, `unknown_tag`, `bad_range`, `bad_argument` (also a schema or type
error, such as `limit: true`, or a value Postgres cannot store), `embedder_unavailable` or
`model_mismatch` (for the last two, `mode="keyword"` still works). The message names the
wrong argument but never quotes it, since the SDK logs it. A call to a tool that does not
exist gets `unknown_tool`, after the prefix `Error executing tool: ` without the name. `--http` refuses to start without an `MCP_TOKEN` of at least 43 characters; a
request without the token, or with another, gets `401` with an empty body, and the log
counts these without the header. There is no access log: request paths and query strings
are never logged.

### The reference query

```sh
uv run tgi selftest --scenario tests/scenarios/reference.json   # a throwaway test server
uv run tgi selftest                         # the server at MCP_BIND, with MCP_TOKEN
```

`tgi selftest` is the scripted MCP client of SPEC §5.7 (A9), also runnable as
`tests/e2e/reference_query.py`: over HTTP it calls `list_tags`, then `search` with the four
reference queries, tag `valencia` and the last 30 days, then `get_context` on the top hit,
and checks that every link is a `t.me` message link and that the context returns the top
message with the same link. It prints counts only, never message text. With `--scenario`
it creates a database `tgi_selftest_<random>` next to the one in `DATABASE_URL` (the role
needs CREATEDB; the configured database is never opened), loads the scenario with the fake
TDLib and the fake embedder, serves it on `127.0.0.1` and a free port, checks the
scenario's `reference` expectations (at least 3 seeded messages found, none of the
excluded, every unrelated one ranked below every seeded one), and drops the database,
on failure too (also on SIGINT and SIGTERM; a SIGKILL cannot be caught and leaves the
`tgi_selftest_<random>` database behind, to drop by hand); it drops no database it did not
create in the same run. Exit 0 on success, 1 when a check fails (with the reason), 3 on any
other error: a refused token (`401`), another HTTP error status or no connection is named,
anything else is reported by its type only. Without `--scenario` it queries the
running server (the archive needs a confirmed `valencia` tag) and checks the links only;
it uses `MCP_BIND` as the address, so with `0.0.0.0` or `::` the server refuses its `Host`
(`421`); run it with `MCP_BIND` set to an address the server accepts.

### Running the MCP server

`deploy/tgi-mcp.service` is a systemd user unit template (it assumes the checkout at
`~/tg-indexer` and the settings in `~/.config/tgi/.env`, mode 0600; the install commands
are in its header). It restarts on failure, every 10 s.

`MCP_BIND` decides who can reach the server. `--http` listens only on a loopback address
or on the address of a local `docker*` or `br-*` interface (read from `ip -j addr`; without
`ip`, only loopback). Anything else, `0.0.0.0` and `::` included, is refused with a one-line
reason unless `--allow-any-bind` is given. For Nano, whose agents run in Docker, the
registration needs:

- the URL `http://<gateway>:<port>/mcp`, where `<gateway>` is the gateway address of the
  Docker network Nano's containers use (`docker network inspect <network>`, `IPAM.Config`).
  Do not assume `172.17.0.1`; user-defined networks get other addresses. Put the same
  address in `MCP_BIND`. When the unit starts before Docker has created the network, the
  bind is refused and the unit retries.
- the token: `MCP_TOKEN` is injected by Nano's credential gateway as
  `Authorization: Bearer …`; it is never put in the agent's own config.

Every bind keeps the MCP SDK's DNS-rebinding check: a request's `Host` must be the bind
address with its port (or `localhost` on a loopback bind, `host.docker.internal` on any
other), otherwise `421`; an `Origin`, when sent, must be `http://` plus one of those, otherwise
`403`. With `--allow-any-bind` on `0.0.0.0` or `::`, every local interface address that
`ip -j addr` lists is accepted; without `ip` that is none (only `localhost` and
`host.docker.internal`), so a client using an address, the Docker gateway's included, gets
`421` until the address is in `MCP_ALLOWED_HOSTS`. Other names a client uses go in `MCP_ALLOWED_HOSTS` (comma-separated, no port).
The rejected header value is not logged.
