# Manual checks after P1 and P2

What the automated tests cannot verify (SPEC §8, last paragraph). Antony runs these once
on the laptop after each phase (sections 1–4 after P1, 5–11 after P2). Each check notes what passing looks like. Record the date and
result next to each box. Never paste message text, phone numbers, codes or keys into this
file or into an issue.

> **Status (2026-10, SPEC v1.12):** the author has run tgi against real Telegram, so the
> TDLib, login and sync checks of sections 1–4 have been done live even though the boxes
> below are not ticked. Section 12 lists the live checks for the v1.12 additions. Since v1.12
> the default embedding provider is `openai`; section 5 describes the Voyage call, and the
> same check applies to OpenAI with `OPENAI_API_KEY`.

## 1. Docker, compose and the `en_US.utf8` database

Every P1 `make ci` so far ran against a user-space Postgres 16 with locale `C.utf8`, not
the compose `db` service.

- [ ] `docker compose config` shows Postgres published on `127.0.0.1:5432` only, and fails
      with `set POSTGRES_PASSWORD in .env` while `.env` leaves it empty (A10, A12).
- [ ] With `.env` filled in: `make ci` starts the compose `db` (`pgvector/pgvector:pg16`,
      initdb `--locale=en_US.utf8`) and ends with every test passing. This is also the run
      that checks Cyrillic case folding (`ВНЖ` finds `внж`, A6) under `en_US.utf8`.
- [ ] `ss -ltn | grep 5432` lists `127.0.0.1:5432` only.

## 2. GitHub Actions

Actions are disabled on the repo, so `.github/workflows/ci.yml` has only been checked with
`actionlint` and has never run. The pasted local `make ci` output is the CI evidence.

- [ ] Only if Actions are enabled: one run of `ci.yml` on a branch is green.

## 3. Real TDLib (the P1.T3 spike)

The adapter (`tgi/td.py`) is tested only against a fake `libtdjson`; real TDLib has
only been loaded (`scripts/tdlib.sh`), never connected. The live run is Antony's, since the
phone code and 2FA are his. On the laptop, with `.env` filled in (`DATA_DIR` outside the
repo):

```sh
set -a; . ./.env; set +a
scripts/tdlib.sh                    # prints: installed …/libtdjson.so (TDLib 1.8.67)
                                    # (or: scripts/tdlib.sh build, if the wheel does not load)
uv run tgi db migrate                      # fresh database: creates the schema; idempotent
uv run tgi accounts add --phone <phone>      # code, then the 2FA password; "account ready"
ls -ld "$DATA_DIR/accounts" "$DATA_DIR/accounts/<phone>"   # drwx------ both
uv run tgi chats list --account <phone>
uv run tgi td spike <supergroup_id> --account <phone> --pages 2
uv run tgi td spike <channel_id> --account <phone> --pages 2
uv run tgi chats enable <small_chat_id> --account <phone>
uv run tgi sync 2>&1 | tee /tmp/tgi-sync-1.log
uv run tgi sync 2>&1 | tee /tmp/tgi-sync-2.log
grep -c . "$DATA_DIR/accounts/tdlib.log"     # TDLib's own error log, for the A10 check
```

`td spike` prints the TDLib version, the chat kind, then per page `from_message_id`, the
count and the time, `from_inclusive` (whether TDLib returns the `from_message_id` message
again; `tgi sync` copes with both) and each `message_id` / `server_message_id`, then the
`t.me` link of the newest message. It prints ids only; paste its output if useful.
Exit codes: 1 not logged in or a setup error (the message says what to fix), 3 a TDLib
error (type and code only; a flood wait shows `retry after Ns`).

- [ ] `db migrate` runs before any other `tgi` command on a fresh database; without it
      `chats list` fails with `relation "chat" does not exist`.
- [ ] `libtdjson` installs on the host (`scripts/tdlib.sh`, else `build`), and its version is recorded.
- [ ] The login state machine works end to end, including 2FA. The session lands in
      `DATA_DIR/accounts/<phone>` with mode `0700`.
- [ ] `tgi chats list --account <phone>` shows the account's dialogs.
- [ ] `chats list` terminates.
- [ ] `td spike` on a cold start, in a supergroup and in a channel: the first page has
      messages (note how many), every `server_message_id` is a number, and the printed link
      opens the newest message. Record `from_inclusive`.
- [ ] A first `tgi sync` on one small enabled chat ends `ok`, then a second one catches up.
- [ ] Flood waits: note any real `retry after N` values in the sync stats (`flood_waits`).
- [ ] The account shows no sign of restriction after the runs.
- [ ] The output of these commands, the two sync logs and `tdlib.log` hold no message
      text, phone, code, password, `api_hash` or `TDLIB_DB_KEY` (A10: `grep` for the phone
      digits, the `api_hash` and a word you know is in the chats).

## 4. Real links (A11)

Links are built from `server_message_id` and are tested only for their format.

- [ ] Supergroup without a username: a `https://t.me/c/<id>/<n>` link from `tgi search`
      opens that message.
- [ ] Channel or supergroup with a username: the `https://t.me/<username>/<n>` link opens it.
- [ ] Basic group and private chat: `tgi search` prints `(no link)`, as specified.

# P2

The P2 tests use the fake embedder and a loopback server only: no Voyage call, no systemd,
no Docker network and no Nano container. With `.env` loaded (`set -a; . ./.env; set +a`):

## 5. One real Voyage call

(Since v1.12 `EMBED_PROVIDER` defaults to `openai`: for Voyage set `EMBED_PROVIDER=voyage`;
for OpenAI run the same steps with `OPENAI_API_KEY` and read the usage on its dashboard.)
In P2, `EMBED_PROVIDER` defaulted to `voyage`. A `.env` copied from the P1 `.env.example` still
says `EMBED_PROVIDER=fake`; set it to `voyage` (or remove the line). `VOYAGE_API_KEY` is the
real key, or any non-empty placeholder when `api.voyageai.com` goes through the OneCLI
proxy. Left empty, as in `.env.example`, `tgi embed` stops with `VOYAGE_API_KEY is not set`
(exit 1).

```sh
uv run tgi embed --max-windows 5
# exit 3 "model_mismatch" means the archive holds P1's fake vectors: drop them and embed 5
uv run tgi embed --reindex --max-windows 5
```

- [ ] It prints `embedded=5 embed_failed=0 tokens=<n>` (after `--reindex`, `dropped=<m>`
      first) with `n > 0` (Voyage's `usage.total_tokens`), and every other window stays
      `stale`.
- [ ] The Voyage dashboard shows matching usage. The dashboard is the source for cost;
      `tgi` prints tokens, never a price.
- [ ] The output and Voyage's error text, if any, hold no key.

## 6. The first full embed run

HNSW inserts cost about 5.6 ms each on P1's measurements, so the first ~200k windows need
over 20 minutes of index time on top of Voyage's rate limits. `tgi sync` gets there over
several runs within its budget; by hand it is one sitting (it takes the sync lock, so a
scheduled sync that starts meanwhile exits 2 and retries next time):

```sh
time uv run tgi embed
```

- [ ] It ends with `embed_failed=0`, or reruns of `uv run tgi embed` bring it to 0 (a `429`
      above 30 s leaves its batch `stale` for the next run). Record the time and `tokens=`.
- [ ] Afterwards no window is `stale`:
      `psql "$DATABASE_URL" -c 'SELECT count(*) FROM "window" WHERE stale'` prints 0.

## 7. Semantic quality on the reference query

With the archive embedded, a confirmed `valencia` tag on the Valencia chats
(`uv run tgi tags confirm <chat_id> valencia`) and the server running (section 8):

```sh
uv run tgi search "продление ВНЖ" "renovación TIE" "residence permit renewal" --tag valencia --since <today-30d>
uv run tgi selftest       # the reference query through MCP; MCP_BIND as the server's
```

- [ ] The top hits are about residence permit renewal in Valencia, in more than one
      language, and the links open those messages.
- [ ] `tgi selftest` prints `selftest ok: tags=… hits=… context=…` and exits 0. A failure
      prints its reason (exit 1) or the cause (exit 3: a refused token, an HTTP status, no
      connection).
- [ ] Note any obvious miss (a known relevant message that does not come back) without its
      text: chat id and message link only.

## 8. The systemd unit in a real user session

The unit is only checked as a file. Install it as its header says, with the settings in
`~/.config/tgi/.env` (mode 0600):

```sh
install -Dm644 deploy/tgi-mcp.service ~/.config/systemd/user/tgi-mcp.service
systemctl --user daemon-reload && systemctl --user enable --now tgi-mcp
systemctl --user status tgi-mcp
journalctl --user -u tgi-mcp -n 50
```

- [ ] The unit is `active (running)` and the journal shows `tgi mcp: serving http`.
- [ ] `systemctl --user restart tgi-mcp` brings it back; after a reboot and login it starts
      by itself. Whether it should also run while logged out (`loginctl enable-linger`) is
      Antony's choice; record it.
- [ ] The journal holds no token, query or message text.

## 9. Bind on the real Docker bridge

The bind guard's interface check is tested with a faked `ip -j addr`; the real bind was
only tried on loopback. With `ip` installed and the gateway of the Docker network Nano's
containers use (`docker network inspect <network>`, `IPAM.Config`; not `172.17.0.1` by
assumption) in `MCP_BIND=<gateway>:8765`:

- [ ] `systemctl --user restart tgi-mcp` starts, and `ss -ltn | grep 8765` lists
      `<gateway>:8765` only.
- [ ] `MCP_BIND=0.0.0.0:8765 uv run tgi mcp --http` is refused with a one-line reason.
- [ ] With the unit started before Docker has created the network (stop Docker, restart
      the unit, start Docker), the bind is refused and the unit succeeds on a later retry.

## 10. Reachability from the Nano container (at registration)

Registration is Antony's step, outside this repo. From inside a Nano agent container:

- [ ] `curl -s -o /dev/null -w '%{http_code}\n' -X POST http://<gateway>:8765/mcp` without a
      token prints `401`.
- [ ] Through Nano's credential gateway (the token injected as `Authorization: Bearer …`),
      the agent's `list_tags` call answers with the archive's tags.
- [ ] The token is not in the agent's own config, and the server's journal shows no token.

## 11. Links (A11)

- [ ] If section 4 is still open, run it now against the embedded archive.

## 12. The v1.12 additions, live

With a real account and a few enabled chats (a short `--years` is enough):

- [ ] `tgi embed --max-windows 5` with the default `EMBED_PROVIDER=openai` embeds five
      windows and prints `tokens=`; the OpenAI usage page shows the matching spend.
- [ ] A first sync of a chat with a long history ends `time_limited` with the backfill
      unfinished, yet the run stats show a sweep and `embedded` > 0 (`BACKFILL_RESERVE`);
      `tgi status` shows `history_complete: false` for it and a falling
      `waiting_for_embedding` over the next runs.
- [ ] `deploy/tgi-sync.service` and `tgi-sync.timer` installed as user units:
      `systemctl --user list-timers` shows the next run; a run started while a sync holds the
      lock ends as success (exit 2); `journalctl --user -u tgi-sync` shows the printed stats.
- [ ] `tgi chats list` and `tgi chats enable` exit 2 while a sync runs; `tgi chats add <link>`
      joins a public group and enables it; a join-request link exits 1, nothing enabled.
- [ ] `tgi recent --hours 24` lists real conversations newest first with working links;
      `--cursor` from the printed `more:` line continues without repeats; `tgi search
      … --compact` shows `matched_by` and cut texts.
- [ ] `tgi status` on the real archive: `sync_lock_held` true during a sync and false after;
      `last_sync_attempt_at` per chat; `schedule.next_run_at` matches the timer when
      `SYNC_EVERY_MINUTES` is set.
- [ ] MCP: `tgi mcp --http` with `MCP_READ_ONLY=1` lists seven tools and never opens TDLib
      (no TDLib log lines); without it, nine. In the MCP client, the read tools carry
      `readOnlyHint: true` and `add_chat` does not (a client marking the server untrusted asks
      before `add_chat` only).
- [ ] An assistant over MCP answers a "what's new in <tag>" request with `recent` (not an
      empty `search`) and states the period from the answer's `now`.

## Outside P2

- The Nano agent's answer quality on the reference query, with `docs/AGENT-NOTES.md` in its
  prompt, is judged at registration.
