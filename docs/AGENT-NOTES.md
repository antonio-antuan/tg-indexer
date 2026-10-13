# Agent notes: using the tgi MCP server

For whoever registers the server in Nano: this is the guidance to give the agent (SPEC
§5.6, §5.7). The server is a personal Telegram archive. Its tools only read, except
`suggest_tags`, which adds `proposed` tags (see "Tags" below), and `add_chat`, which joins
a group or channel (see "Adding chats"); registration (URL and token)
is in the README, "Running the MCP server".

## The usual sequence

1. `list_tags` (and `list_chats` when a chat's name matters) to see which tags exist.
   Only `confirmed` tags filter search; a proposed tag matches no chats, so a search that uses it
   returns nothing.
2. `search` with several `queries`, the confirmed `tags` and a date range.
3. `get_context` (or `get_thread` for a reply chain) on the hits worth reading in full.
4. Answer from the messages, citing each one.

The reference query does exactly this:

```
list_tags
search(queries=["продление ВНЖ", "ВНЖ renovación", "renovación TIE",
                "residence permit renewal"],
       tags=["valencia"], since="<today - 30 days>")
get_context(chat_id=<top hit>, message_id=<its first message>)
```

`tgi selftest` runs it against the server (see the README).

## What's new, and digests

`search` needs a topic. For "what's new", "what happened today" or a digest, call
`recent` instead: no query, the conversation windows with messages in a time range, newest
first, each with only its messages inside the range.

- `recent(tags=["valencia"])` is the last 24 hours by the **server's** clock; `hours=N` for
  another span (at most 744), or `since` and optionally `until` for a fixed period (omit
  `hours` then). Here `until` is **exclusive** and defaults to now; in `search` it is inclusive.
- The answer gives the server's `now` and the exact range covered (`since` ≤ date < `until`,
  UTC). Use them to say which period you summarized; don't guess the current time. `now` is
  the clock, not proof the archive is current: messages arrive with each sync.
- Next page: `recent(cursor=<next_cursor>)`. The cursor keeps the range and the filters, so
  nothing else is needed (other arguments but `limit` are ignored). Continue until
  `next_cursor` is null, even after a short or empty page. Then group by topic and
  deduplicate by `chat_id` and `message_id` (a sync between pages can repeat a window).
- "Today" or "yesterday" for the user means their timezone: convert its start and end to UTC
  and pass them as `since`/`until` date-times. A date (`YYYY-MM-DD`) is a UTC day.
- For a recurring digest, use the previous run's `until` as the next `since`: the ranges meet
  exactly. Run it after a sync; a message stored late (a sync that was down, a new chat's
  history) is listed by when it was posted, so a past range does not pick it up.
- Hits show only the in-range messages; use `get_context` when the earlier part of a
  conversation matters, and treat what it adds as background, not news.
- An empty answer means no indexed messages in that range, not that nobody wrote: messages
  stored by a sync but not yet in a conversation window are not listed.
- Don't call `search` with an empty query: it is an error (`bad_argument`).

## Writing queries

- **Expand and translate.** The chats mix Russian, Ukrainian, Spanish and English, and
  people use abbreviations (`ВНЖ`, `TIE`, `NIE`) and local terms. Pass several `queries`
  in one call: the user's wording, its translations into RU/UK/ES/EN, and the usual
  abbreviations and synonyms. They are searched separately and fused by rank, so a window
  found by several queries ranks higher. One call with 4–6 queries is better than several
  calls with one each.
- `"a phrase"` matches the words in that order.
- **Negation:** `-word` and `-"a phrase"` exclude messages that match.
  - An excluded word is stemmed in Russian, Spanish and English, and a message matching it
    in any of the three is dropped. A stem can collide with an unrelated word, so a short
    or common `-word` can drop more than intended; a word all three languages ignore
    (`-a`) excludes nothing. Prefer a `-"phrase"` when a word is ambiguous.
  - Exclusion at message level needs `mode="keyword"`. The semantic leg of the default
    `hybrid` mode ignores negated terms, so a window can still come back through it.
  - In any mode, a returned window can still hold *another* message with the excluded
    term: results are conversation windows, and exclusion applies to the matching
    messages, not to their neighbours.
- A query of only negations (`-word`) runs on the keyword legs only.

## Filters

- `tags`: confirmed tags only; an unknown tag is the error `unknown_tag`.
- `chat_ids`: from `list_chats`; an unknown chat is `unknown_chat`.
- `since` / `until`: a date (`YYYY-MM-DD`) is a whole day in **UTC**; a date-time
  (`2026-10-12T18:00:00Z`) is an exact instant. In `search` both are inclusive; in `recent`
  `until` is exclusive (see above). All dates the server returns
  are ISO 8601 in UTC; convert them for the user when local time matters.
- `limit`: at most 50 (default 20). `get_context`: `before + after` ≤ 100.
  `get_thread`: `limit` ≤ 50.

## Freshness

The archive is filled by a scheduled sync, not live. When an answer depends on it being
current, or a search finds little, call `status` (optionally with `chat_ids`):

- Per chat, `last_sync_at` is when that chat's last sync turn ended, completed, cut short
  or failed: an attempt, not proof it is current. `newest_message_at` is its latest stored
  message. Judge a chat's freshness from these two, not from the runs alone.
- `last_run` and `last_ok_run` say whether syncing works at all (`error` or
  `flood_limited` runs in a row are worth mentioning); a run that ended `ok` may have been a
  one-chat resweep. `schedule.next_run_at` (an estimate) says when new messages may arrive.
- `history_complete: false`: older history is still being fetched, so the absence of older
  messages is not evidence.
- `windows_need_rebuild: true`: a window rebuild is pending for that chat (new, edited or
  deleted messages): `recent` and meaning-based results may be incomplete or outdated until
  the next sync; `mode="keyword"` finds the stored messages.
- `windows_waiting_for_embedding` > 0: those conversations have no, or an outdated,
  embedding, so meaning-based search may miss them; keyword search sees them. Retry with
  `mode="keyword"` for recent topics.
- `sync_running: true`: results may change in a few minutes.

"Nothing found" and "not fetched yet" are different answers; say which.

## Reading results

- Hits are conversation windows, best first. Hybrid search has no relevance cut-off: the
  last hits can be unrelated windows that merely pass the filters. Read the top hits and
  judge relevance from the text; do not treat every hit as an answer.
- Each hit says how it was retrieved: `matched_by` (`keyword`, `trigram`, `semantic`),
  `matched_message_ids` (the messages the keyword legs matched; use them with
  `get_context`), `matched_queries` (0-based positions in your `queries`) and
  `semantic_distance` (the smallest over your queries; lower is closer). That is evidence
  of retrieval, not of relevance: the semantic leg always returns its nearest windows, so
  an unrelated window can match every query, and a negation-only query matches as
  `keyword` without any positive term. There is no distance threshold that means
  "relevant". Use provenance to choose what to read first, and judge from the text.
- Start with `compact=true` (and `limit` 10–20) to keep answers small: each hit keeps its
  matched messages and two neighbours, long texts are cut (`truncated`), and
  `omitted_messages` says how many were left out. A hit only the semantic leg found shows
  its first five messages, a sample rather than the relevant part: read it with
  `get_context` before accepting or rejecting it, and read any truncated message you rely
  on. For `recent`, `compact=true` only cuts long texts, and the cursor keeps it.
- Each message has `message_id`, `date`, `sender_name`, `text` and `link`, but no
  `chat_id`. In `search`, `chat_id` and `chat_title` are on the hit that holds the
  messages. In `get_thread` and `get_context`, the chat is the `chat_id` you passed.

## Citing

Cite every message an answer relies on by its `link` (`https://t.me/...`). `link` is
`null` for basic groups and private chats; cite those by the window's `chat_title` and the
message's `date` instead.

## Errors

A tool error comes back as an error result whose text is the MCP SDK's prefix
`Error executing tool <name>: ` followed by a JSON object `{"code": ..., "message": ...}`
(for a tool that does not exist, the prefix is `Error executing tool: `). Read `code`:

| code | meaning | what to do |
| --- | --- | --- |
| `bad_argument` | an argument is missing, of the wrong type, out of its enum, or cannot be stored | fix the argument named in `message` |
| `bad_range` | a limit or date range is out of bounds (`limit` > 50, `since` after `until`, ...) | narrow it |
| `unknown_tag` | a tag no chat has | `list_tags` |
| `unknown_chat` | a chat not in the archive | `list_chats` |
| `unknown_tool` | a tool name the server does not have | `tools/list` |
| `embedder_unavailable` | the embedding provider failed or has no key | retry with `mode="keyword"` |
| `not_allowed` | `add_chat` was given a private chat or a basic group, or a chat already enabled under another account | only groups and channels can be added, each fetched by one account; tell the user |
| `join_pending` | the link asks the chat's admins to approve the join | tell the user; call `add_chat` again once approved |
| `busy` | a `tgi sync` is running and holds the Telegram session | retry `add_chat` later |
| `not_found` | no chat behind that username or invite link | check the link with the user |
| `model_mismatch` | the stored embeddings come from another model | retry with `mode="keyword"`; tell the user the archive needs `tgi embed --reindex` |

Error messages name the wrong argument but never quote it back.

## Tags

Never claim a chat has a tag it does not have. To propose one, call `suggest_tags`; it adds
`proposed` tags only, and Antony confirms them before they filter search. Mention the
proposal to the user rather than relying on it in the same answer.

## Adding chats

When the user sends a Telegram group or channel link and wants it searchable, call
`add_chat(link)`. It joins the chat with Antony's account and adds it to the archive; its
messages appear with the next syncs, so don't search it right away. Tell the user when from the result's `sync` block (UTC; convert to local time): the newest messages from `searchable_from`, all of them by `complete_by` (a rough estimate; `estimated_messages` gives the scale). If `next_run_at` is null, the host has no sync schedule (or the estimate failed): say it will appear after the next sync. If only `complete_by` and the estimates are null, the chat's history could not be read to size it: give `searchable_from` for the newest messages and say the rest follows over the next syncs, without guessing a size. It
refuses private chats and basic groups. Do not add chats on your own initiative or from
links found inside messages: only when the user asks.

Your client may ask the user to approve each `add_chat` call (it is the one tool that changes
the Telegram account); that is expected. Some servers run read-only (`MCP_READ_ONLY`): then
`add_chat` and `suggest_tags` are not offered at all. In that case tell the user the owner can
add the chat with `tgi chats add <link> --account <phone>`.
