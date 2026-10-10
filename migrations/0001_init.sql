-- SPEC §4. "window" is a reserved word in Postgres and is always quoted.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE chat (
    chat_id           bigint PRIMARY KEY,
    account_phone     text NOT NULL,
    title             text NOT NULL,
    kind              text NOT NULL
                      CHECK (kind IN ('supergroup', 'channel', 'basic_group', 'private')),
    username          text,
    enabled           boolean NOT NULL DEFAULT false,
    added_at          timestamptz NOT NULL DEFAULT now(),
    backfill_done     boolean NOT NULL DEFAULT false,
    backfill_years    integer CHECK (backfill_years > 0),
    oldest_message_id bigint,
    newest_message_id bigint,
    last_sync_at      timestamptz
);

CREATE TABLE chat_tag (
    chat_id     bigint NOT NULL REFERENCES chat ON DELETE CASCADE,
    tag         text NOT NULL,
    status      text NOT NULL CHECK (status IN ('proposed', 'confirmed')),
    proposed_by text,
    PRIMARY KEY (chat_id, tag)
);

CREATE TABLE message (
    chat_id             bigint NOT NULL REFERENCES chat ON DELETE CASCADE,
    message_id          bigint NOT NULL,
    server_message_id   bigint NOT NULL,
    sender_id           bigint,
    sender_name         text,
    date                timestamptz NOT NULL,
    edit_date           timestamptz,
    reply_to_message_id bigint,
    text                text NOT NULL,
    has_media           boolean NOT NULL DEFAULT false,
    deleted_at          timestamptz,
    tsv                 tsvector GENERATED ALWAYS AS (
                            to_tsvector('russian', text)
                            || to_tsvector('spanish', text)
                            || to_tsvector('english', text)
                        ) STORED,
    PRIMARY KEY (chat_id, message_id),
    -- TDLib server message ids: message_id = server_id << 20 (§4 rules).
    CHECK (message_id % 1048576 = 0 AND server_message_id = message_id >> 20)
);

CREATE INDEX message_tsv_idx ON message USING gin (tsv);
CREATE INDEX message_text_trgm_idx ON message USING gin (text gin_trgm_ops);
CREATE INDEX message_chat_date_idx ON message (chat_id, date);
CREATE INDEX message_chat_reply_idx ON message (chat_id, reply_to_message_id);

CREATE TABLE "window" (
    window_id        bigserial PRIMARY KEY,
    chat_id          bigint NOT NULL REFERENCES chat ON DELETE CASCADE,
    first_message_id bigint NOT NULL,
    start_date       timestamptz NOT NULL,
    end_date         timestamptz NOT NULL,
    message_ids      bigint[] NOT NULL,
    text             text NOT NULL,
    text_hash        bytea,
    stale            boolean NOT NULL DEFAULT true,
    UNIQUE (chat_id, first_message_id)
);

CREATE INDEX window_chat_start_idx ON "window" (chat_id, start_date);
CREATE INDEX window_message_ids_idx ON "window" USING gin (message_ids);

CREATE TABLE window_embedding (
    window_id   bigint PRIMARY KEY REFERENCES "window" ON DELETE CASCADE,
    model       text NOT NULL,
    dim         integer NOT NULL,
    embedding   vector(1024) NOT NULL,
    embedded_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX window_embedding_hnsw_idx ON window_embedding
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 128);

CREATE TABLE sync_run (
    run_id      bigserial PRIMARY KEY,
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    status      text NOT NULL,
    stats       jsonb NOT NULL DEFAULT '{}',
    error       text
);
