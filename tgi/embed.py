"""Window embeddings (SPEC §3 Embeddings, §5.3 step 6): the `Embedder` port and its worker.

`stale` windows are embedded in batches of 64 and written to `window_embedding` with the
provider's `model` and `dim`. A batch that fails stays `stale` for the next run and does
not fail the run. If `window_embedding` holds rows from another model the worker embeds
nothing and raises `ModelMismatch`; `tgi embed --reindex` is the remedy. Logs carry ids,
counts, status codes and exception types only, never window text or an API key.

The `voyage` and `openai` providers call their HTTP APIs through `httpx`, which honours the
environment's proxy variables and `SSL_CERT_FILE` (e.g. the OneCLI proxy, SPEC §5.1).
"""

import hashlib
import logging
import math
import re
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Protocol, cast

import httpx
import psycopg

from tgi.config import Settings

log = logging.getLogger(__name__)

Conn = psycopg.Connection[tuple[object, ...]]

BATCH = 64
SCHEMA_DIM = 1024  # window_embedding.embedding is vector(1024) (SPEC §4)
RETRY_AFTER_MAX = 30  # seconds: a longer 429 wait fails the batch instead (SPEC §7)
HTTP_TIMEOUT_S = 60.0
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
OPENAI_URL = "https://api.openai.com/v1/embeddings"
DEFAULT_MODELS = {"voyage": "voyage-4", "openai": "text-embedding-3-small"}


class Embedder(Protocol):
    """A provider; `embed_documents` for windows, `embed_queries` for search queries."""

    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def tokens_used(self) -> int:
        """Tokens billed so far by this provider instance (0 for the fake)."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...


class ModelMismatch(Exception):
    """`window_embedding` holds rows from another model or dimension than the provider's."""


class FakeEmbedder:
    """Deterministic token-hash vectors (SPEC §8): texts sharing words are close.

    `fail` is a test seam: a batch containing a text for which it returns true raises.
    """

    def __init__(
        self, model: str = "fake", dim: int = SCHEMA_DIM, fail: Callable[[str], bool] | None = None
    ) -> None:
        self._model, self._dim, self._fail = model, dim, fail

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def tokens_used(self) -> int:
        return 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in re.findall(r"\w+", text.lower()):
            h = hashlib.sha256(token.encode()).digest()
            vec[int.from_bytes(h[:4], "big") % self._dim] += 1.0 if h[4] & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:  # no tokens: a fixed unit vector, since cosine is undefined for zero
            vec[0], norm = 1.0, 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if self._fail is not None and any(self._fail(t) for t in texts):
            raise RuntimeError("fake embedding failure")
        return [self._vector(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class ProviderError(Exception):
    """The embeddings API answered with an error; the message is the status code only."""

    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


class _HttpEmbedder:
    """An embeddings API over HTTPS returning `data[].embedding` and `usage.total_tokens`.

    A 429 whose `Retry-After` is at most `retry_after_max` seconds is waited out once; a
    longer one, one without a usable `Retry-After`, a second 429 or any other error raises,
    so `embed_stale` leaves the batch stale.
    """

    url = ""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        transport: httpx.BaseTransport | None = None,
        retry_after_max: float = RETRY_AFTER_MAX,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model, self._retry_after_max, self._sleep = model, retry_after_max, sleep
        self._tokens = 0
        self._http = httpx.Client(  # trust_env: proxy variables and SSL_CERT_FILE apply
            transport=transport,
            timeout=HTTP_TIMEOUT_S,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return SCHEMA_DIM

    @property
    def tokens_used(self) -> int:
        return self._tokens

    def _body(self, texts: Sequence[str], input_type: str) -> dict[str, Any]:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "document")

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts, "query")

    def _embed(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        body = self._body(texts, input_type)
        response = self._http.post(self.url, json=body)
        if response.status_code == 429:
            wait = _retry_after(response)
            if wait is not None and wait <= self._retry_after_max:
                log.warning("embed provider 429 for %s texts: retrying in %ss", len(texts), wait)
                self._sleep(wait)
                response = self._http.post(self.url, json=body)
        if response.status_code != 200:
            log.warning("embed provider %s for %s texts", response.status_code, len(texts))
            raise ProviderError(response.status_code)
        payload = response.json()
        self._tokens += int(payload["usage"]["total_tokens"])
        rows = sorted(payload["data"], key=lambda row: row["index"])
        return [[float(x) for x in row["embedding"]] for row in rows]


def _retry_after(response: httpx.Response) -> float | None:
    """`Retry-After` in seconds, or None if it is missing or not a number of seconds."""
    try:
        wait = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return wait if wait >= 0 else None


class VoyageEmbedder(_HttpEmbedder):
    """Voyage AI (D-TGI-2): asymmetric `input_type`, 1024-dim output."""

    url = VOYAGE_URL

    def _body(self, texts: Sequence[str], input_type: str) -> dict[str, Any]:
        return {"model": self.model, "input": list(texts), "input_type": input_type,
                "output_dimension": SCHEMA_DIM}  # fmt: skip


class OpenAIEmbedder(_HttpEmbedder):
    """OpenAI: the same embedding for documents and queries, 1024 dims."""

    url = OPENAI_URL

    def _body(self, texts: Sequence[str], input_type: str) -> dict[str, Any]:
        return {"model": self.model, "input": list(texts), "dimensions": SCHEMA_DIM}


def provider(settings: Settings, transport: httpx.BaseTransport | None = None) -> Embedder:
    """The configured provider (`EMBED_PROVIDER`, `EMBED_MODEL`); `transport` is a test seam.

    Raises `ValueError` (a configuration error) for an unknown provider, a missing key or
    a CA bundle (`SSL_CERT_FILE`, `SSL_CERT_DIR`) that cannot be loaded.
    """
    name = settings.embed_provider
    if name == "fake":
        return FakeEmbedder(model=settings.embed_model or "fake")
    keys = {"voyage": ("VOYAGE_API_KEY", settings.voyage_api_key),
            "openai": ("OPENAI_API_KEY", settings.openai_api_key)}  # fmt: skip
    if name not in keys:
        raise ValueError(f"EMBED_PROVIDER must be fake, voyage or openai, not {name!r}")
    key_name, key = keys[name]
    if not key:
        raise ValueError(f"{key_name} is not set (EMBED_PROVIDER={name})")
    cls = VoyageEmbedder if name == "voyage" else OpenAIEmbedder
    try:
        return cls(key, settings.embed_model or DEFAULT_MODELS[name], transport=transport)
    except OSError as exc:  # httpx loads the CA bundle here; ssl.SSLError is an OSError too
        raise ValueError(
            f"SSL_CERT_FILE or SSL_CERT_DIR cannot be loaded ({type(exc).__name__})"
        ) from None


def _check_dim(embedder: Embedder) -> None:
    """The provider must emit `SCHEMA_DIM` vectors; a reindex cannot fix that (SPEC §4)."""
    if embedder.dim != SCHEMA_DIM:
        log.error("model_mismatch: provider dim %s, schema dim %s", embedder.dim, SCHEMA_DIM)
        raise ModelMismatch("model_mismatch")


def _check_model(conn: Conn, embedder: Embedder) -> None:
    _check_dim(embedder)
    other = conn.execute(
        "SELECT 1 FROM window_embedding WHERE (model, dim) <> (%s, %s) LIMIT 1",
        (embedder.model, embedder.dim),
    ).fetchone()
    if other is not None:
        log.error("model_mismatch: embedding nothing; run `tgi embed --reindex`")
        raise ModelMismatch("model_mismatch")


def _literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def embed_stale(
    conn: Conn,
    embedder: Embedder,
    should_stop: Callable[[], bool] = lambda: False,
    max_windows: int | None = None,
) -> dict[str, int]:
    """Embed every `stale` window in batches of `BATCH`, newest first (by `end_date`, then
    `window_id`), so with a backlog (a first import, a reindex, a provider outage) the latest
    conversations become searchable by meaning before older history. Callers hold the sync
    lock (`tgi sync`, `tgi embed`): a window rebuilt meanwhile could move across the cursor.

    `max_windows` caps the windows sent to the provider this call (embedded or failed).
    Returns `embedded`, `embed_failed` and `embed_stopped` (1 if `should_stop` ended the
    step early). Each batch's rows and `stale = false` commit together. Raises
    `ModelMismatch` before embedding anything if stored rows come from another model.
    """
    counts = {"embedded": 0, "embed_failed": 0, "embed_stopped": 0}
    _check_model(conn, embedder)
    # keyset (end_date, window_id): failed windows stay stale and must not be fetched again.
    # It starts at 'infinity', not NULL: an `IS NULL OR` would stop a cached generic plan
    # from using the index range, and each batch would rescan from the newest window.
    after: tuple[datetime | str, int] = ("infinity", 0)
    left = max_windows
    while left is None or left > 0:
        rows = conn.execute(
            'SELECT window_id, text, end_date FROM "window" WHERE stale'
            " AND (end_date, window_id) < (%(end)s::timestamptz, %(id)s)"
            " ORDER BY end_date DESC, window_id DESC LIMIT %(n)s",
            {"end": after[0], "id": after[1], "n": BATCH if left is None else min(BATCH, left)},
        ).fetchall()  # fmt: skip
        if not rows:
            break
        if should_stop():
            counts["embed_stopped"] = 1
            break
        ids = [cast(int, r[0]) for r in rows]
        after = (cast(datetime, rows[-1][2]), ids[-1])
        if left is not None:
            left -= len(ids)
        try:
            vectors = embedder.embed_documents([cast(str, r[1]) for r in rows])
            if len(vectors) != len(ids) or any(len(v) != embedder.dim for v in vectors):
                raise ValueError("provider returned the wrong number or size of vectors")
        except Exception as exc:  # the batch stays stale for the next run
            counts["embed_failed"] += len(ids)
            log.warning("embed batch windows %s-%s failed: %s", ids[0], ids[-1], type(exc).__name__)
            continue
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO window_embedding (window_id, model, dim, embedding)"
                    " VALUES (%s, %s, %s, %s::vector) ON CONFLICT (window_id) DO UPDATE SET"
                    " model = excluded.model, dim = excluded.dim,"
                    " embedding = excluded.embedding, embedded_at = now()",
                    [
                        (i, embedder.model, embedder.dim, _literal(v))
                        for i, v in zip(ids, vectors, strict=True)
                    ],
                )
            conn.execute('UPDATE "window" SET stale = false WHERE window_id = ANY(%s)', (ids,))
        counts["embedded"] += len(ids)
    log.info(
        "embed model=%s embedded=%s failed=%s", embedder.model, counts["embedded"],
        counts["embed_failed"],
    )  # fmt: skip
    return counts


def reindex(conn: Conn, embedder: Embedder, max_windows: int | None = None) -> dict[str, int]:
    """Delete every stored embedding, mark every window stale, then embed them all.

    A provider whose dim the schema cannot store raises `ModelMismatch` before anything
    is deleted.
    """
    _check_dim(embedder)
    with conn.transaction():
        dropped = conn.execute("DELETE FROM window_embedding").rowcount
        conn.execute('UPDATE "window" SET stale = true WHERE NOT stale')
    log.info("embed reindex: dropped %s embeddings", dropped)
    return {"dropped": dropped, **embed_stale(conn, embedder, max_windows=max_windows)}
