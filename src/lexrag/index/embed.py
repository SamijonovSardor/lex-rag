"""Text → vectors through OpenRouter's OpenAI-compatible /embeddings endpoint.

`baai/bge-m3`: multilingual (Uzbek, Russian, …), 1 024 dimensions, 8 192-token input,
$0.01 per million tokens on 2026-11-06. The same model embeds the user's question at query
time, so a Russian question lands near the Uzbek article that answers it.
"""

import logging
from collections.abc import Sequence
from typing import Protocol

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from lexrag.config import Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model: str
    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingError(Exception):
    pass


class _Retryable(EmbeddingError):
    pass


class OpenRouterEmbedder:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        if not settings.openrouter_api_key:
            raise EmbeddingError("OPENROUTER_API_KEY is not set (put it in .env)")
        self.model = settings.embedding_model
        self.provider_sort = settings.provider_sort
        self.dim = settings.embedding_dim
        self.batch_size = settings.embed_batch_size
        self.tokens = 0  # prompt tokens billed so far, for the cost line in the report
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "X-Title": "lex-rag",
            },
            timeout=120.0,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._embed_batch(list(texts[start : start + self.batch_size])))
        return out

    @retry(
        retry=retry_if_exception_type((_Retryable, httpx.TransportError)),
        wait=wait_exponential(multiplier=2, max=60) + wait_random(0, 1),
        stop=stop_after_attempt(6),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body: dict = {"model": self.model, "input": batch}
        if self.provider_sort:
            body["provider"] = {"sort": self.provider_sort}
        r = self._client.post("/embeddings", json=body)
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise EmbeddingError(f"{r.status_code}: {r.text[:500]}")
        body = r.json()
        if "data" not in body:  # OpenRouter reports some upstream failures inside a 200
            raise _Retryable(f"no data in response: {str(body)[:200]}")
        rows = sorted(body["data"], key=lambda d: d["index"])
        vectors = [row["embedding"] for row in rows]
        if len(vectors) != len(batch):
            raise EmbeddingError(f"asked for {len(batch)} vectors, got {len(vectors)}")
        if any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"expected {self.dim}-dim vectors from {self.model}")
        self.tokens += body.get("usage", {}).get("prompt_tokens", 0)
        return vectors
