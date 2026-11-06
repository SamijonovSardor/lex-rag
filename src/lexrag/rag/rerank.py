"""Second-stage ranking through OpenRouter's /rerank endpoint (Cohere, Qwen, …).

Retrieval scores each chunk on its own (a vector, a word match). A reranker (cross-encoder)
reads the question and the chunk *together* and scores how well the chunk answers it.
Too slow for 20 000 chunks, fine for the 30 candidates retrieval already picked.
"""

import logging
from collections.abc import Sequence

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


class RerankError(Exception):
    pass


class _Retryable(RerankError):
    pass


class OpenRouterReranker:
    """Bounded in time: a reranker that answers late is worth less than none (the spec's
    p95 latency target is 12 s for the whole answer). On timeout the caller keeps the
    retrieval order."""

    def __init__(
        self,
        settings: Settings,
        *,
        timeout_s: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = settings.rerank_model
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "X-Title": "lex-rag",
            },
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_exponential(multiplier=0.5, max=2) + wait_random(0, 1),
        stop=stop_after_attempt(2),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance score of each document, in input order."""
        if not documents:
            return []
        try:
            r = self._client.post(
                "/rerank", json={"model": self.model, "query": query, "documents": list(documents)}
            )
        except httpx.TimeoutException as e:
            raise RerankError(f"{self.model} did not answer in time") from e
        except httpx.TransportError as e:
            raise _Retryable(str(e)) from e
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise RerankError(f"{r.status_code}: {r.text[:500]}")
        out = [0.0] * len(documents)
        for row in r.json()["results"]:
            out[row["index"]] = float(row["relevance_score"])
        return out
