"""One question in, one spec-shaped result out: language → retrieval → answer.

Errors stay inside the item (status "error" + code), so one failing question never sinks
the other questions of a batch (the teacher's spec, section 5).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import psycopg

from lexrag.config import Settings
from lexrag.index.embed import EmbeddingError, OpenRouterEmbedder
from lexrag.rag.answer import Answer, generate
from lexrag.rag.llm import LLMError, OpenRouterLLM
from lexrag.rag.rerank import OpenRouterReranker, RerankError
from lexrag.rag.retrieve import Context, Retrieval, Retriever
from lexrag.rag.rewrite import to_uzbek_query
from lexrag.text.lang import Language, detect_language

log = logging.getLogger(__name__)

LANGUAGES: tuple[Language, ...] = ("uz-Latn", "uz-Cyrl", "ru")


@dataclass
class ItemResult:
    id: str
    status: str  # ok | no_answer | error | timeout
    language: Language | None = None
    answer: Answer | None = None
    retrieval: Retrieval | None = None
    error: dict[str, Any] | None = None
    timing_ms: dict[str, int] = field(default_factory=dict)

    @property
    def contexts(self) -> list[Context]:
        return self.retrieval.contexts if self.retrieval else []


class RagPipeline:
    def __init__(self, conn: psycopg.Connection, settings: Settings) -> None:
        self.settings = settings
        self.embedder = OpenRouterEmbedder(settings)
        self.reranker = OpenRouterReranker(settings) if settings.rerank_model else None
        self.llm = OpenRouterLLM(settings)
        self.retriever = Retriever(
            conn, self.embedder, self.reranker,
            candidates=settings.retrieval_candidates,
            rerank_candidates=settings.rerank_candidates,
        )  # fmt: skip

    def close(self) -> None:
        self.embedder.close()
        self.llm.close()
        if self.reranker:
            self.reranker.close()

    def retrieve(
        self, question: str, lang: Language, top_k: int, seed: int | None = 42
    ) -> tuple[Retrieval, tuple[int, int]]:
        """Search; a Russian question is first rewritten into an Uzbek query.
        Returns the retrieval and the (input, output) tokens the rewrite used."""
        if lang != "ru":
            return self.retriever.search(question, lang, top_k), (0, 0)
        t = time.perf_counter()
        try:
            query, tin, tout = to_uzbek_query(
                self.llm, question, seed, self.settings.rewrite_reasoning_effort
            )
        except LLMError:  # degrade to cross-lingual search rather than fail the question
            log.warning("rewrite failed; searching with the Russian question", exc_info=True)
            query, tin, tout = None, 0, 0
        rewrite_ms = round((time.perf_counter() - t) * 1000)
        r = self.retriever.search(question, lang, top_k, query=query)
        r.timing_ms = {"rewrite_ms": rewrite_ms, **r.timing_ms}
        r.timing_ms["retrieval_ms"] += rewrite_ms
        return r, (tin, tout)

    def run(
        self,
        item_id: str,
        question: str,
        *,
        language: str | None = None,
        answer_language: str = "auto",
        top_k: int = 5,
        temperature: float = 0.0,
        seed: int | None = 42,
    ) -> ItemResult:
        t0 = time.perf_counter()
        lang: Language = language if language in LANGUAGES else detect_language(question)  # type: ignore[assignment]
        out_lang: Language = answer_language if answer_language in LANGUAGES else lang  # type: ignore[assignment]
        result = ItemResult(id=item_id, status="ok", language=lang)
        try:
            result.retrieval, (tin, tout) = self.retrieve(question, lang, top_k, seed)
            t = time.perf_counter()
            result.answer = generate(
                self.llm, question, result.retrieval.contexts, out_lang,
                temperature=temperature, seed=seed,
                reasoning_effort=self.settings.llm_reasoning_effort,
            )  # fmt: skip
            result.answer.input_tokens += tin
            result.answer.output_tokens += tout
            result.timing_ms = {
                **result.retrieval.timing_ms,
                "generation_ms": round((time.perf_counter() - t) * 1000),
            }
            result.status = "no_answer" if result.answer.no_answer else "ok"
        except (EmbeddingError, RerankError, psycopg.Error) as e:
            log.exception("retrieval failed for %s", item_id)
            result.status = "error"
            result.error = {"code": "RETRIEVAL_FAILED", "message": str(e)[:300], "retryable": True}
        except LLMError as e:
            log.exception("generation failed for %s", item_id)
            result.status = "timeout" if e.code == "LLM_TIMEOUT" else "error"
            result.error = {"code": e.code, "message": str(e)[:300], "retryable": True}
        result.timing_ms["total_ms"] = round((time.perf_counter() - t0) * 1000)
        return result
