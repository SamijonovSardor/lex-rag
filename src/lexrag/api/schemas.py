"""Request and response bodies of the teacher's spec ("RAG API — test spetsifikatsiyasi
(lex.uz)", 2026-11-03), field for field. Section numbers refer to that document."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from lexrag.rag.pipeline import ItemResult
from lexrag.rag.retrieve import Context

# ---- request (section 3) --------------------------------------------------------------


class Options(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    return_contexts: bool = True
    return_debug: bool = False
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = 42
    as_of_date: date | None = None
    answer_language: Literal["auto", "uz-Latn", "uz-Cyrl", "ru"] = "auto"


class Item(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)
    language: Literal["uz-Latn", "uz-Cyrl", "ru"] | None = None

    @field_validator("question")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question is empty")
        return v


class BatchRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=128)
    batch_id: str = Field(min_length=1, max_length=128)
    batch_index: int = Field(ge=1)
    total_batches: int = Field(ge=1)
    options: Options = Options()
    items: list[Item] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_ids(self) -> "BatchRequest":
        ids = [i.id for i in self.items]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate item ids: {', '.join(dup)}")
        return self


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language: Literal["uz-Latn", "uz-Cyrl", "ru"] | None = None
    options: Options = Options()


# ---- response (section 4) -------------------------------------------------------------


class System(BaseModel):
    pipeline_version: str
    llm_model: str
    embedding_model: str
    reranker: str | None
    index_version: str
    lex_snapshot_date: str


class Citation(BaseModel):
    ref: int
    doc_id: str  # "lex:6445145": the original act, https://lex.uz/docs/6445145
    doc_aliases: list[str]  # extra: every lex.uz id of the act (uz Cyrillic, Latin, ru, en)
    doc_title: str
    doc_type: str
    url: str
    article: str | None
    part: str | None
    edition_date: date | None
    doc_status: str
    chunk_id: str
    quote: str
    source_url: str  # extra: the Latin page with the paragraph anchor


class ContextOut(BaseModel):
    rank: int
    chunk_id: str
    doc_id: str
    doc_aliases: list[str]
    doc_title: str
    article: str | None
    text: str | None
    retrieval_score: float
    rerank_score: float | None
    used_in_answer: bool


class ErrorOut(BaseModel):
    code: str
    message: str
    retryable: bool


class Result(BaseModel):
    id: str
    status: Literal["ok", "no_answer", "error", "timeout"]
    answer: str | None
    answer_language: str | None
    no_answer: bool
    confidence: float | None
    citations: list[Citation]
    contexts: list[ContextOut]
    debug: dict | None = None
    timing: dict[str, int]
    usage: dict[str, int] | None
    error: ErrorOut | None


class Summary(BaseModel):
    total: int
    ok: int
    no_answer: int
    error: int


class Timing(BaseModel):
    received_at: datetime
    completed_at: datetime | None
    total_ms: int | None


class BatchResponse(BaseModel):
    run_id: str
    batch_id: str
    status: Literal["processing", "completed", "failed"]
    system: System | None = None
    timing: Timing | None = None
    summary: Summary | None = None
    results: list[Result] | None = None


def lex(doc_id: int) -> str:
    return f"lex:{doc_id}"


def aliases(c: Context) -> list[str]:
    """The reported id first, then the act's other lex.uz ids."""
    return [lex(c.lex_id), *(lex(a) for a in sorted(c.aliases) if a != c.lex_id)]


def to_result(r: ItemResult, options: Options) -> Result:
    a = r.answer
    used = a.used_contexts if a else set()
    contexts = [
        ContextOut(
            rank=i,
            chunk_id=c.chunk_id,
            doc_id=lex(c.lex_id),
            doc_aliases=aliases(c),
            doc_title=c.doc_title,
            article=c.article,
            text=c.text if options.return_contexts else None,
            retrieval_score=round(c.similarity, 4),
            rerank_score=round(c.rerank_score, 4) if c.rerank_score is not None else None,
            used_in_answer=c.chunk_id in used,
        )
        for i, c in enumerate(r.contexts, start=1)
    ]
    citations = [
        Citation(
            ref=cit.ref,
            doc_id=lex(cit.context.lex_id),
            doc_aliases=aliases(cit.context),
            doc_title=cit.context.doc_title,
            doc_type=cit.context.doc_type,
            url=cit.context.url,
            article=cit.context.article,
            part=cit.part,
            edition_date=cit.context.edition_date,
            doc_status=cit.context.doc_status,
            chunk_id=cit.context.chunk_id,
            quote=cit.quote,
            source_url=f"{cit.context.source_url}#{cit.elem_id}",
        )
        for cit in (a.citations if a else [])
    ]
    debug = None
    if options.return_debug and r.retrieval:
        debug = {
            "rewritten_query": r.retrieval.query,
            "retrieved_total": r.retrieval.retrieved_total,
            "detected_language": r.language,
            "rejected_citations": a.rejected if a else [],
        }
    return Result(
        id=r.id,
        status=r.status,  # type: ignore[arg-type]
        answer=a.answer if a else None,
        answer_language=a.language if a else None,
        no_answer=bool(a and a.no_answer),
        confidence=a.confidence if a else None,
        citations=citations,
        contexts=contexts,
        debug=debug,
        timing=r.timing_ms,
        usage={"input_tokens": a.input_tokens, "output_tokens": a.output_tokens} if a else None,
        error=ErrorOut(**r.error) if r.error else None,
    )
