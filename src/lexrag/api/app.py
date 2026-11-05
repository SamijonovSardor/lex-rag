"""The HTTP API of the teacher's spec.

    POST /v1/rag/batch            1–10 questions → answers, citations, contexts
    GET  /v1/rag/batch/{id}       a stored batch (async polling, re-checks)
    POST /v1/rag/query            one question (debug)
    GET  /v1/meta                 models, index version, lex.uz snapshot date
    GET  /v1/health               liveness + index loaded

    uv run uvicorn lexrag.api.app:app --port 8000

Questions of a batch run in parallel on `api_workers` threads, each with its own database
connection and clients. A batch that takes longer than `batch_sync_timeout_s` is answered
with 202 and keeps running; GET returns it when done.
"""

import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool

from lexrag.api.schemas import (
    BatchRequest,
    BatchResponse,
    Options,
    QueryRequest,
    Summary,
    System,
    Timing,
    to_result,
)
from lexrag.api.store import BatchStore
from lexrag.config import Settings, get_settings
from lexrag.index import db
from lexrag.index.indexer import PIPELINE_VERSION
from lexrag.rag.pipeline import RagPipeline

log = logging.getLogger("lexrag.api")


class Utf8JSON(JSONResponse):
    media_type = "application/json; charset=utf-8"


def error(status: int, code: str, message: str, **extra: Any) -> Utf8JSON:
    return Utf8JSON(
        status_code=status, content={"error": {"code": code, "message": message, **extra}}
    )


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


class Runtime:
    """Everything the endpoints share, built once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pool = ConnectionPool(settings.database_url, min_size=1, max_size=4, open=True)
        self.store = BatchStore(self.pool)
        abandoned = self.store.fail_abandoned()
        if abandoned:
            log.warning(
                "%d batches were left processing by a previous run: marked failed", abandoned
            )
        self.items = ThreadPoolExecutor(settings.api_workers, thread_name_prefix="item")
        self.batches = ThreadPoolExecutor(4, thread_name_prefix="batch")
        self.running: dict[str, Future] = {}
        self._local = threading.local()
        self._pipelines: list[RagPipeline] = []
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def pipeline(self) -> RagPipeline:
        p = getattr(self._local, "pipeline", None)
        if p is None:
            p = RagPipeline(db.connect(self.settings.database_url), self.settings)
            self._local.pipeline = p
            with self._lock:
                self._pipelines.append(p)
        return p

    def system(self) -> System:
        with self.pool.connection() as conn:
            meta = db.read_meta(conn)
        return System(
            pipeline_version=PIPELINE_VERSION,
            llm_model=self.settings.llm_model,
            embedding_model=meta.get("embedding_model", self.settings.embedding_model),
            reranker=self.settings.rerank_model or None,
            index_version=meta.get("index_version", "none"),
            lex_snapshot_date=meta.get("lex_snapshot_date", ""),
        )

    def allow(self, key: str) -> float | None:
        """Sliding one-second window. Returns seconds to wait, or None if allowed."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 1.0:
                hits.popleft()
            if len(hits) >= self.settings.rate_limit_per_s:
                return max(0.1, 1.0 - (now - hits[0]))
            hits.append(now)
        return None

    def close(self) -> None:
        self.items.shutdown(wait=False, cancel_futures=True)
        self.batches.shutdown(wait=False, cancel_futures=True)
        for p in self._pipelines:
            p.retriever.conn.close()
            p.close()
        self.pool.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rt = Runtime(get_settings())
    yield
    app.state.rt.close()


app = FastAPI(title="lex-rag", version=PIPELINE_VERSION, lifespan=lifespan)


@app.exception_handler(ApiError)
async def _api_error(_: Request, e: ApiError) -> Utf8JSON:
    resp = error(e.status, e.code, e.message)
    for k, v in (e.headers or {}).items():
        resp.headers[k] = v
    return resp


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, e: RequestValidationError) -> Utf8JSON:
    details = [
        {"field": ".".join(str(x) for x in err["loc"][1:]), "message": err["msg"]}
        for err in e.errors()
    ]
    return error(400, "VALIDATION_ERROR", "invalid request", details=details)


def rt(request: Request) -> Runtime:
    return request.app.state.rt


def api_key(request: Request, authorization: str | None = Header(default=None)) -> str:
    runtime = rt(request)
    if not authorization or not authorization.startswith("Bearer "):
        raise ApiError(401, "UNAUTHORIZED", "missing 'Authorization: Bearer <API_KEY>' header")
    key = authorization.removeprefix("Bearer ").strip()
    if key not in runtime.settings.api_key_set:
        raise ApiError(403, "FORBIDDEN", "unknown API key")
    wait = runtime.allow(key)
    if wait is not None:
        raise ApiError(
            429, "RATE_LIMITED", "too many requests", {"Retry-After": str(int(wait + 0.999))}
        )
    return key


# ---- batch processing ------------------------------------------------------------------


def request_hash(req: BatchRequest) -> str:
    return hashlib.sha256(
        json.dumps(req.model_dump(mode="json"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def answer_item(runtime: Runtime, item_id: str, question: str, language: str | None, o: Options):
    return runtime.pipeline().run(
        item_id, question, language=language, answer_language=o.answer_language,
        top_k=o.top_k, temperature=o.temperature, seed=o.seed,
    )  # fmt: skip


def process(
    runtime: Runtime, req: BatchRequest, received_at: datetime, previous: dict | None
) -> dict[str, Any]:
    """Answer the batch (or, for a resubmission, only its retryable failures) and store it."""
    t0 = time.perf_counter()
    kept = {r["id"]: r for r in (previous or {}).get("results") or []}
    todo = [
        i for i in req.items
        if i.id not in kept or (kept[i.id]["error"] or {}).get("retryable")
    ]  # fmt: skip
    futures = {
        i.id: runtime.items.submit(answer_item, runtime, i.id, i.question, i.language, req.options)
        for i in todo
    }
    fresh = {
        k: to_result(f.result(), req.options).model_dump(mode="json") for k, f in futures.items()
    }
    results = [fresh.get(i.id) or kept[i.id] for i in req.items]
    summary = Summary(
        total=len(results),
        ok=sum(r["status"] == "ok" for r in results),
        no_answer=sum(r["status"] == "no_answer" for r in results),
        error=sum(r["status"] in ("error", "timeout") for r in results),
    )
    done = datetime.now(UTC)
    response = BatchResponse(
        run_id=req.run_id,
        batch_id=req.batch_id,
        status="completed",
        system=runtime.system(),
        timing=Timing(
            received_at=received_at,
            completed_at=done,
            total_ms=round((time.perf_counter() - t0) * 1000),
        ),
        summary=summary,
        results=results,  # type: ignore[arg-type]
    ).model_dump(mode="json")
    runtime.store.finish(req.batch_id, "completed", response, done)
    return response


def start(
    runtime: Runtime, req: BatchRequest, received_at: datetime, previous: dict | None
) -> Future:
    def job() -> dict[str, Any]:
        try:
            return process(runtime, req, received_at, previous)
        except Exception as e:
            log.exception("batch %s failed", req.batch_id)
            body = BatchResponse(run_id=req.run_id, batch_id=req.batch_id, status="failed")
            runtime.store.finish(
                req.batch_id, "failed", {**body.model_dump(mode="json"), "error": str(e)[:300]},
                datetime.now(UTC),
            )  # fmt: skip
            raise
        finally:
            runtime.running.pop(req.batch_id, None)

    fut = runtime.batches.submit(job)
    runtime.running[req.batch_id] = fut
    return fut


def processing(req_or_row: Any) -> Utf8JSON:
    return Utf8JSON(
        status_code=202,
        content={
            "run_id": req_or_row.run_id,
            "batch_id": req_or_row.batch_id,
            "status": "processing",
        },
    )


@app.post("/v1/rag/batch")
def post_batch(req: BatchRequest, request: Request, _: str = Depends(api_key)) -> Utf8JSON:
    runtime = rt(request)
    received_at = datetime.now(UTC)
    h = request_hash(req)
    stored = runtime.store.claim(req.batch_id, req.run_id, h, received_at)
    previous = None
    if stored is not None:
        if stored.request_hash != h:
            raise ApiError(409, "BATCH_ID_CONFLICT", "this batch_id was sent with other content")
        if stored.status == "processing" and req.batch_id in runtime.running:
            return processing(stored)
        response = stored.response or {}
        retry_items = [
            r for r in response.get("results") or [] if (r["error"] or {}).get("retryable")
        ]
        if stored.status == "completed" and not retry_items:
            return Utf8JSON(content=response)  # idempotent: the stored answer, not recomputed
        previous = response if stored.status == "completed" else None
        runtime.store.restart(req.batch_id)
        received_at = stored.received_at

    fut = start(runtime, req, received_at, previous)
    try:
        return Utf8JSON(content=fut.result(timeout=runtime.settings.batch_sync_timeout_s))
    except FutureTimeout:
        return processing(req)
    except Exception as e:
        raise ApiError(500, "INTERNAL_ERROR", f"batch failed: {str(e)[:200]}") from e


@app.get("/v1/rag/batch/{batch_id}")
def get_batch(batch_id: str, request: Request, _: str = Depends(api_key)) -> Utf8JSON:
    stored = rt(request).store.get(batch_id)
    if stored is None:
        raise ApiError(404, "NOT_FOUND", f"no batch {batch_id}")
    if stored.status == "processing":
        return processing(stored)
    return Utf8JSON(content=stored.response)


@app.post("/v1/rag/query")
def post_query(req: QueryRequest, request: Request, _: str = Depends(api_key)) -> Utf8JSON:
    runtime = rt(request)
    fut = runtime.items.submit(answer_item, runtime, "q", req.question, req.language, req.options)
    result = to_result(fut.result(), req.options).model_dump(mode="json")
    return Utf8JSON(content={"system": runtime.system().model_dump(mode="json"), "result": result})


@app.get("/v1/meta")
def meta(request: Request) -> Utf8JSON:
    runtime = rt(request)
    with runtime.pool.connection() as conn:
        docs, chunks = conn.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM chunks)"
        ).fetchone()
    return Utf8JSON(
        content={
            **runtime.system().model_dump(mode="json"),
            "documents": docs,
            "chunks": chunks,
            "canonical_doc_id": runtime.settings.canonical_id,
            "doc_id_note": (
                "doc_id is the lex.uz number of the act as in https://lex.uz/docs/{id} "
                "(the original text); doc_aliases lists the act's other lex.uz ids "
                "(Latin, Russian, English versions)"
            ),
            "languages": ["uz-Latn", "uz-Cyrl", "ru"],
            "max_batch_items": 10,
        }
    )


@app.get("/v1/health")
def health(request: Request) -> Utf8JSON:
    try:
        with rt(request).pool.connection() as conn:
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except Exception as e:
        return Utf8JSON(status_code=503, content={"status": "down", "error": str(e)[:200]})
    status = "ok" if chunks else "degraded"
    return Utf8JSON(
        status_code=200 if chunks else 503, content={"status": status, "chunks": chunks}
    )
