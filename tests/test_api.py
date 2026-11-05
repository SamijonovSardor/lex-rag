"""The API against the teacher's checklist (spec section 7), with a fake RAG pipeline.

Needs the Postgres of `docker compose up -d db`; works in a throwaway schema.
"""

import threading
import time
import uuid
from collections.abc import Iterator
from typing import ClassVar

import psycopg
import pytest
from fastapi.testclient import TestClient
from test_answer import COMPLETION, CONTEXTS

from lexrag.api import app as api
from lexrag.config import Settings
from lexrag.index import db
from lexrag.rag.answer import build_answer
from lexrag.rag.pipeline import ItemResult
from lexrag.rag.retrieve import Retrieval

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}
BASE_URL = Settings().database_url


class FakePipeline:
    calls: ClassVar[list[str]] = []
    fail_once: ClassVar[set[str]] = set()
    delay_s = 0.0

    def run(self, item_id: str, question: str, **kw) -> ItemResult:
        FakePipeline.calls.append(item_id)
        time.sleep(FakePipeline.delay_s)
        retrieval = Retrieval(query=question, contexts=CONTEXTS, retrieved_total=12,
                              timing_ms={"retrieval_ms": 5})  # fmt: skip
        if item_id in FakePipeline.fail_once:
            FakePipeline.fail_once.discard(item_id)
            return ItemResult(
                id=item_id, status="error", language="uz-Latn", retrieval=None,
                error={"code": "LLM_TIMEOUT", "message": "slow", "retryable": True},
                timing_ms={"total_ms": 1},
            )  # fmt: skip
        if "javobsiz" in question:
            data = {"no_answer": True, "answer": "Topilmadi.", "citations": [], "confidence": 0.1}
        else:
            data = {
                "no_answer": False,
                "answer": "Kamida yigirma bir kalendar kun [1].",
                "citations": [{"ref": 1, "context": 1, "quote": "kamida yigirma bir kalendar kun"}],
                "confidence": 0.8,
            }
        return ItemResult(
            id=item_id, status="no_answer" if data["no_answer"] else "ok", language="uz-Latn",
            answer=build_answer(data, CONTEXTS, "uz-Latn", COMPLETION), retrieval=retrieval,
            timing_ms={"retrieval_ms": 5, "generation_ms": 7, "total_ms": 12},
        )  # fmt: skip


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    try:
        conn = db.connect(BASE_URL, timeout_s=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no database: {e}")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    conn.execute(f"CREATE SCHEMA {schema}")
    conn.execute(f"SET search_path TO {schema}, public")
    db.init_schema(conn, 1024)
    db.write_meta(conn, {"index_version": "lex-idx-2026-10-30", "lex_snapshot_date": "2026-10-30",
                         "embedding_model": "baai/bge-m3"})  # fmt: skip
    settings = Settings(
        database_url=f"{BASE_URL}?options=-csearch_path%3D{schema},public",
        api_keys=KEY, rate_limit_per_s=50, batch_sync_timeout_s=5,
    )  # fmt: skip
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    local = threading.local()
    monkeypatch.setattr(
        api.Runtime, "pipeline", lambda self: getattr(local, "p", None) or FakePipeline()
    )
    FakePipeline.calls, FakePipeline.fail_once, FakePipeline.delay_s = [], set(), 0.0
    try:
        with TestClient(api.app) as c:
            yield c
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


def batch(batch_id: str = "run-b01", n: int = 2, **over) -> dict:
    body = {
        "run_id": "run",
        "batch_id": batch_id,
        "batch_index": 1,
        "total_batches": 10,
        "options": {"top_k": 5, "return_contexts": True, "return_debug": True, "seed": 42},
        "items": [
            {"id": f"q{i:03d}", "question": f"Taʼtil necha kun? {i}"} for i in range(1, n + 1)
        ],
    }
    return {**body, **over}


def test_auth_is_required(client: TestClient) -> None:
    assert client.post("/v1/rag/batch", json=batch()).status_code == 401
    r = client.post("/v1/rag/batch", json=batch(), headers={"Authorization": "Bearer nope"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "FORBIDDEN"


@pytest.mark.parametrize(
    ("body", "field"),
    [
        (batch(n=11), "items"),
        (batch(items=[{"id": "q1", "question": "a"}, {"id": "q1", "question": "b"}]), ""),
        (batch(items=[{"id": "q1", "question": "   "}]), "items.0.question"),
        (batch(items=[]), "items"),
    ],
)
def test_validation_errors_are_400(client: TestClient, body: dict, field: str) -> None:
    r = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "VALIDATION_ERROR"
    assert any(d["field"].startswith(field) for d in err["details"])


def test_batch_answers_every_item_in_spec_shape(client: TestClient) -> None:
    body = batch(n=3)
    body["items"][2]["question"] = "javobsiz savol"
    r = client.post("/v1/rag/batch", json=body, headers=AUTH)
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/json; charset=utf-8"
    out = r.json()
    assert (out["run_id"], out["batch_id"], out["status"]) == ("run", "run-b01", "completed")
    assert out["system"]["index_version"] == "lex-idx-2026-10-30"
    assert out["summary"] == {"total": 3, "ok": 2, "no_answer": 1, "error": 0}
    assert [x["id"] for x in out["results"]] == ["q001", "q002", "q003"]

    first = out["results"][0]
    cit = first["citations"][0]
    assert first["answer"] == "Kamida yigirma bir kalendar kun [1]."
    assert (cit["ref"], cit["doc_id"], cit["article"], cit["part"]) == (
        1,
        "lex:1",
        "134-modda",
        "1-qism",
    )
    assert cit["quote"] == "kamida yigirma bir kalendar kun"
    assert cit["doc_aliases"] == ["lex:1", "lex:-1", "lex:7"]  # reported id first
    assert first["contexts"][0]["doc_aliases"][0] == "lex:1"
    assert {cit["edition_date"], cit["doc_status"]} == {"2026-01-01", "amalda"}
    ctx_ids = {c["chunk_id"] for c in first["contexts"]}
    assert cit["chunk_id"] in ctx_ids  # a citation never points outside the contexts
    assert first["contexts"][0]["used_in_answer"] is True
    assert first["contexts"][0]["retrieval_score"] == 0.8
    assert first["debug"]["retrieved_total"] == 12
    assert out["results"][2]["no_answer"] is True and out["results"][2]["citations"] == []


def test_same_batch_id_returns_stored_result_and_other_content_conflicts(
    client: TestClient,
) -> None:
    first = client.post("/v1/rag/batch", json=batch(), headers=AUTH).json()
    again = client.post("/v1/rag/batch", json=batch(), headers=AUTH)
    assert again.status_code == 200 and again.json() == first
    assert len(FakePipeline.calls) == 2  # not recomputed

    other = batch()
    other["items"][0]["question"] = "boshqa savol"
    r = client.post("/v1/rag/batch", json=other, headers=AUTH)
    assert r.status_code == 409 and r.json()["error"]["code"] == "BATCH_ID_CONFLICT"


def test_one_failing_item_does_not_sink_the_batch_and_is_retried(client: TestClient) -> None:
    FakePipeline.fail_once = {"q002"}
    out = client.post("/v1/rag/batch", json=batch(), headers=AUTH).json()
    assert out["summary"] == {"total": 2, "ok": 1, "no_answer": 0, "error": 1}
    failed = out["results"][1]
    assert failed["status"] == "error" and failed["error"]["retryable"] is True
    assert failed["error"]["code"] == "LLM_TIMEOUT"

    FakePipeline.calls = []
    again = client.post("/v1/rag/batch", json=batch(), headers=AUTH).json()
    assert FakePipeline.calls == ["q002"]  # only the failed item ran again
    assert again["summary"]["ok"] == 2


def test_slow_batch_goes_async(client: TestClient) -> None:
    client.app.state.rt.settings = client.app.state.rt.settings.model_copy(
        update={"batch_sync_timeout_s": 0.05}
    )
    FakePipeline.delay_s = 0.3
    r = client.post("/v1/rag/batch", json=batch("slow-b01"), headers=AUTH)
    assert r.status_code == 202 and r.json()["status"] == "processing"
    for _ in range(50):
        g = client.get("/v1/rag/batch/slow-b01", headers=AUTH)
        if g.json()["status"] == "completed":
            break
        time.sleep(0.1)
    assert g.status_code == 200 and g.json()["summary"]["total"] == 2


def test_get_unknown_batch_is_404(client: TestClient) -> None:
    assert client.get("/v1/rag/batch/nope", headers=AUTH).status_code == 404


def test_rate_limit_answers_429_with_retry_after(client: TestClient) -> None:
    client.app.state.rt.settings = client.app.state.rt.settings.model_copy(
        update={"rate_limit_per_s": 1}
    )
    assert client.get("/v1/rag/batch/nope", headers=AUTH).status_code == 404
    r = client.get("/v1/rag/batch/nope", headers=AUTH)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1


def test_meta_and_health(client: TestClient) -> None:
    meta = client.get("/v1/meta").json()
    assert meta["index_version"] == "lex-idx-2026-10-30"
    assert meta["lex_snapshot_date"] == "2026-10-30"
    assert meta["llm_model"] == "openai/gpt-6-luna"
    health = client.get("/v1/health")
    assert health.status_code == 503  # the throwaway schema has no chunks
    assert health.json()["status"] == "degraded"


def test_uzbek_letters_survive_the_round_trip(client: TestClient) -> None:
    question = "Oʻgʻil va qiz: taʼtil necha kun? gʻ oʻ"
    body = batch(items=[{"id": "q1", "question": question}])
    raw = client.post("/v1/rag/batch", json=body, headers=AUTH).content.decode("utf-8")
    assert question in raw  # sent back unescaped, okina and tutuq intact (debug echo)
    assert "taʼtil" in raw


def test_root_needs_no_key_and_no_database(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200 and r.json()["service"] == "lex-rag"
