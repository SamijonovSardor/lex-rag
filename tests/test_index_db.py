"""Indexer against a real Postgres + pgvector (`docker compose up -d db`).

Each test works in its own throwaway schema, so the real index is never touched.
Skipped when the database is not reachable.
"""

import hashlib
import os
import uuid
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
import pytest
from test_chunk import doc_with, unit

from lexrag.config import Settings
from lexrag.index import db
from lexrag.index.indexer import index_documents
from lexrag.models import ActForm, DocRef, DocStatus, Document
from lexrag.storage import ParsedStore, RawStore

DB_URL = os.environ.get("LEXRAG_TEST_DATABASE_URL", Settings().database_url)
DIM = 1024


class FakeEmbedder:
    model = "fake/model"
    dim = DIM

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded += texts
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).standard_normal(DIM)
            out.append((v / np.linalg.norm(v)).tolist())
        return out


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    try:
        c = db.connect(DB_URL, timeout_s=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no database at {DB_URL}: {e}")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    c.execute(f"CREATE SCHEMA {schema}")
    c.execute(f"SET search_path TO {schema}, public")
    db.init_schema(c, DIM)
    try:
        yield c
    finally:
        c.execute(f"DROP SCHEMA {schema} CASCADE")
        c.close()


def store(tmp_path: Path, *docs: Document) -> ParsedStore:
    s = ParsedStore(tmp_path / "parsed")
    for d in docs:
        s.save(d)
    return s


def run(conn, parsed: ParsedStore, ids, embedder, *, prune=True):
    catalog = {
        i: DocRef(doc_id=i, title="t", status=DocStatus.IN_FORCE, form=ActForm.LAW) for i in ids
    }
    return index_documents(
        conn, ids, parsed, RawStore(Path("unused")), catalog, embedder, Settings(),
        prune=prune, snapshot=lambda _: date(2026, 10, 30),
    )  # fmt: skip


def two_docs() -> tuple[Document, Document]:
    a = doc_with(unit(0, "1", "Har kim mehnat qilish huquqiga ega."), unit(1, "2", "Oʻzbekiston."))
    b = doc_with(unit(0, "1", "Soliq toʻlash majburiydir."))
    b = b.model_copy(update={"doc_id": -200, "twins": {"uz_cyrl": 200}})
    return a, b


def test_second_run_embeds_nothing(conn, tmp_path: Path) -> None:
    a, b = two_docs()
    parsed, emb = store(tmp_path, a, b), FakeEmbedder()

    first = run(conn, parsed, [a.doc_id, b.doc_id], emb)
    assert (first.docs_written, first.chunks_written, first.chunks_embedded) == (2, 3, 3)
    meta = db.read_meta(conn)
    assert meta["index_version"] == "lex-idx-2026-10-30"
    assert meta["embedding_model"] == "fake/model"

    again = run(conn, parsed, [a.doc_id, b.doc_id], emb)
    assert (again.docs_skipped, again.docs_written, again.chunks_embedded) == (2, 0, 0)
    assert len(emb.embedded) == 3


def test_changed_document_reembeds_only_changed_chunks(conn, tmp_path: Path) -> None:
    a, b = two_docs()
    parsed, emb = store(tmp_path, a, b), FakeEmbedder()
    run(conn, parsed, [a.doc_id, b.doc_id], emb)

    new_unit = unit(1, "2", "Oʻzbekiston Respublikasi.")
    edited = a.model_copy(update={"units": [a.units[0], new_unit], "content_hash": "h2"})
    parsed.save(edited)
    report = run(conn, parsed, [a.doc_id, b.doc_id], emb)
    assert (report.docs_written, report.docs_skipped) == (1, 1)
    assert (report.chunks_embedded, report.chunks_reused) == (1, 1)
    text = conn.execute("SELECT text FROM chunks WHERE chunk_id = 'lex:100#art-2#p1'").fetchone()
    assert text == ("Oʻzbekiston Respublikasi.",)


def test_prune_removes_documents_that_left_scope(conn, tmp_path: Path) -> None:
    a, b = two_docs()
    parsed, emb = store(tmp_path, a, b), FakeEmbedder()
    run(conn, parsed, [a.doc_id, b.doc_id], emb)

    partial = run(conn, parsed, [a.doc_id], emb, prune=False)
    assert partial.docs_deleted == 0
    full = run(conn, parsed, [a.doc_id], emb)
    assert full.docs_deleted == 1
    assert conn.execute("SELECT count(*) FROM chunks WHERE doc_id = -200").fetchone() == (0,)


def test_both_search_paths_find_the_chunk(conn, tmp_path: Path) -> None:
    a, b = two_docs()
    parsed, emb = store(tmp_path, a, b), FakeEmbedder()
    run(conn, parsed, [a.doc_id, b.doc_id], emb)

    # Full text: the Uzbek okina (ʻ) stays inside the word.
    hit = conn.execute(
        "SELECT chunk_id FROM chunks WHERE tsv @@ plainto_tsquery('simple', %s)", ("toʻlash",)
    ).fetchall()
    assert hit == [("lex:200#art-1#p1",)]

    # Vector: a chunk's own vector is its nearest neighbour.
    target = conn.execute(
        "SELECT embedding FROM chunks WHERE chunk_id = 'lex:100#art-1#p1'"
    ).fetchone()[0]
    nearest = conn.execute(
        "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT 1", (target,)
    ).fetchone()
    assert nearest == ("lex:100#art-1#p1",)

    paragraphs = conn.execute(
        "SELECT paragraphs FROM chunks WHERE chunk_id = 'lex:100#art-1#p1'"
    ).fetchone()[0]
    assert paragraphs[0]["qism"] == 1
