"""Silver (data/parsed) → gold (Postgres): incremental, idempotent indexing.

For every parsed document:
  1. build its `documents` row and its chunks;
  2. skip it if `index_hash` (text + metadata + chunker version + model) is unchanged;
  3. otherwise embed only chunks whose `embed_hash` has no stored vector, and swap the
     document's rows in one transaction.
A second run over unchanged data embeds nothing and costs nothing. Documents that left the
scope (repealed, removed) are deleted on a full run.
"""

import hashlib
import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import psycopg

from lexrag.config import Settings
from lexrag.index import db
from lexrag.index.chunk import CHUNKER_VERSION, chunk_document
from lexrag.index.embed import Embedder
from lexrag.models import ActForm, DocRef, DocStatus, Document, IndexedDocument
from lexrag.storage import ParsedStore, RawStore

log = logging.getLogger(__name__)

PIPELINE_VERSION = "0.2.0"

_DOC_TYPE = {
    ActForm.CONSTITUTION: "konstitutsiya",
    ActForm.CODE: "kodeks",
    ActForm.LAW: "qonun",
    ActForm.DECREE: "farmon",
    ActForm.DIRECTIVE: "farmoyish",
    ActForm.RESOLUTION: "qaror",
    ActForm.ORDER: "buyruq",
}
_DOC_STATUS = {
    DocStatus.IN_FORCE: "amalda",
    DocStatus.REPEALED: "kuchini yo'qotgan",
    DocStatus.UNKNOWN: "noma'lum",
}


def edition_on(versions: list[date], snapshot: date, fallback: date | None) -> date | None:
    """The edition in force on `snapshot`. lex.uz also lists editions that start later
    (26 of our documents have one, e.g. 01.01.2028); those are not the text we hold."""
    past = [v for v in versions if v <= snapshot]
    return max(past) if past else fallback


def document_row(
    doc: Document, ref: DocRef | None, snapshot: date, settings: Settings
) -> IndexedDocument:
    lex_id = doc.twins.get(settings.canonical_id, doc.doc_id)
    form = ref.form if ref else None
    return IndexedDocument(
        doc_id=doc.doc_id,
        lex_id=lex_id,
        aliases=sorted({doc.doc_id, *doc.twins.values()}),
        title=doc.title,
        doc_type=_DOC_TYPE.get(form, "qonun") if form else "qonun",
        number=doc.number,
        adopted=doc.adopted,
        edition_date=edition_on(doc.versions, snapshot, doc.entry_into_force or doc.adopted),
        doc_status=_DOC_STATUS[ref.status if ref else DocStatus.UNKNOWN],
        url=f"{settings.base_url}/docs/{lex_id}",
        source_url=doc.url,
        content_hash=doc.content_hash,
        snapshot_date=snapshot,
    )


def index_hash(row: IndexedDocument, model: str) -> str:
    payload = json.dumps(
        [row.model_dump(mode="json"), model, CHUNKER_VERSION], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def snapshot_of(raw: RawStore, doc_id: int) -> date:
    """Download day of the raw HTML (file mtime; Phase 1 wrote each file once)."""
    return datetime.fromtimestamp(raw.path(doc_id).stat().st_mtime).date()


@dataclass
class IndexReport:
    docs_seen: int = 0
    docs_skipped: int = 0
    docs_written: int = 0
    docs_deleted: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0
    chunks_reused: int = 0
    snapshot_date: date | None = None


def index_documents(
    conn: psycopg.Connection,
    ids: Iterable[int],
    parsed: ParsedStore,
    raw: RawStore,
    catalog: dict[int, DocRef],
    embedder: Embedder,
    settings: Settings,
    *,
    prune: bool,
    snapshot: Callable[[int], date] | None = None,
) -> IndexReport:
    snapshot = snapshot or (lambda doc_id: snapshot_of(raw, doc_id))
    report = IndexReport()
    ids = list(ids)
    existing = db.index_hashes(conn)

    for n, doc_id in enumerate(ids, start=1):
        report.docs_seen += 1
        doc = parsed.load(doc_id)
        snap = snapshot(doc_id)
        report.snapshot_date = max(filter(None, [report.snapshot_date, snap]))
        row = document_row(doc, catalog.get(doc_id), snap, settings)
        h = index_hash(row, embedder.model)
        if existing.get(doc_id) == h:
            report.docs_skipped += 1
            continue

        chunks = chunk_document(doc, row, embedder.model)
        stored = db.stored_vectors(conn, doc_id)
        todo = [c for c in chunks if c.embed_hash not in stored]
        fresh = dict(
            zip(
                (c.embed_hash for c in todo),
                embedder.embed([c.embed_text for c in todo]) if todo else [],
                strict=True,
            )
        )
        vectors = [
            np.asarray(fresh.get(c.embed_hash, stored.get(c.embed_hash)), dtype=np.float32)
            for c in chunks
        ]
        with conn.transaction():
            db.replace_document(conn, row, h, chunks, vectors)
        report.docs_written += 1
        report.chunks_written += len(chunks)
        report.chunks_embedded += len(todo)
        report.chunks_reused += len(chunks) - len(todo)
        log.info(
            "[%d/%d] %s: %d chunks (%d embedded) — %s",
            n, len(ids), doc_id, len(chunks), len(todo), doc.title[:60],
        )  # fmt: skip

    if prune:
        gone = set(existing) - set(ids)
        with conn.transaction():
            report.docs_deleted = db.delete_documents(conn, gone)

    changed = report.docs_written or report.docs_deleted or not db.read_meta(conn)
    if report.snapshot_date and changed:
        with conn.transaction():
            db.write_meta(
                conn,
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "index_version": f"lex-idx-{report.snapshot_date.isoformat()}",
                    "lex_snapshot_date": report.snapshot_date.isoformat(),
                    "embedding_model": embedder.model,
                    "embedding_dim": str(embedder.dim),
                    "chunker_version": CHUNKER_VERSION,
                    "canonical_id": settings.canonical_id,
                },
            )
    return report
