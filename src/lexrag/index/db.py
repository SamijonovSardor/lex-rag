"""Postgres + pgvector: the gold layer.

Two ways to find a chunk, both indexed here:
  embedding  vector(1024) + HNSW  → meaning ("dam olish kuni" ≈ "ta'til")
  tsv        tsvector + GIN       → exact words and numbers ("18¹-modda", "BHM")
Phase 4 combines them (hybrid retrieval).
"""

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from lexrag.models import Chunk, IndexedDocument

# 'simple' text search config: lower-cases and splits words, no stemming. Postgres ships
# no Uzbek dictionary, and a Russian or English stemmer would mangle Uzbek words.
SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  doc_id        bigint PRIMARY KEY,
  lex_id        bigint NOT NULL,
  aliases       bigint[] NOT NULL,
  title         text NOT NULL,
  doc_type      text NOT NULL,
  number        text,
  adopted       date,
  edition_date  date,
  doc_status    text NOT NULL,
  url           text NOT NULL,
  source_url    text NOT NULL,
  content_hash  text NOT NULL,
  snapshot_date date NOT NULL,
  index_hash    text NOT NULL,
  indexed_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_aliases_idx ON documents USING gin (aliases);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id      text PRIMARY KEY,
  doc_id        bigint NOT NULL REFERENCES documents ON DELETE CASCADE,
  unit_position int NOT NULL,
  elem_id       bigint NOT NULL,
  kind          text NOT NULL,
  article       text,
  article_title text,
  piece         int NOT NULL,
  pieces        int NOT NULL,
  heading       text NOT NULL,
  path          text[] NOT NULL,
  paragraphs    jsonb NOT NULL,
  amendments    text[] NOT NULL,
  text          text NOT NULL,
  embed_hash    text NOT NULL,
  embedding     vector({dim}) NOT NULL,
  tsv           tsvector GENERATED ALWAYS AS (
                  to_tsvector('simple', coalesce(article, '') || ' ' || heading || ' ' || text)
                ) STORED
);
CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

CREATE TABLE IF NOT EXISTS index_meta (
  key   text PRIMARY KEY,
  value text NOT NULL
);
"""

_DOC_COLUMNS = (
    "doc_id, lex_id, aliases, title, doc_type, number, adopted, edition_date, doc_status, "
    "url, source_url, content_hash, snapshot_date, index_hash"
)
_CHUNK_COLUMNS = (
    "chunk_id, doc_id, unit_position, elem_id, kind, article, article_title, piece, pieces, "
    "heading, path, paragraphs, amendments, text, embed_hash, embedding"
)


def connect(url: str, *, timeout_s: int = 10) -> psycopg.Connection:
    """Autocommit connection; multi-statement writes use `with conn.transaction():`."""
    conn = psycopg.connect(url, autocommit=True, connect_timeout=timeout_s)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    conn.execute(SCHEMA.replace("{dim}", str(int(dim))))


def index_hashes(conn: psycopg.Connection) -> dict[int, str]:
    return dict(conn.execute("SELECT doc_id, index_hash FROM documents").fetchall())


def stored_vectors(conn: psycopg.Connection, doc_id: int) -> dict[str, np.ndarray]:
    rows = conn.execute(
        "SELECT embed_hash, embedding FROM chunks WHERE doc_id = %s", (doc_id,)
    ).fetchall()
    return {h: v.to_numpy() for h, v in rows}


def replace_document(
    conn: psycopg.Connection,
    doc: IndexedDocument,
    index_hash: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
) -> None:
    """Upsert the document and swap in its chunks, in the caller's transaction."""
    d = doc.model_dump()
    conn.execute(
        f"INSERT INTO documents ({_DOC_COLUMNS}) VALUES ({', '.join(['%s'] * 14)}) "
        "ON CONFLICT (doc_id) DO UPDATE SET "
        + ", ".join(f"{c} = EXCLUDED.{c}" for c in _DOC_COLUMNS.split(", ")[1:])
        + ", indexed_at = now()",
        (
            d["doc_id"], d["lex_id"], d["aliases"], d["title"], d["doc_type"], d["number"],
            d["adopted"], d["edition_date"], d["doc_status"], d["url"], d["source_url"],
            d["content_hash"], d["snapshot_date"], index_hash,
        ),
    )  # fmt: skip
    conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc.doc_id,))
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO chunks ({_CHUNK_COLUMNS}) VALUES ({', '.join(['%s'] * 16)})",
            [
                (
                    c.chunk_id, c.doc_id, c.unit_position, c.elem_id, c.kind, c.article,
                    c.article_title, c.piece, c.pieces, c.heading, c.path,
                    Jsonb([p.model_dump() for p in c.paragraphs]), c.amendments, c.text,
                    c.embed_hash, v,
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )  # fmt: skip


def delete_documents(conn: psycopg.Connection, doc_ids: Iterable[int]) -> int:
    ids = list(doc_ids)
    if not ids:
        return 0
    return conn.execute("DELETE FROM documents WHERE doc_id = ANY(%s)", (ids,)).rowcount


def write_meta(conn: psycopg.Connection, meta: dict[str, str]) -> None:
    meta = {**meta, "built_at": datetime.now(UTC).isoformat(timespec="seconds")}
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO index_meta (key, value) VALUES (%s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            list(meta.items()),
        )


def read_meta(conn: psycopg.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM index_meta").fetchall())
