"""Copy the built index (documents, chunks, index_meta) from one Postgres to another,
e.g. local Docker → Neon. Vectors are copied, not recomputed: no OpenRouter cost.

    uv run python scripts/copy_index.py --to "$LEXRAG_NEON_URL"

The target's tables are emptied first. The HNSW index is built after the rows are in:
one bulk build is much faster than updating the graph on every insert.
"""

import argparse
import os
import sys
import time

from lexrag.config import get_settings
from lexrag.index import db

TABLES = {  # the generated `tsv` column is left out: the target computes it
    "documents": db._DOC_COLUMNS + ", indexed_at",
    "chunks": db._CHUNK_COLUMNS,
    "index_meta": "key, value",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", default=get_settings().database_url)
    ap.add_argument("--to", dest="target", default=os.environ.get("LEXRAG_NEON_URL", ""))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if not args.target:
        raise SystemExit("give --to or set LEXRAG_NEON_URL")

    s = get_settings()
    src = db.connect(args.source)
    dst = db.connect(args.target, timeout_s=30)
    print("target:", dst.execute("SELECT version()").fetchone()[0][:60])
    print("pgvector:", dst.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0])  # fmt: skip

    db.init_schema(dst, s.embedding_dim)
    dst.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
    dst.execute("TRUNCATE chunks, documents, index_meta")

    for table, columns in TABLES.items():
        t = time.perf_counter()
        n = 0
        with (
            src.cursor().copy(f"COPY {table} ({columns}) TO STDOUT (FORMAT binary)") as out,
            dst.cursor().copy(f"COPY {table} ({columns}) FROM STDIN (FORMAT binary)") as inp,
        ):
            for block in out:
                inp.write(block)
                n += len(block)
        count = dst.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows, {n / 1e6:.0f} MB in {time.perf_counter() - t:.0f} s")

    t = time.perf_counter()
    dst.execute("SET maintenance_work_mem = '256MB'")
    dst.execute(
        "CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)"
    )
    print(f"  HNSW index built in {time.perf_counter() - t:.0f} s")

    for table in ("documents", "chunks"):
        a = src.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        b = dst.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert a == b, f"{table}: {a} rows locally, {b} on the target"
    size = dst.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
    print(f"done: counts match; target database size {size}")
    print("meta:", db.read_meta(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
