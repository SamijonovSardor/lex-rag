"""Command line entry points. Airflow tasks call the same functions; the CLI exists to run
and debug each pipeline stage by hand.

    lexrag discover                  # listing → data/catalog.jsonl
    lexrag fetch --limit 20          # in-scope catalog docs → data/raw/
    lexrag parse                     # data/raw → data/parsed/
"""

import argparse
import asyncio
import logging
import sys
from collections import Counter

from lexrag.config import get_settings
from lexrag.models import ActForm, DocRef
from lexrag.scraper.discover import discover, in_scope
from lexrag.scraper.fetch import fetch_many
from lexrag.scraper.http import LexClient
from lexrag.scraper.parse import check_document, parse_document
from lexrag.storage import ParsedStore, RawStore, read_jsonl, write_jsonl

log = logging.getLogger("lexrag")

_FORMS = {"constitution": ActForm.CONSTITUTION, "code": ActForm.CODE, "law": ActForm.LAW}


async def cmd_discover(args: argparse.Namespace) -> int:
    s = get_settings()
    forms = [_FORMS[f] for f in args.forms]
    async with LexClient(s) as client:
        refs = await discover(client, forms, max_pages=args.max_pages)
    n = write_jsonl(s.catalog_path, refs)
    scoped = [r for r in refs if in_scope(r)]
    by_form = Counter(r.form.name for r in scoped if r.form)
    print(f"catalog: {n} docs → {s.catalog_path}; in scope: {len(scoped)} {dict(by_form)}")
    return 0


async def cmd_fetch(args: argparse.Namespace) -> int:
    s = get_settings()
    ids = args.ids or [r.doc_id for r in read_jsonl(s.catalog_path, DocRef) if in_scope(r)]
    ids = ids[: args.limit] if args.limit else ids
    async with LexClient(s) as client:
        report = await fetch_many(
            client, RawStore(s.raw_dir), ids,
            concurrency=args.concurrency, skip_existing=not args.force,
        )  # fmt: skip
    print(
        f"fetched {len(report.fetched)}, skipped {len(report.skipped)}, failed {len(report.failed)}"
    )
    for doc_id, err in report.failed.items():
        print(f"  {doc_id}: {err}")
    return 1 if report.failed else 0


def cmd_parse(args: argparse.Namespace) -> int:
    s = get_settings()
    raw, parsed = RawStore(s.raw_dir), ParsedStore(s.parsed_dir)
    ids = args.ids or raw.ids()
    errors = 0
    for doc_id in ids:
        doc = parse_document(raw.load(doc_id), doc_id)
        issues = check_document(doc)
        parsed.save(doc)
        errors += any(i.startswith("error") for i in issues)
        flag = "  " + "; ".join(issues) if issues else ""
        print(f"{doc_id:>9}  {len(doc.articles):>4} art  {len(doc.units):>4} units  "
              f"{doc.source_script}  {doc.title[:60]}{flag}")  # fmt: skip
    print(f"parsed {len(ids)} docs, {errors} with errors")
    return 1 if errors else 0


def cmd_index(args: argparse.Namespace) -> int:
    from lexrag.index import db
    from lexrag.index.chunk import chunk_document
    from lexrag.index.embed import OpenRouterEmbedder
    from lexrag.index.indexer import document_row, index_documents, snapshot_of

    s = get_settings()
    raw, parsed = RawStore(s.raw_dir), ParsedStore(s.parsed_dir)
    catalog = {r.doc_id: r for r in read_jsonl(s.catalog_path, DocRef)}
    ids = args.ids or parsed.ids()
    ids = ids[: args.limit] if args.limit else ids

    if args.dry_run:  # chunk only: no database, no API calls
        chunks = []
        for doc_id in ids:
            doc = parsed.load(doc_id)
            row = document_row(doc, catalog.get(doc_id), snapshot_of(raw, doc_id), s)
            chunks += chunk_document(doc, row, s.embedding_model)
        chars = sum(len(c.embed_text) for c in chunks)
        split = len({(c.doc_id, c.unit_position) for c in chunks if c.pieces > 1})
        print(f"{len(ids)} docs → {len(chunks)} chunks ({split} long units split); "
              f"{chars / 1e6:.1f}M chars to embed, ~{chars / 4 / 1e6:.1f}M tokens")  # fmt: skip
        return 0

    embedder = OpenRouterEmbedder(s)
    with db.connect(s.database_url) as conn:
        db.init_schema(conn, s.embedding_dim)
        report = index_documents(
            conn, ids, parsed, raw, catalog, embedder, s,
            prune=not (args.ids or args.limit),
        )  # fmt: skip
        meta = db.read_meta(conn)
    embedder.close()
    print(f"docs: {report.docs_written} written, {report.docs_skipped} unchanged, "
          f"{report.docs_deleted} deleted; chunks: {report.chunks_written} written, "
          f"{report.chunks_embedded} embedded, {report.chunks_reused} reused")  # fmt: skip
    print(f"embedding tokens: {embedder.tokens:,}; index {meta.get('index_version')}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from lexrag.index import db
    from lexrag.rag.pipeline import RagPipeline

    s = get_settings()
    with db.connect(s.database_url) as conn:
        rag = RagPipeline(conn, s)
        r = rag.run("cli", args.question, top_k=args.top_k)
        rag.close()
    print(f"[{r.status}] language={r.language}  timing={r.timing_ms}")
    if r.error:
        print(f"error: {r.error}")
    if r.answer:
        print(f"\n{r.answer.answer}\n")
        for c in r.answer.citations:
            ctx = c.context
            print(f"  [{c.ref}] {ctx.doc_title[:60]} — {ctx.article or ctx.heading[:40]}, {c.part}")
            print(f"      «{c.quote[:160]}»")
            print(f"      {ctx.source_url}#{c.elem_id}")
        for bad in r.answer.rejected:
            print(f"  rejected: {bad}")
    print("\ncontexts:")
    for i, ctx in enumerate(r.contexts, start=1):
        rr = f" rerank={ctx.rerank_score:.3f}" if ctx.rerank_score is not None else ""
        print(f"  {i}. sim={ctx.similarity:.3f}{rr} v#{ctx.vector_rank} t#{ctx.text_rank}  "
              f"{ctx.doc_title[:50]} — {ctx.article or '-'}  {ctx.chunk_id}")  # fmt: skip
    return 0 if r.status in ("ok", "no_answer") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lexrag")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="list documents on lex.uz")
    d.add_argument("--forms", nargs="+", choices=list(_FORMS), default=list(_FORMS))
    d.add_argument("--max-pages", type=int)

    f = sub.add_parser("fetch", help="download in-scope documents")
    f.add_argument("--ids", nargs="+", type=int)
    f.add_argument("--limit", type=int)
    f.add_argument("--concurrency", type=int, default=2)
    f.add_argument("--force", action="store_true", help="re-download existing files")

    pa = sub.add_parser("parse", help="raw HTML → structured JSON")
    pa.add_argument("--ids", nargs="+", type=int)

    ix = sub.add_parser("index", help="parsed JSON → chunks + embeddings in Postgres")
    ix.add_argument("--ids", nargs="+", type=int)
    ix.add_argument("--limit", type=int)
    ix.add_argument("--dry-run", action="store_true", help="chunk and count, no DB or API")

    ask = sub.add_parser("ask", help="answer one question from the index")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=5)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1251
    if args.cmd == "discover":
        return asyncio.run(cmd_discover(args))
    if args.cmd == "fetch":
        return asyncio.run(cmd_fetch(args))
    if args.cmd == "index":
        return cmd_index(args)
    if args.cmd == "ask":
        return cmd_ask(args)
    return cmd_parse(args)


if __name__ == "__main__":
    raise SystemExit(main())
