"""Download document pages into the raw store."""

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from lexrag.scraper.http import LexClient, LexError
from lexrag.storage import RawRecord, RawStore

log = logging.getLogger(__name__)


class EmptyDocumentError(LexError):
    """Page loaded but carries no document body (placeholder, moved or access-limited act)."""


def doc_path(doc_id: int) -> str:
    return f"/uz/docs/{doc_id}"


def doc_url(doc_id: int, elem_id: int | None = None, base: str = "https://lex.uz") -> str:
    url = f"{base}{doc_path(doc_id)}"
    return f"{url}#{elem_id}" if elem_id is not None else url


async def fetch_document(client: LexClient, store: RawStore, doc_id: int) -> RawRecord:
    html = await client.get_text(doc_path(doc_id))
    if "lx_elem" not in html:
        raise EmptyDocumentError(f"doc {doc_id}: no document body in page")
    return store.save(doc_id, html)


@dataclass
class FetchReport:
    fetched: list[RawRecord] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)


async def fetch_many(
    client: LexClient,
    store: RawStore,
    doc_ids: Iterable[int],
    *,
    concurrency: int = 2,
    skip_existing: bool = False,
) -> FetchReport:
    """Fetch documents; one failure never aborts the batch — it is recorded instead.

    The client's rate limiter caps the request rate; `concurrency` only lets a slow
    download overlap with the next request.
    """
    report = FetchReport()
    sem = asyncio.Semaphore(concurrency)

    async def one(doc_id: int) -> None:
        if skip_existing and store.exists(doc_id):
            report.skipped.append(doc_id)
            return
        async with sem:
            try:
                rec = await fetch_document(client, store, doc_id)
            except Exception as exc:
                log.warning("fetch %s failed: %r", doc_id, exc)
                report.failed[doc_id] = repr(exc)
            else:
                report.fetched.append(rec)
                log.info("fetched %s (%d KB)", doc_id, rec.size // 1024)

    await asyncio.gather(*(one(i) for i in dict.fromkeys(doc_ids)))
    return report
