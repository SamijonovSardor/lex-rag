"""Phase 3: keep the index current with lex.uz.

  daily_changes   RSS feed of new acts → (a) new substantive laws and codes, (b) every act
                  of ours that a new amending law changes: the amending law's text links
                  each act it amends (docs/recon.md), and those are fetched again;
  full_refresh    the whole listing again → catches what the feed missed, repealed acts;
  parse_changed   raw → parsed for the given ids; returns the ids whose text changed;
  index_changed   parsed → Postgres, incrementally (unchanged chunks keep their vectors).

Airflow (dags/lexrag_pipeline.py) runs these as tasks chained by data assets;
`lexrag update` runs the same steps by hand.

A known gap, closed by the weekly run: an amendment published today may enter into force
later, and lex.uz updates the consolidated text on that date, not on publication.
"""

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from selectolax.lexbor import LexborHTMLParser

from lexrag.config import Settings
from lexrag.index.indexer import IndexReport
from lexrag.models import ActForm, DocRef, DocStatus, RssItem
from lexrag.scraper.discover import discover, fetch_rss, in_scope, is_substantive
from lexrag.scraper.fetch import doc_path, fetch_document, fetch_many
from lexrag.scraper.http import LexClient
from lexrag.scraper.parse import parse_document
from lexrag.storage import ParsedStore, RawStore, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

# The feed's description starts with the act type: "Oʻzbekiston Respublikasining Qonuni
# №OʻRQ-1184. Qabul qilingan sana …". Decrees, resolutions and orders are out of scope.
_ACT_TYPE_RE = re.compile(r"\b(Qonuni|Kodeksi|Konstitutsiya\w*)\b")
_DOC_LINK_RE = re.compile(r"/docs/(-?\d+)")


def is_law(item: RssItem) -> bool:
    return bool(_ACT_TYPE_RE.search(item.description.split(".")[0]))


def linked_documents(html: str) -> set[int]:
    """lex.uz ids an act's text links to: for an amending law, the acts it amends."""
    ids = set()
    for a in LexborHTMLParser(html).css("div.lx_elem a"):
        m = _DOC_LINK_RE.search(a.attributes.get("href") or "")
        if m:
            ids.add(int(m[1]))
    return ids


def alias_index(parsed: ParsedStore) -> dict[int, int]:
    """Every lex.uz id of every act we hold (Cyrillic, Latin, ru, en) → our doc_id."""
    out: dict[int, int] = {}
    for doc_id in parsed.ids():
        out[doc_id] = doc_id
        for twin in parsed.load(doc_id).twins.values():
            out.setdefault(twin, doc_id)
    return out


@dataclass
class DailyPlan:
    new: list[int] = field(default_factory=list)  # substantive laws we do not hold yet
    amended: list[int] = field(default_factory=list)  # our acts changed by an amending law
    amending: dict[int, list[int]] = field(default_factory=dict)  # amending law → targets
    ignored: int = 0  # decrees, resolutions, orders: out of scope

    def summary(self) -> str:
        return (f"{len(self.new)} new, {len(self.amended)} amended (by {len(self.amending)} "
                f"amending laws), {self.ignored} other acts ignored")  # fmt: skip


async def plan_daily(
    client: LexClient, aliases: dict[int, int], seen: set[int]
) -> tuple[DailyPlan, list[int]]:
    """What the feed's unseen items mean for our index; also returns every id in the feed."""
    items = await fetch_rss(client)
    plan = DailyPlan()
    for item in items:
        if item.doc_id in seen:
            continue
        if not is_law(item):
            plan.ignored += 1
        elif is_substantive(item.title):
            if item.doc_id in aliases:  # republished act we hold: its text may have changed
                plan.amended.append(aliases[item.doc_id])
            else:
                plan.new.append(item.doc_id)
        else:
            html = await client.get_text(doc_path(item.doc_id))
            targets = sorted({aliases[t] for t in linked_documents(html) if t in aliases})
            plan.amending[item.doc_id] = targets
            plan.amended += [t for t in targets if t not in plan.amended]
    return plan, [i.doc_id for i in items]


async def fetch_latin(client: LexClient, raw: RawStore, doc_id: int) -> int:
    """Fetch an act; when lex.uz has its Latin twin under another id, keep that one (the
    whole corpus is Latin variants) and return the id that was kept."""
    await fetch_document(client, raw, doc_id)
    latin = parse_document(raw.load(doc_id), doc_id).twins.get("uz_latn")
    if latin and latin != doc_id:
        await fetch_document(client, raw, latin)
        raw.path(doc_id).unlink()
        return latin
    return doc_id


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _add_to_catalog(settings: Settings, refs: list[DocRef]) -> None:
    path = settings.catalog_path
    catalog = {r.doc_id: r for r in read_jsonl(path, DocRef)} if path.exists() else {}
    catalog.update({r.doc_id: r for r in refs})
    write_jsonl(path, catalog.values())


async def daily_changes(settings: Settings) -> list[int]:
    """Fetch what today's feed changed. Returns the ids whose raw HTML was downloaded."""
    raw, parsed = RawStore(settings.raw_dir), ParsedStore(settings.parsed_dir)
    state_path = settings.data_dir / "update_state.json"
    seen = set(json.loads(state_path.read_text())["seen_rss"]) if state_path.exists() else set()
    aliases = alias_index(parsed)
    fetched: list[int] = []
    async with LexClient(settings) as client:
        plan, feed_ids = await plan_daily(client, aliases, seen)
        log.info("feed: %s", plan.summary())
        new_refs = []
        for item_id in plan.new:
            try:
                doc_id = await fetch_latin(client, raw, item_id)
            except Exception as e:  # retried by the weekly full refresh
                log.warning("new act %s not fetched: %r", item_id, e)
                continue
            doc = parse_document(raw.load(doc_id), doc_id)
            form = ActForm.CODE if "kodeks" in doc.title.lower() else ActForm.LAW
            new_refs.append(DocRef(doc_id=doc_id, title=doc.title, status=DocStatus.IN_FORCE,
                                   form=form))  # fmt: skip
            fetched.append(doc_id)
        if new_refs:
            _add_to_catalog(settings, new_refs)
        report = await fetch_many(client, raw, plan.amended)
        fetched += [r.doc_id for r in report.fetched]
    _write_json(state_path, {"seen_rss": sorted(seen | set(feed_ids))[-5000:]})
    return fetched


async def full_refresh(settings: Settings) -> list[int]:
    """List every act again and download every in-scope one. Returns the in-scope ids."""
    raw = RawStore(settings.raw_dir)
    async with LexClient(settings) as client:
        refs = await discover(client, [ActForm.CONSTITUTION, ActForm.CODE, ActForm.LAW])
        write_jsonl(settings.catalog_path, refs)
        ids = [r.doc_id for r in refs if in_scope(r)]
        report = await fetch_many(client, raw, ids)
    for doc_id, err in report.failed.items():
        log.warning("fetch %s failed: %s", doc_id, err)
    return [i for i in ids if raw.exists(i)]


def parse_changed(settings: Settings, ids: list[int]) -> list[int]:
    """Parse the given raw documents; return those whose normalized text changed."""
    raw, parsed = RawStore(settings.raw_dir), ParsedStore(settings.parsed_dir)
    changed = []
    for doc_id in ids:
        doc = parse_document(raw.load(doc_id), doc_id)
        old = parsed.load(doc_id).content_hash if parsed.path(doc_id).exists() else None
        parsed.save(doc)
        if doc.content_hash != old:
            changed.append(doc_id)
    return changed


def index_changed(settings: Settings, ids: list[int], *, prune: bool) -> IndexReport:
    """Index `ids`; with `prune`, `ids` is the whole scope and other documents are removed."""
    from lexrag.index import db
    from lexrag.index.embed import OpenRouterEmbedder
    from lexrag.index.indexer import index_documents

    raw, parsed = RawStore(settings.raw_dir), ParsedStore(settings.parsed_dir)
    catalog = {r.doc_id: r for r in read_jsonl(settings.catalog_path, DocRef)}
    embedder = OpenRouterEmbedder(settings)
    try:
        with db.connect(settings.database_url) as conn:
            db.init_schema(conn, settings.embedding_dim)
            return index_documents(conn, ids, parsed, raw, catalog, embedder, settings, prune=prune)
    finally:
        embedder.close()
