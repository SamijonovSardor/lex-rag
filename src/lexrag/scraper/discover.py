"""Find which documents exist on lex.uz.

Two sources:
- search listings (`/uz/search/nat`) — full inventory of constitution / codes / laws,
  20 rows per page, paginated through ASP.NET WebForms postbacks;
- RSS (`/uz/rss`) — newest acts, used by the daily incremental DAG.
"""

import logging
import re
from collections.abc import AsyncIterator
from datetime import date, datetime
from email.utils import parsedate_to_datetime

from defusedxml import ElementTree
from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from lexrag.models import ActForm, DocRef, DocStatus, RssItem
from lexrag.scraper.http import LexClient
from lexrag.text.normalize import normalize_text

log = logging.getLogger(__name__)

LISTING_PATH = "/uz/search/nat"
SORT_LEGISLATION = 3975  # "Qonun hujjatlari" group of the national database
LANG_UZ_LATIN = 4
NEXT_PAGE_TARGET = "ucFoundActsControl$LinkButton1"  # the "Keyingisi" postback

_DOC_HREF_RE = re.compile(r"^/uz/docs/(-?\d+)$")
_REQUISITES_RE = re.compile(
    r"^(?P<act_type>.+?),\s*(?P<date>\d{2}\.\d{2}\.\d{4})\s*yildagi\s*(?P<number>.+?)(?:-son)?$"
)
# Icon of the row: fa-check-circle (y) — in force; fa-times-circle (r, n) — no longer in force.
_STATUS = {
    "status_code_y": DocStatus.IN_FORCE,
    "status_code_r": DocStatus.REPEALED,
    "status_code_n": DocStatus.REPEALED,
    "fa-check-circle": DocStatus.IN_FORCE,
    "fa-times-circle": DocStatus.REPEALED,
}


def listing_path(form: ActForm) -> str:
    return f"{LISTING_PATH}?lang={LANG_UZ_LATIN}&sort_id={SORT_LEGISLATION}&form_id={form.value}"


def parse_requisites(text: str) -> tuple[str | None, date | None, str | None]:
    """ "Oʻzbekiston Respublikasining Qonuni, 21.10.2026 yildagi OʻRQ-1184-son"
    → ("Oʻzbekiston Respublikasining Qonuni", 2026-10-21, "OʻRQ-1184")"""
    m = _REQUISITES_RE.match(text)
    if not m:
        return None, None, None
    return m["act_type"], datetime.strptime(m["date"], "%d.%m.%Y").date(), m["number"]


def _row_status(row: Node) -> DocStatus:
    icon = row.css_first("span.lx_act_state i")
    classes = (icon.attributes.get("class") or "").split() if icon else []
    # status_code_* is the precise signal; the icon shape is the fallback for new codes
    for cls in sorted(classes, key=lambda c: not c.startswith("status_code_")):
        if cls in _STATUS:
            return _STATUS[cls]
    return DocStatus.UNKNOWN


def parse_listing_page(
    html: str, form: ActForm | None = None
) -> tuple[list[DocRef], bool, dict[str, str]]:
    """Returns (rows, has_next_page, hidden form state needed for the next postback)."""
    tree = HTMLParser(html)
    refs: list[DocRef] = []
    for row in tree.css("tr.dd-table__main-item"):
        link = next(
            (a for a in row.css("a.lx_link") if _DOC_HREF_RE.match(a.attributes.get("href") or "")),
            None,
        )
        if link is None:
            continue
        badge = row.css_first("span.badge-nine")
        requisites = normalize_text(badge.text()) if badge else None
        act_type, adopted, number = parse_requisites(requisites or "")
        refs.append(
            DocRef(
                doc_id=int(_DOC_HREF_RE.match(link.attributes["href"])[1]),
                title=normalize_text(link.text()),
                status=_row_status(row),
                form=form,
                requisites=requisites,
                act_type=act_type,
                adopted=adopted,
                number=number,
            )
        )
    has_next = tree.css_first("a#ucFoundActsControl_LinkButton1[href]") is not None
    state = {
        name: inp.attributes.get("value") or ""
        for inp in tree.css("input[type=hidden]")
        if (name := inp.attributes.get("name") or "").startswith("__")
    }
    return refs, has_next, state


async def iter_listing(
    client: LexClient, form: ActForm, *, max_pages: int | None = None
) -> AsyncIterator[DocRef]:
    path = listing_path(form)
    html = await client.get_text(path)
    page, seen = 1, set[int]()
    while True:
        refs, has_next, state = parse_listing_page(html, form)
        fresh = [r for r in refs if r.doc_id not in seen]
        if refs and not fresh:
            # Postback state got lost and the server re-sent a page we already have.
            log.warning("listing %s: page %d repeats previous rows, stopping", form.name, page)
            return
        for ref in fresh:
            seen.add(ref.doc_id)
            yield ref
        log.info("listing %s: page %d → %d rows", form.name, page, len(fresh))
        if not has_next or (max_pages is not None and page >= max_pages):
            return
        html = await client.post_form(
            path, {**state, "__EVENTTARGET": NEXT_PAGE_TARGET, "__EVENTARGUMENT": ""}
        )
        page += 1


async def discover(
    client: LexClient, forms: list[ActForm], *, max_pages: int | None = None
) -> list[DocRef]:
    refs: list[DocRef] = []
    for form in forms:
        refs.extend([r async for r in iter_listing(client, form, max_pages=max_pages)])
    return refs


# --- scope -------------------------------------------------------------------------------

# Acts whose only purpose is to change / repeal / ratify / approve other acts. Their content
# already lives in the consolidated text of the act they target, so indexing them only adds
# near-duplicate noise to retrieval.
_NON_SUBSTANTIVE_RE = re.compile(
    r"(oʻzgartish|oʻzgartirish|qoʻshimcha|tuzatish)\w*\b.{0,80}?\bkiritish"
    r"|kuchini yoʻqotgan deb topish"
    r"|ratifikatsiya qilish"
    r"|\bqoʻshilish\w*\s+(toʻgʻrisida|haqida)"  # accession to an international treaty
    r"|kodeksini tasdiqlash",  # the code itself is a separate document
    re.IGNORECASE,
)


def is_substantive(title: str) -> bool:
    return not _NON_SUBSTANTIVE_RE.search(normalize_text(title))


def in_scope(ref: DocRef) -> bool:
    return ref.status == DocStatus.IN_FORCE and is_substantive(ref.title)


# --- RSS ---------------------------------------------------------------------------------


def parse_rss(xml: bytes) -> list[RssItem]:
    root = ElementTree.fromstring(xml)
    items: list[RssItem] = []
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        m = re.search(r"/docs/(-?\d+)", link)
        if not m:
            continue
        pub = item.findtext("pubDate")
        items.append(
            RssItem(
                doc_id=int(m[1]),
                title=normalize_text(item.findtext("title") or ""),
                description=normalize_text(item.findtext("description") or ""),
                published=parsedate_to_datetime(pub) if pub else None,
            )
        )
    return items


async def fetch_rss(client: LexClient) -> list[RssItem]:
    return parse_rss(await client.get_bytes("/uz/rss"))
