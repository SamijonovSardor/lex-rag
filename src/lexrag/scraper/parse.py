"""lex.uz document page (HTML) → structured `Document`.

The body is a flat sequence of blocks, each with a stable element id:

    <div class="KIND lx_elem"> …UI buttons… <div name="ID" id="ID">content</div></div>

    TEXT_HEADER_DEFAULT   heading: qism / boʻlim / bob / paragraf
    CLAUSE_DEFAULT        article heading "18¹-modda. Title"
    ACT_TEXT, TABLE_STD   body paragraphs
    ACT_TITLE, ACT_FORM…  requisites of the act (metadata, not body text)

plus non-content blocks marked `lx_no_select`:

    CHANGES_ORIGINS   "(… Qonuni tahririda — …)" — amendment provenance of the open unit
    COMMENT           LexUZ editorial note, or a link to the previous edition (dropped)
    INDEXES_ON_REF    OKOZ classifier of the element named in its id

We rebuild the heading tree with a stack and cut the stream into units: one per article,
plus free-text units for anything outside articles (preambles, annexes).
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, date, datetime

from selectolax.lexbor import LexborHTMLParser as HTMLParser
from selectolax.lexbor import LexborNode as Node

from lexrag.models import Document, Paragraph, Unit
from lexrag.scraper.discover import parse_requisites
from lexrag.scraper.fetch import doc_url
from lexrag.text.normalize import cyrillic_ratio, normalize_text

_SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_SUP_RE = re.compile(r"<sup[^>]*>\s*(\d+)\s*</sup>", re.I)
_BR_RE = re.compile(r"<br\s*/?>", re.I)
_CELL_END_RE = re.compile(r"</t[dh]>", re.I)
_ROW_END_RE = re.compile(r"</(?:tr|p|li)>", re.I)

_ARTICLE_RE = re.compile(r"^(\d+[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*-\s*modda\b\.?\s*(.*)$", re.I | re.S)
_DATE_RE = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})")
_ONDATE_RE = re.compile(r"ONDATE2=(\d{2}\.\d{2}\.\d{4})")
_OPEN_URL_RE = re.compile(r"openUrl\('/(?:uz/)?docs/(-?\d+)'\)")
_ON_LBC_RE = re.compile(r"^onLBC(-?\d+)$")
_TWIN_TITLES = {
    "На русском": "ru",
    "In english": "en",
    "Ўзбекча": "uz_cyrl",
    "O'zbekcha": "uz_latn",
}

_HEADER_KIND = "TEXT_HEADER_DEFAULT"
_ARTICLE_KIND = "CLAUSE_DEFAULT"
_META_KINDS = {
    "ACT_TITLE", "ACT_FORM", "ACCEPTING_BODY", "ACT_ESSENTIAL_ELEMENTS",
    "ACT_ESSENTIAL_ELEMENTS_NUM", "GRIF_PARLAMENT", "SIGNATURE",
}  # fmt: skip
_CYRILLIC_DOC_THRESHOLD = 0.5

Normalizer = Callable[[str], str]


def fragment_text(node: Node) -> str:
    """Visible text of an HTML fragment, keeping the structure that matters:
    <sup>1</sup> → ¹ (article 18¹ ≠ article 181), <br>/rows → newlines, cells → " | "."""
    html = _SUP_RE.sub(lambda m: m[1].translate(_SUPERSCRIPT), node.html or "")
    html = _CELL_END_RE.sub(" | ", _ROW_END_RE.sub("\n", _BR_RE.sub("\n", html)))
    body = HTMLParser(html).body
    text = body.text(separator="") if body else ""
    return "\n".join(line.strip().removesuffix("|").strip() for line in text.splitlines())


def heading_rank(text: str) -> int:
    """Depth of a heading; a new heading closes every open heading of equal or deeper rank."""
    head = re.split(r"[.\n]", text.lower(), maxsplit=1)[0][:60]
    if re.search(r"\bqism\b", head):
        return 1
    if "boʻlim" in head:
        return 2
    if re.search(r"\bbob\b", head):
        return 3
    if "§" in head or "paragraf" in head:
        return 4
    return 5


def _join_lines(text: str) -> str:
    return ". ".join(line.rstrip(".") for line in text.splitlines() if line)


def _parse_date(text: str) -> date | None:
    m = _DATE_RE.search(text)
    return date(int(m[3]), int(m[2]), int(m[1])) if m else None


class _UnitBuilder:
    def __init__(self) -> None:
        self.units: list[Unit] = []
        self.current: Unit | None = None
        self._headings: list[tuple[int, str]] = []

    def _path(self) -> list[str]:
        return [text for _, text in self._headings]

    def heading(self, text: str) -> None:
        self.close()
        rank = heading_rank(text)
        while self._headings and self._headings[-1][0] >= rank:
            self._headings.pop()
        self._headings.append((rank, _join_lines(text)))

    def article(self, elem_id: int, text: str) -> None:
        self.close()
        text = " ".join(text.split())
        m = _ARTICLE_RE.match(text)
        number, title = (m[1], m[2].strip() or None) if m else (None, text or None)
        self.current = Unit(
            position=len(self.units),
            kind="article",
            elem_id=elem_id,
            number=number,
            title=title,
            path=self._path(),
        )

    def paragraph(self, elem_id: int, text: str) -> None:
        if self.current is None:
            self.current = Unit(
                position=len(self.units), kind="text", elem_id=elem_id, path=self._path()
            )
        self.current.paragraphs.append(Paragraph(elem_id=elem_id, text=text))

    def close(self) -> None:
        if self.current is not None:
            self.units.append(self.current)
        self.current = None


def _blocks(tree: HTMLParser) -> list[tuple[str, str, Node]]:
    """(kind, marker, node) in document order; marker is 'elem' or 'aux'."""
    out = []
    for node in tree.css("div[class]"):
        classes = (node.attributes.get("class") or "").split()
        if "lx_elem" in classes:
            out.append((classes[0], "elem", node))
        elif "lx_no_select" in classes:
            out.append((classes[0], "aux", node))
    return out


def _content(node: Node) -> tuple[int | None, Node | None]:
    inner = node.css_first("div[name]")
    if inner is None:
        return None, None
    try:
        return int(inner.attributes.get("id") or ""), inner
    except ValueError:  # "edi-5449384": link to a previous edition
        return None, inner


def _detect_cyrillic(blocks: list[tuple[str, str, Node]]) -> bool:
    sample = []
    for kind, marker, node in blocks:
        if marker == "elem" and kind not in _META_KINDS:
            sample.append(node.text())
            if sum(map(len, sample)) > 20_000:
                break
    return cyrillic_ratio(" ".join(sample)) > _CYRILLIC_DOC_THRESHOLD


def _header_metadata(tree: HTMLParser) -> tuple[str | None, date | None]:
    requisites, entry = None, None
    for item in tree.css(".docHeader__item"):
        label, value = (
            item.css_first(".docHeader__item-label"),
            item.css_first(".docHeader__item-value"),
        )
        label_text = normalize_text(label.text(), transliterate=True) if label else ""
        if value is not None and "kuchga kirish" in label_text.lower():
            entry = _parse_date(value.text())
        elif value is None and label_text and requisites is None:
            requisites = label_text
    return requisites, entry


def _twins(tree: HTMLParser, doc_id: int) -> dict[str, int]:
    twins = {}
    for node in tree.css("div.docContentHeader__item-link[title]"):
        lang = _TWIN_TITLES.get(node.attributes.get("title") or "")
        if lang is None:
            continue
        m = _OPEN_URL_RE.search(node.attributes.get("onclick") or "")
        twins[lang] = int(m[1]) if m else doc_id  # no onclick → the language we are on
    return twins


def _share_title(tree: HTMLParser) -> str | None:
    link = tree.css_first('a.dropdown-item[href*="telegram.me/share"]')
    href = (link.attributes.get("href") or "") if link else ""
    _, _, title = href.partition(" ")
    return title.strip() or None


def content_hash(doc: Document) -> str:
    h = hashlib.sha256(doc.title.encode())
    for u in doc.units:
        for part in (u.kind, u.number or "", u.title or "", " / ".join(u.path), u.text):
            h.update(b"\x1f" + part.encode())
    return h.hexdigest()


def parse_document(html: str, doc_id: int) -> Document:
    tree = HTMLParser(html)
    blocks = _blocks(tree)
    cyrillic = _detect_cyrillic(blocks)

    def norm(text: str) -> str:
        return normalize_text(text, transliterate=cyrillic)

    builder = _UnitBuilder()
    meta: dict[str, str] = {}
    topics: dict[int, list[str]] = {}

    for kind, marker, node in blocks:
        if marker == "aux":
            _handle_aux(kind, node, builder, topics, norm)
            continue
        elem_id, inner = _content(node)
        if elem_id is None or inner is None:
            continue
        text = norm(fragment_text(inner))
        if not text:
            continue
        if kind in _META_KINDS:
            meta.setdefault(kind, " ".join(text.split()))
        elif kind == _HEADER_KIND:
            builder.heading(text)
        elif kind == _ARTICLE_KIND:
            builder.article(elem_id, text)
        else:
            builder.paragraph(elem_id, text)
    builder.close()

    for unit in builder.units:
        ids = [unit.elem_id, *(p.elem_id for p in unit.paragraphs)]
        unit.topics = list(dict.fromkeys(t for i in ids for t in topics.get(i, [])))

    requisites, entry = _header_metadata(tree)
    act_type, adopted, number = parse_requisites(requisites or "")
    page_title = tree.css_first("title")
    title = _share_title(tree) or meta.get("ACT_TITLE") or (page_title.text() if page_title else "")
    doc = Document(
        doc_id=doc_id,
        url=doc_url(doc_id),
        title=norm(title),
        act_type=act_type or meta.get("ACT_FORM"),
        number=number or meta.get("ACT_ESSENTIAL_ELEMENTS_NUM", "").removesuffix("-son") or None,
        adopted=adopted,
        entry_into_force=entry,
        versions=sorted({_parse_date(d) for d in _ONDATE_RE.findall(html)} - {None}),
        twins=_twins(tree, doc_id),
        source_script="cyrl" if cyrillic else "latn",
        units=builder.units,
        parsed_at=datetime.now(UTC),
    )
    doc.content_hash = content_hash(doc)
    return doc


def _handle_aux(
    kind: str,
    node: Node,
    builder: _UnitBuilder,
    topics: dict[int, list[str]],
    norm: Normalizer,
) -> None:
    if kind == "INDEXES_ON_REF":
        target = node.css_first("div[name]")
        m = _ON_LBC_RE.match((target.attributes.get("id") or "") if target else "")
        if m:
            # "01.00.00.00 Konstitutsiyaviy tuzum / 01.01.00.00 … / 01.01.01.00 Suverenitet]"
            leaves = [
                norm(v.text()).split(" / ")[-1].rstrip("]").strip() for v in node.css(".iorVal")
            ]
            topics.setdefault(int(m[1]), []).extend(x for x in leaves if x)
        return
    if builder.current is None:
        return
    _, inner = _content(node)
    if inner is None:
        return
    text = " ".join(norm(fragment_text(inner)).split())
    if not text:
        return
    if kind == "CHANGES_ORIGINS":
        builder.current.amendments.append(text)
    elif kind == "COMMENT" and node.css_first(".COMMENTLEXUZ") is not None:
        builder.current.notes.append(text)


def check_document(doc: Document) -> list[str]:
    """Data-quality findings. The Airflow DAG fails the task on any 'error:' finding."""
    issues = []
    if not doc.units:
        issues.append("error: no units parsed")
    articles = doc.articles
    dupes = sorted(n for n, c in Counter(a.number for a in articles if a.number).items() if c > 1)
    if dupes:
        issues.append(f"warn: duplicate article numbers {dupes[:10]}")
    if articles and sum(1 for a in articles if not a.paragraphs) > len(articles) * 0.5:
        issues.append("error: most articles have no text")
    if articles and sum(1 for a in articles if a.number is None) > len(articles) * 0.2:
        issues.append("warn: many article headings without a number")
    return issues
