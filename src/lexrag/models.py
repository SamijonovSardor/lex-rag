"""Data contracts shared by the scraper, the Airflow DAGs and (later) the indexer.

Medallion layout:
  raw/     bronze — HTML exactly as lex.uz served it (gzip)
  parsed/  silver — `Document` JSON: clean Latin text, structured into units
  Postgres gold   — chunks + embeddings (Phase 2)
"""

from datetime import date, datetime
from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class DocStatus(StrEnum):
    IN_FORCE = "in_force"  # lex.uz icon class status_code_y
    REPEALED = "repealed"  # status_code_r
    UNKNOWN = "unknown"


class ActForm(IntEnum):
    """`form_id` values of lex.uz national-database search."""

    CONSTITUTION = 4131
    CODE = 3964
    LAW = 3968
    DECREE = 3973  # farmon
    DIRECTIVE = 588  # farmoyish
    RESOLUTION = 3972  # qaror (President, Cabinet and other bodies)
    ORDER = 486  # buyruq (ministries)


class DocRef(BaseModel):
    """One row of a lex.uz listing: enough to decide whether to fetch the document."""

    doc_id: int
    title: str
    status: DocStatus
    form: ActForm | None = None
    # "Oʻzbekiston Respublikasining Qonuni, 21.10.2026 yildagi OʻRQ-1184-son"
    requisites: str | None = None
    act_type: str | None = None  # "Oʻzbekiston Respublikasining Qonuni"
    adopted: date | None = None
    number: str | None = None  # "OʻRQ-1184"


class RssItem(BaseModel):
    doc_id: int
    title: str
    description: str
    published: datetime | None = None


class Paragraph(BaseModel):
    elem_id: int  # lex.uz DOM id → deep link https://lex.uz/uz/docs/{doc_id}#{elem_id}
    text: str


class Unit(BaseModel):
    """Smallest citable piece of a document: an article (modda) or a free text block
    (preamble, annex, text placed directly under a heading)."""

    position: int
    kind: Literal["article", "text"]
    elem_id: int
    number: str | None = None  # "18¹" — superscripts kept, they are part of the number
    title: str | None = None
    path: list[str] = Field(default_factory=list)  # heading breadcrumbs, outermost first
    paragraphs: list[Paragraph] = Field(default_factory=list)
    amendments: list[str] = Field(default_factory=list)  # "(… Qonuni tahririda — …)"
    notes: list[str] = Field(default_factory=list)  # LexUZ editorial cross-references
    topics: list[str] = Field(default_factory=list)  # OKOZ legal classifier

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    @property
    def heading(self) -> str:
        if self.kind == "article":
            head = f"{self.number}-modda" if self.number else "Modda"
            return f"{head}. {self.title}" if self.title else head
        return self.path[-1] if self.path else ""


class Document(BaseModel):
    doc_id: int
    url: str
    title: str
    act_type: str | None = None
    number: str | None = None
    adopted: date | None = None
    entry_into_force: date | None = None
    versions: list[date] = Field(default_factory=list)  # edition dates lex.uz offers
    twins: dict[str, int] = Field(default_factory=dict)  # same act in ru / en / uz_cyrl / uz_latn
    source_script: Literal["latn", "cyrl"] = "latn"  # cyrl → text was transliterated by us
    units: list[Unit] = Field(default_factory=list)
    content_hash: str = ""  # sha256 of the normalized text: drives incremental re-indexing
    parsed_at: datetime

    @property
    def articles(self) -> list[Unit]:
        return [u for u in self.units if u.kind == "article"]


# ---- Phase 2: the index (gold layer) -------------------------------------------------


class IndexedDocument(BaseModel):
    """One row of `documents`: what the API reports about a source act."""

    doc_id: int  # the Latin variant we parsed; key into data/parsed
    lex_id: int  # the id the API reports (settings.canonical_id)
    aliases: list[int]  # every lex.uz id of this act (uz_cyrl, uz_latn, ru, en)
    title: str
    doc_type: str  # "konstitutsiya" | "kodeks" | "qonun"
    number: str | None = None
    adopted: date | None = None
    edition_date: date | None = None  # the edition our text is, as of snapshot_date
    doc_status: str  # "amalda" | "kuchini yo'qotgan"
    url: str  # https://lex.uz/docs/{lex_id}
    source_url: str  # the Latin page; paragraph anchors (#elem_id) only work there
    content_hash: str
    snapshot_date: date  # day the HTML was downloaded from lex.uz


class ChunkParagraph(BaseModel):
    elem_id: int
    qism: int  # "N-qism" of the article: a new paragraph opens one, a list band "1)" does not
    text: str


class Chunk(BaseModel):
    """The unit of retrieval: one article, or one piece of a long article."""

    chunk_id: str  # "lex:{lex_id}#art-{number}#p{piece}" — stable across rebuilds
    doc_id: int
    unit_position: int
    elem_id: int
    kind: Literal["article", "text"]
    article: str | None  # "18¹-modda"; None for preambles and text outside articles
    article_title: str | None
    piece: int  # 1-based; >1 only for articles longer than the chunker's limit
    pieces: int
    heading: str
    path: list[str]
    paragraphs: list[ChunkParagraph]
    amendments: list[str]
    embed_text: str  # title + path + heading + text: what the embedding model sees
    embed_hash: str  # sha256(model + embed_text): unchanged hash = reuse the stored vector

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)
