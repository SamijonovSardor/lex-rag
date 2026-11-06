"""Question → top-k contexts.

  1. vector search   — the question's bge-m3 vector vs every chunk (cosine, HNSW index)
  2. full-text search — the question's words in `tsv` (any word may match, rarer words weigh more)
  3. fusion (RRF)    — score = Σ 1 / (60 + rank) over the two lists: a chunk near the top of
                       both wins, and the two scores never need to be on the same scale
  4. rerank          — optional cross-encoder over the fused top candidates

Russian questions skip full-text search: the index is Uzbek, so Russian words never match.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import psycopg

from lexrag.index.embed import Embedder
from lexrag.models import ChunkParagraph
from lexrag.rag.rerank import OpenRouterReranker, RerankError
from lexrag.text.lang import Language
from lexrag.text.normalize import normalize_query

log = logging.getLogger(__name__)
_EMBED_POOL = ThreadPoolExecutor(8, thread_name_prefix="query-embed")

RRF_K = 60

# Question words carry no topic; matching them would reward any chunk that asks a question.
_STOPWORDS = {
    "qancha", "qanday", "qaysi", "qachon", "qayerda", "nima", "nimalar", "kim", "kimlar",
    "necha", "nechta", "mi", "mumkinmi", "va", "yoki", "bilan", "uchun", "boʻyicha", "haqida",
    "bu", "shu", "u", "ham", "edi", "boʻladi", "boʻlishi", "kerak", "mumkin", "qilinadi",
    "etiladi", "qilish", "eng", "har", "bir", "oʻz", "agar",
}  # fmt: skip
_TOKEN_RE = re.compile(r"[\wʻʼ]+")


@dataclass
class Context:
    chunk_id: str
    doc_id: int  # Latin source id (internal)
    lex_id: int  # id reported by the API
    doc_title: str
    doc_type: str
    url: str
    source_url: str
    edition_date: date | None
    doc_status: str
    article: str | None
    article_title: str | None
    heading: str
    paragraphs: list[ChunkParagraph]
    amendments: list[str]
    similarity: float  # cosine similarity question ↔ chunk
    vector_rank: int | None = None
    text_rank: int | None = None
    rrf: float = 0.0
    rerank_score: float | None = None
    aliases: list[int] = field(default_factory=list)  # every lex.uz id of the act
    unit_position: int = -1
    neighbour: bool = False  # added as the article next to a top candidate

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)


@dataclass
class Retrieval:
    query: str  # what was searched (normalized)
    contexts: list[Context]
    retrieved_total: int  # distinct candidates before cutting to top_k
    timing_ms: dict[str, int] = field(default_factory=dict)


def fulltext_query(query: str) -> str:
    """'Mehnat taʼtili qancha?' → 'mehnat or taʼtili' for websearch_to_tsquery."""
    words = [w.lower() for w in _TOKEN_RE.findall(query)]
    keep = [w for w in words if len(w) > 1 and w not in _STOPWORDS]
    return " or ".join(dict.fromkeys(keep))


def rrf(*rankings: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
    return scores


_DETAILS_SQL = """
SELECT c.chunk_id, c.doc_id, d.lex_id, d.title, d.doc_type, d.url, d.source_url,
       d.edition_date, d.doc_status, c.article, c.article_title, c.heading, c.paragraphs,
       c.amendments, 1 - (c.embedding <=> %(v)s) AS similarity, d.aliases, c.unit_position
FROM chunks c JOIN documents d USING (doc_id)
WHERE c.chunk_id = ANY(%(ids)s)
"""

# The articles right before and after a chunk, same act (first piece of each).
_NEIGHBOURS_SQL = """
SELECT DISTINCT n.chunk_id
FROM chunks c JOIN chunks n
  ON n.doc_id = c.doc_id AND n.piece = 1 AND n.kind = 'article'
 AND n.unit_position IN (c.unit_position - 1, c.unit_position + 1)
WHERE c.chunk_id = ANY(%s)
"""


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection,
        embedder: Embedder,
        reranker: OpenRouterReranker | None = None,
        *,
        candidates: int = 50,
        rerank_candidates: int = 30,
        neighbours_of: int = 8,
    ) -> None:
        self.conn = conn
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_candidates = rerank_candidates
        # Rules often continue in the next article (150: general limitation period, 151:
        # special periods). The articles next to the top `neighbours_of` candidates join the
        # reranker's pool; the reranker decides whether they earn a top-k place.
        self.neighbours_of = neighbours_of
        self.use(conn)

    def use(self, conn: psycopg.Connection) -> None:
        """Search through `conn` (also after a reconnect: session settings are per connection)."""
        self.conn = conn
        # HNSW returns at most ef_search rows (default 40): raise it above `candidates`.
        conn.execute(f"SET hnsw.ef_search = {max(100, 2 * self.candidates)}")

    def search(
        self, question: str, language: Language, top_k: int = 5, *, query: str | None = None
    ) -> Retrieval:
        """`query` overrides what is searched (the Uzbek rewrite of a Russian question).
        The reranker reads the searched text too: it is far weaker across languages."""
        timing: dict[str, int] = {}
        t0 = time.perf_counter()
        q = normalize_query(query or question, language if query is None else None)
        # The embedding is a network call (~1 s); the full-text search needs no vector, so it
        # runs on the database meanwhile.
        pending = _EMBED_POOL.submit(self.embedder.embed, [q])
        fts = fulltext_query(q) if (query or language != "ru") else ""
        by_text = (
            [
                r[0]
                for r in self.conn.execute(
                    "SELECT chunk_id FROM chunks, websearch_to_tsquery('simple', %s) q "
                    "WHERE tsv @@ q ORDER BY ts_rank_cd(tsv, q) DESC LIMIT %s",
                    (fts, self.candidates),
                )
            ]
            if fts
            else []
        )
        vec = np.asarray(pending.result()[0], dtype=np.float32)
        timing["embed_ms"] = _ms(t0)

        t = time.perf_counter()
        by_vector = [
            r[0]
            for r in self.conn.execute(
                "SELECT chunk_id FROM chunks ORDER BY embedding <=> %s LIMIT %s",
                (vec, self.candidates),
            )
        ]
        fused = rrf(by_vector, by_text)
        order = sorted(fused, key=lambda k: -fused[k])
        extra: list[str] = []
        if self.reranker and self.neighbours_of:
            near = self.conn.execute(_NEIGHBOURS_SQL, (order[: self.neighbours_of],)).fetchall()
            head_ids = set(order[: self.rerank_candidates])
            extra = sorted({r[0] for r in near} - head_ids)
        details = {
            row[0]: row for row in self.conn.execute(_DETAILS_SQL, {"v": vec, "ids": order + extra})
        }
        timing["search_ms"] = _ms(t)

        v_rank = {k: i for i, k in enumerate(by_vector, start=1)}
        t_rank = {k: i for i, k in enumerate(by_text, start=1)}
        contexts = [_context(details[k], v_rank.get(k), t_rank.get(k), fused[k]) for k in order]
        neighbours = [_context(details[k], None, None, 0.0) for k in extra]
        for c in neighbours:
            c.neighbour = True

        if self.reranker and contexts:
            t = time.perf_counter()
            head = contexts[: self.rerank_candidates] + neighbours
            try:
                scores = self.reranker.scores(
                    q, [f"{c.doc_title}\n{c.heading}\n{c.text}"[:6000] for c in head]
                )
            except RerankError:  # late or failing reranker: keep the fused order
                log.warning("rerank skipped", exc_info=True)
                timing["rerank_skipped"] = 1
            else:
                for c, s in zip(head, scores, strict=True):
                    c.rerank_score = s
                contexts = sorted(head, key=lambda c: -(c.rerank_score or 0.0))
            timing["rerank_ms"] = _ms(t)

        timing["retrieval_ms"] = _ms(t0)
        return Retrieval(
            query=q,
            contexts=contexts[:top_k],
            retrieved_total=len(order) + len(extra),
            timing_ms=timing,
        )


def _context(row: tuple, v_rank: int | None, t_rank: int | None, score: float) -> Context:
    (chunk_id, doc_id, lex_id, title, doc_type, url, source_url, edition, status, article,
     article_title, heading, paragraphs, amendments, similarity, aliases,
     unit_position) = row  # fmt: skip
    return Context(
        chunk_id=chunk_id,
        doc_id=doc_id,
        lex_id=lex_id,
        doc_title=title,
        doc_type=doc_type,
        url=url,
        source_url=source_url,
        edition_date=edition,
        doc_status=status,
        article=article,
        article_title=article_title,
        heading=heading,
        paragraphs=[ChunkParagraph.model_validate(p) for p in paragraphs],
        amendments=amendments,
        similarity=float(similarity),
        vector_rank=v_rank,
        text_rank=t_rank,
        rrf=score,
        aliases=list(aliases),
        unit_position=unit_position,
    )


def _ms(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)
