from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from test_chunk import doc_with, unit
from test_index_db import conn, run, store  # noqa: F401  (conn is a fixture)

from lexrag.rag.retrieve import Retriever, fulltext_query, rrf

KEYWORDS = ["mehnat", "soliq", "sud", "отпуск"]


class KeywordEmbedder:
    """Vector = which topic words a text contains; 'отпуск' (Russian) shares mehnat's axis,
    the way a multilingual model puts both languages in one space."""

    model = "fake/keywords"
    dim = 1024

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            low = t.lower()
            v = np.full(self.dim, 1e-3)
            v[0] += low.count("mehnat") + low.count("отпуск")
            v[1] += low.count("soliq")
            v[2] += low.count("sud")
            out.append((v / np.linalg.norm(v)).tolist())
        return out


class ReverseReranker:
    model = "fake/rerank"

    def scores(self, query: str, documents: Sequence[str]) -> list[float]:
        return [float(i) for i in range(len(documents))]  # last candidate wins


def test_rrf_rewards_agreement() -> None:
    scores = rrf(["a", "b", "c"], ["b", "x"])
    assert max(scores, key=scores.get) == "b"  # 2nd + 1st beats 1st alone
    assert scores["a"] == pytest.approx(1 / 61)


def test_fulltext_query_drops_question_words() -> None:
    assert fulltext_query("Mehnat taʼtili qancha boʻladi?") == "mehnat or taʼtili"
    assert fulltext_query("Qancha?") == ""


@pytest.fixture
def indexed(conn, tmp_path: Path):  # noqa: F811
    labour = doc_with(
        unit(0, "1", "Mehnat taʼtili kamida 21 kun."),
        unit(1, "2", "Mehnat shartnomasi yozma tuziladi."),
    )
    tax = doc_with(unit(0, "1", "Soliq toʻlash majburiydir."))
    tax = tax.model_copy(update={"doc_id": -200, "twins": {"uz_cyrl": 200}})
    court = doc_with(unit(0, "1", "Sud mustaqildir."))
    court = court.model_copy(update={"doc_id": -300, "twins": {"uz_cyrl": 300}})
    parsed = store(tmp_path, labour, tax, court)
    run(conn, parsed, [-100, -200, -300], KeywordEmbedder())
    return conn


def test_hybrid_search_ranks_the_matching_article_first(indexed) -> None:
    r = Retriever(indexed, KeywordEmbedder()).search("Mehnat taʼtili necha kun?", "uz-Latn", 2)
    top = r.contexts[0]
    assert top.chunk_id == "lex:100#art-1#p1"  # vector ties with art-2; full text breaks it
    assert (top.vector_rank is not None, top.text_rank) == (True, 1)
    assert top.lex_id == 100 and top.article == "1-modda"
    assert top.url == "https://lex.uz/docs/100"
    assert top.paragraphs[0].qism == 1
    assert len(r.contexts) == 2 and r.retrieved_total >= 3
    assert {"embed_ms", "search_ms", "retrieval_ms"} <= set(r.timing_ms)


def test_russian_question_uses_vectors_only(indexed) -> None:
    r = Retriever(indexed, KeywordEmbedder()).search("Сколько длится отпуск?", "ru", 3)
    assert r.query == "Сколько длится отпуск?"  # not transliterated
    assert all(c.text_rank is None for c in r.contexts)
    assert r.contexts[0].doc_id == -100


def test_cyrillic_uzbek_question_is_transliterated(indexed) -> None:
    r = Retriever(indexed, KeywordEmbedder()).search("Солиқ тўлаш", "uz-Cyrl", 1)
    assert r.query == "Soliq toʻlash"
    assert r.contexts[0].chunk_id == "lex:200#art-1#p1"


def test_reranker_reorders_and_scores(indexed) -> None:
    retriever = Retriever(indexed, KeywordEmbedder(), ReverseReranker(), rerank_candidates=3)
    r = retriever.search("Mehnat taʼtili", "uz-Latn", 3)
    assert [c.rerank_score for c in r.contexts] == [2.0, 1.0, 0.0]
    assert "rerank_ms" in r.timing_ms
