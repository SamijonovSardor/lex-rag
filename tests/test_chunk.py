import itertools
from datetime import UTC, date, datetime

from conftest import JK, parsed

from lexrag.config import Settings
from lexrag.index.chunk import (
    MAX_CHARS,
    TARGET_CHARS,
    article_slug,
    chunk_document,
    number_parts,
    split_pieces,
)
from lexrag.index.indexer import document_row, edition_on
from lexrag.models import ActForm, ChunkParagraph, DocRef, DocStatus, Document, Paragraph, Unit

MODEL = "baai/bge-m3"


def doc_with(*units: Unit, twins: dict[str, int] | None = None) -> Document:
    return Document(
        doc_id=-100,
        url="https://lex.uz/uz/docs/-100",
        title="Sinov qonuni",
        twins=twins if twins is not None else {"uz_cyrl": 100, "uz_latn": -100, "ru": 101},
        units=list(units),
        content_hash="h",
        parsed_at=datetime(2026, 10, 30, tzinfo=UTC),
    )


def unit(position: int, number: str | None, *texts: str) -> Unit:
    return Unit(
        position=position,
        kind="article" if number else "text",
        elem_id=1000 + position,
        number=number,
        title="Nomi" if number else None,
        path=["1-bob. Umumiy qoidalar"],
        paragraphs=[
            Paragraph(elem_id=2000 + position * 100 + i, text=t) for i, t in enumerate(texts)
        ],
    )


def row_for(doc: Document, canonical: str = "uz_cyrl"):
    ref = DocRef(doc_id=doc.doc_id, title=doc.title, status=DocStatus.IN_FORCE, form=ActForm.LAW)
    return document_row(doc, ref, date(2026, 10, 30), Settings(canonical_id=canonical))


def test_bands_continue_the_part_new_paragraphs_open_one() -> None:
    texts = [
        "Sud quyidagi vakolatlarga ega:",
        "1) ishlarni koʻradi;",
        "2) qaror chiqaradi.",
        "Sud qarori majburiydir.",
        "a) birinchi holat;",
        "gʻ) yana bir holat.",
        "Ushbu moddaning ikkinchi qismi qoʻllanilmaydi.",
    ]
    assert number_parts(texts) == [1, 1, 1, 2, 2, 2, 3]


def test_article_slug_is_ascii() -> None:
    assert article_slug("18") == "18"
    assert article_slug("18¹") == "18.1"
    assert article_slug("223³") == "223.3"
    assert article_slug("12¹⁰") == "12.10"


def test_short_article_is_one_chunk_with_spec_style_id() -> None:
    doc = doc_with(unit(0, "18¹", "Birinchi qism.", "1) band;", "Ikkinchi qism."))
    [c] = chunk_document(doc, row_for(doc), MODEL)
    assert c.chunk_id == "lex:100#art-18.1#p1"
    assert c.article == "18¹-modda"
    assert (c.piece, c.pieces) == (1, 1)
    assert [p.qism for p in c.paragraphs] == [1, 1, 2]
    assert c.embed_text.startswith("Sinov qonuni\n1-bob. Umumiy qoidalar\n18¹-modda. Nomi\n")


def test_canonical_id_follows_settings() -> None:
    doc = doc_with(unit(0, "1", "Matn."))
    assert row_for(doc, "uz_cyrl").lex_id == 100
    assert row_for(doc, "uz_latn").lex_id == -100
    row = row_for(doc)
    assert row.aliases == [-100, 100, 101]
    assert row.url == "https://lex.uz/docs/100"
    assert row.source_url == "https://lex.uz/uz/docs/-100"
    assert (row.doc_type, row.doc_status) == ("qonun", "amalda")


def test_long_article_splits_at_paragraphs_with_overlap() -> None:
    texts = [f"{i}-qism matni. " + "soʻz " * 150 for i in range(12)]  # ~770 chars each
    doc = doc_with(unit(0, "5", *texts))
    chunks = chunk_document(doc, row_for(doc), MODEL)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [
        f"lex:100#art-5#p{i}" for i in range(1, len(chunks) + 1)
    ]
    assert all(len(c.text) <= TARGET_CHARS + 800 for c in chunks)
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.paragraphs[-1] == nxt.paragraphs[0]  # one paragraph of overlap
    covered = {p.elem_id for c in chunks for p in c.paragraphs}
    assert covered == {p.elem_id for p in doc.units[0].paragraphs}


def test_huge_single_paragraph_is_sliced() -> None:
    big = ChunkParagraph(elem_id=1, qism=1, text="Jadval qatori. " * 1000)  # 15 000 chars
    pieces = split_pieces([big])
    assert len(pieces) > 5
    assert all(sum(len(p.text) for p in piece) <= MAX_CHARS for piece in pieces)
    assert "".join(p.text for piece in pieces for p in piece).replace(" ", "") == big.text.replace(
        " ", ""
    )


def test_unnumbered_and_duplicate_articles_get_unique_ids() -> None:
    doc = doc_with(unit(0, None, "Preambula."), unit(1, "3", "A."), unit(2, "3", "B."))
    ids = [c.chunk_id for c in chunk_document(doc, row_for(doc), MODEL)]
    assert ids == ["lex:100#u0#p1", "lex:100#art-3#p1", "lex:100#u2#p1"]


def test_decree_without_articles_is_chunked_by_point() -> None:
    doc = doc_with(
        unit(0, None, "Aholi turmush darajasini oshirish maqsadida:"),
        unit(
            1, None, "1. Pensiyalar oshirilsin.", "a) 10 foizga;", "b) 15 foizga.", "2. Nafaqalar."
        ),
        unit(
            2,
            None,
            "3. Bazaviy miqdor belgilansin.",
            "1. Ilovadagi birinchi band.",
            "1.1. Kichik band.",
        ),
    )
    chunks = chunk_document(doc, row_for(doc), MODEL)
    ids = [c.chunk_id.split("#", 1)[1] for c in chunks]
    assert ids == ["u0#p1", "band-1#p1", "band-2#p1", "band-3#p1", "u2-band-1#p1"]
    assert [c.article for c in chunks] == [None, "1-band", "2-band", "3-band", "1-band"]
    band1 = chunks[1]
    assert [p.text for p in band1.paragraphs] == [
        "1. Pensiyalar oshirilsin.",
        "a) 10 foizga;",
        "b) 15 foizga.",
    ]
    assert {p.qism for p in band1.paragraphs} == {1}  # sub-items stay in the point's part
    assert [p.text for p in chunks[4].paragraphs][-1] == "1.1. Kichik band."  # "1.1." is no point
    assert band1.elem_id == band1.paragraphs[0].elem_id  # the link opens at the point


def test_points_are_not_split_out_of_articles() -> None:
    doc = doc_with(unit(0, "5", "1. Birinchi.", "2. Ikkinchi."))
    assert [c.chunk_id for c in chunk_document(doc, row_for(doc), MODEL)] == ["lex:100#art-5#p1"]


def test_empty_units_are_skipped() -> None:
    doc = doc_with(unit(0, "1", ""), unit(1, "2", "Matn."))
    assert [c.article for c in chunk_document(doc, row_for(doc), MODEL)] == ["2-modda"]


def test_embed_hash_changes_with_text_and_model() -> None:
    a = chunk_document(doc := doc_with(unit(0, "1", "Matn.")), row_for(doc), MODEL)[0]
    b = chunk_document(doc := doc_with(unit(0, "1", "Matn!")), row_for(doc), MODEL)[0]
    c = chunk_document(doc := doc_with(unit(0, "1", "Matn.")), row_for(doc), "other/model")[0]
    assert len({a.embed_hash, b.embed_hash, c.embed_hash}) == 3


def test_edition_ignores_future_editions() -> None:
    versions = [date(2020, 1, 1), date(2026, 7, 25), date(2028, 1, 1)]
    assert edition_on(versions, date(2026, 10, 30), None) == date(2026, 7, 25)
    assert edition_on([], date(2026, 10, 30), date(2019, 5, 1)) == date(2019, 5, 1)


def test_real_criminal_code_chunks() -> None:
    doc = parsed("jk", JK)
    chunks = chunk_document(doc, row_for(doc), MODEL)
    by_article = {c.article: c for c in chunks if c.piece == 1}
    assert "18¹-modda" in by_article
    assert by_article["18¹-modda"].chunk_id.startswith("lex:111453#art-18.1#")
    assert len({c.chunk_id for c in chunks}) == len(chunks)
