import difflib
import re
from datetime import date

from conftest import CONST_CYRL, CONST_LATN, JK, KSUD, parsed

from lexrag.scraper.parse import check_document, content_hash, heading_rank

UI_STRINGS = ("Hujjatga taklif yuborish", "Audioni tinglash", "Hujjat elementidan havola olish")


def test_criminal_code_structure() -> None:
    doc = parsed("jk", JK)
    assert doc.title == "Oʻzbekiston Respublikasining Jinoyat kodeksi"
    assert doc.source_script == "latn"
    assert len(doc.articles) == 406
    assert check_document(doc) == []

    first = doc.articles[0]
    assert first.number == "1"
    assert first.title == "Oʻzbekiston Respublikasining jinoyat toʻgʻrisidagi qonunchiligi"
    assert first.path == [
        "UMUMIY QISM",
        "BIRINCHI BOʻLIM. UMUMIY QOIDALAR",
        "I bob. Jinoyat kodeksining vazifalari va prinsiplari",
    ]


def test_inserted_articles_keep_superscript_numbers() -> None:
    doc = parsed("jk", JK)
    by_number = {a.number: a for a in doc.articles}
    # 18¹ was inserted later; it must not collide with article 181.
    assert by_number["18¹"].title.startswith("Aqli rasolikni istisno etmaydigan")
    assert "181" in by_number
    assert by_number["18¹"].elem_id != by_number["181"].elem_id


def test_heading_stack_closes_deeper_levels() -> None:
    doc = parsed("jk", JK)
    # Moving to the next boʻlim must drop the previous bob from the path.
    paths = {tuple(a.path[:2]) for a in doc.articles}
    assert ("UMUMIY QISM", "BIRINCHI BOʻLIM. UMUMIY QOIDALAR") in paths
    assert all(len(a.path) <= 4 for a in doc.articles)
    assert (
        heading_rank("MAXSUS QISM") < heading_rank("IKKINCHI BOʻLIM") < heading_rank("V bob. Ayb")
    )


def test_no_ui_chrome_leaks_into_text() -> None:
    for doc in (parsed("jk", JK), parsed("ksud", KSUD)):
        for unit in doc.units:
            blob = unit.text + " ".join(unit.amendments + unit.notes)
            assert not any(s in blob for s in UI_STRINGS), unit.number


def test_amendments_and_notes_are_separated_from_text() -> None:
    doc = parsed("jk", JK)
    amended = [a for a in doc.articles if a.amendments]
    assert len(amended) > 100
    assert all("tahririda" not in a.text[:40] for a in amended)
    assert any(a.notes for a in doc.articles)


def test_requisites_and_language_twins() -> None:
    doc = parsed("ksud", KSUD)
    assert doc.act_type == "Oʻzbekiston Respublikasining Konstitutsiyaviy qonuni"
    assert doc.number == "OʻRQ-1181"
    assert doc.adopted == date(2026, 10, 14)
    assert doc.entry_into_force == date(2026, 10, 15)
    assert date(2028, 1, 1) in doc.versions  # an edition scheduled for the future
    assert doc.twins == {"ru": 8543217, "en": 8543232, "uz_cyrl": 8536927, "uz_latn": -8536927}
    assert doc.url == "https://lex.uz/uz/docs/-8536927"
    assert len(doc.articles) == 120


def test_paragraph_ids_are_deep_link_anchors() -> None:
    doc = parsed("ksud", KSUD)
    art = doc.articles[0]
    assert art.elem_id == -8540364
    assert art.paragraphs[0].elem_id == -8540365


def test_constitution_preamble_is_a_text_unit() -> None:
    doc = parsed("const_lat", CONST_LATN)
    assert doc.units[0].kind == "text"
    assert doc.units[0].text.startswith("MUQADDIMA")
    assert doc.articles[0].number == "1" and doc.articles[0].title is None
    assert doc.articles[0].topics == ["01.01.01.00 Suverenitet"]


def test_transliteration_matches_official_latin_text() -> None:
    cyr, lat = parsed("const", CONST_CYRL), parsed("const_lat", CONST_LATN)
    assert cyr.source_script == "cyrl"
    assert [a.number for a in cyr.articles] == [a.number for a in lat.articles]
    assert [a.path for a in cyr.articles] == [a.path for a in lat.articles]

    def tokens(doc):
        return re.findall(r"\w+", "\n".join(u.text for u in doc.units).lower())

    a, b = tokens(cyr), tokens(lat)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    agreement = sum(m.size for m in matcher.get_matching_blocks()) / len(b)
    assert agreement > 0.999


def test_content_hash_tracks_text_only() -> None:
    doc = parsed("ksud", KSUD).model_copy(deep=True)
    original = content_hash(doc)
    assert original == doc.content_hash
    doc.parsed_at = doc.parsed_at.replace(year=2000)
    assert content_hash(doc) == original  # re-parsing the same text is not a change
    doc.articles[0].paragraphs[0].text += " (oʻzgartirildi)"
    assert content_hash(doc) != original
