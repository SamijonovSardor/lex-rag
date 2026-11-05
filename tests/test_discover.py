from datetime import date

import pytest
from conftest import fixture_text

from lexrag.models import ActForm, DocRef, DocStatus
from lexrag.scraper.discover import in_scope, is_substantive, parse_listing_page, parse_rss


def test_listing_page_rows_and_pagination_state() -> None:
    refs, has_next, state = parse_listing_page(fixture_text("laws1.html.gz"), ActForm.LAW)
    assert len(refs) == 20
    assert has_next
    assert {"__VIEWSTATE", "__EVENTTARGET", "__VIEWSTATEGENERATOR"} <= state.keys()

    first = refs[0]
    assert first.doc_id == 8548754
    assert first.status == DocStatus.IN_FORCE
    assert first.form == ActForm.LAW
    assert first.act_type == "Oʻzbekiston Respublikasining Qonuni"
    assert first.adopted == date(2026, 10, 21)
    assert first.number == "OʻRQ-1184"


def test_last_listing_page_has_no_next() -> None:
    refs, has_next, _ = parse_listing_page(fixture_text("codes2.html.gz"), ActForm.CODE)
    assert not has_next
    assert len(refs) == 8
    statuses = {r.doc_id: r.status for r in refs}
    assert statuses[-111453] == DocStatus.IN_FORCE  # Jinoyat kodeksi
    assert statuses[-142859] == DocStatus.REPEALED  # old Mehnat kodeksi


@pytest.mark.parametrize(
    ("title", "substantive"),
    [
        ("Oʻzbekiston Respublikasining Konstitutsiyaviy sudi toʻgʻrisida", True),
        ("Oʻzbekiston Respublikasining Mehnat kodeksi", True),
        ("Ayrim qonun hujjatlariga oʻzgartish va qoʻshimchalar kiritish toʻgʻrisida", False),
        ("Soliq kodeksiga o'zgartirishlar kiritish haqida", False),  # ASCII apostrophe
        ("“…”gi qarorni oʻz kuchini yoʻqotgan deb topish toʻgʻrisida", False),
        ("Shartnomani ratifikatsiya qilish toʻgʻrisida", False),
        ("Konstitutsiyasining 90-moddasiga tuzatish kiritish toʻgʻrisida", False),
        ("Yangi taraqqiyot banki toʻgʻrisidagi Bitimga qoʻshilishi haqida", False),
        ("Oʻzbekiston Respublikasining Jinoyat kodeksini tasdiqlash toʻgʻrisida", False),
        ("Qonunlar loyihalarini Qonunchilik palatasiga kiritish tartibi toʻgʻrisida", True),
    ],
)
def test_substantive_filter(title: str, substantive: bool) -> None:
    assert is_substantive(title) is substantive


def test_scope_requires_in_force() -> None:
    ref = DocRef(doc_id=1, title="Oila kodeksi", status=DocStatus.REPEALED)
    assert not in_scope(ref)
    assert in_scope(ref.model_copy(update={"status": DocStatus.IN_FORCE}))


def test_rss() -> None:
    items = parse_rss(fixture_text("rss.xml.gz").encode("utf-8"))
    assert len(items) > 10
    assert items[0].doc_id == -8563549
    assert items[0].published is not None
    assert "Vazirlar Mahkamasining qarori" in items[0].description
