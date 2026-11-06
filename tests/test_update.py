import json
from pathlib import Path

import httpx
import respx
from conftest import JK, fixture_text

from lexrag.config import Settings
from lexrag.models import RssItem
from lexrag.scraper.http import LexClient
from lexrag.storage import ParsedStore, RawStore
from lexrag.update import daily_changes, is_law, linked_documents, parse_changed, plan_daily

AMENDING = -8547667  # "…ayrim qonun hujjatlariga … oʻzgartirishlar kiritish toʻgʻrisida"


def item(desc: str) -> RssItem:
    return RssItem(doc_id=1, title="t", description=desc)


def test_only_laws_codes_and_the_constitution_count() -> None:
    assert is_law(
        item("Oʻzbekiston Respublikasining Qonuni №OʻRQ-1184. Qabul qilingan sana 21.10.2026.")
    )
    assert is_law(item("Oʻzbekiston Respublikasining Kodeksi. Qabul qilingan sana 01.01.2026."))
    assert not is_law(item("Oʻzbekiston Respublikasi Vazirlar Mahkamasining qarori №596. Qabul …"))
    assert not is_law(item("Oʻzbekiston Respublikasi Prezidentining Farmoni №PF-236. Qabul …"))


def test_an_amending_law_links_the_acts_it_amends() -> None:
    assert linked_documents(fixture_text("amending.html.gz")) == {-3523891, -4674902}


def rss(*items: tuple[int, str, str]) -> str:
    body = "".join(
        f"<item><title>{t}</title><link>https://lex.uz/uz/docs/{i}</link>"
        f"<description>{d}</description><pubDate>Thu, 29 Oct 2026 20:05:59 GMT</pubDate></item>"
        for i, t, d in items
    )
    return f'<?xml version="1.0"?><rss><channel>{body}</channel></rss>'


FEED = rss(
    (-1, "Yoʻl harakati qoidalari", "Vazirlar Mahkamasining qarori №596. Qabul …"),
    (-2, "Yangi qonun toʻgʻrisida", "Oʻzbekiston Respublikasining Qonuni №OʻRQ-1200. Qabul …"),
    (AMENDING, "Ayrim qonun hujjatlariga oʻzgartirishlar kiritish toʻgʻrisida",
     "Oʻzbekiston Respublikasining Qonuni №OʻRQ-1150. Qabul …"),
    (-3, "Allaqachon koʻrilgan qonun toʻgʻrisida", "Oʻzbekiston Respublikasining Qonuni №1. …"),
)  # fmt: skip


@respx.mock(base_url="https://lex.uz")
async def test_plan_from_the_feed(respx_mock: respx.MockRouter, settings: Settings) -> None:
    respx_mock.get("/uz/rss").mock(return_value=httpx.Response(200, text=FEED))
    respx_mock.get(f"/uz/docs/{AMENDING}").mock(
        return_value=httpx.Response(200, text=fixture_text("amending.html.gz"))
    )
    # we hold the Administrative Code (Latin -3523891) and the Tax Code under its Cyrillic id
    aliases = {-3523891: -3523891, 4674902: -4674902, -4674902: -4674902}
    async with LexClient(settings) as client:
        plan, feed = await plan_daily(client, aliases, seen={-3})
    assert plan.new == [-2]
    assert plan.amended == [-4674902, -3523891]
    assert plan.amending == {AMENDING: [-4674902, -3523891]}
    assert plan.ignored == 1
    assert feed == [-1, -2, AMENDING, -3]


@respx.mock(base_url="https://lex.uz")
async def test_daily_remembers_the_feed(respx_mock: respx.MockRouter, settings: Settings) -> None:
    only_decrees = rss((-1, "Qaror", "Vazirlar Mahkamasining qarori №1. …"))
    respx_mock.get("/uz/rss").mock(return_value=httpx.Response(200, text=only_decrees))
    assert await daily_changes(settings) == []
    state = json.loads((settings.data_dir / "update_state.json").read_text())
    assert state == {"seen_rss": [-1]}


def test_parse_reports_only_changed_text(settings: Settings) -> None:
    raw = RawStore(settings.raw_dir)
    raw.save(JK, fixture_text("jk.html.gz"))
    assert parse_changed(settings, [JK]) == [JK]  # first time: new
    assert parse_changed(settings, [JK]) == []  # same HTML again: nothing to index
    assert ParsedStore(settings.parsed_dir).path(JK).exists()
    assert Path(settings.parsed_dir).is_dir()
