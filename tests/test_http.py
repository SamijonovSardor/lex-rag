import time

import httpx
import pytest
import respx

from lexrag.config import Settings
from lexrag.scraper.fetch import EmptyDocumentError, fetch_document, fetch_many
from lexrag.scraper.http import LexClient, LexError, NotFoundError, RateLimiter
from lexrag.storage import RawStore

DOC_HTML = '<div class="ACT_TEXT lx_elem"><div name="1" id="1">matn</div></div>'


@respx.mock(base_url="https://lex.uz")
async def test_retries_server_errors(respx_mock: respx.MockRouter, settings: Settings) -> None:
    route = respx_mock.get("/uz/docs/1").mock(
        side_effect=[
            httpx.Response(503),
            httpx.ConnectError("boom"),
            httpx.Response(200, text="ok"),
        ]
    )
    async with LexClient(settings) as client:
        assert await client.get_text("/uz/docs/1") == "ok"
    assert route.call_count == 3


@respx.mock(base_url="https://lex.uz")
async def test_client_errors_are_not_retried(
    respx_mock: respx.MockRouter, settings: Settings
) -> None:
    route = respx_mock.get("/uz/docs/1").mock(return_value=httpx.Response(400))
    async with LexClient(settings) as client:
        with pytest.raises(LexError):
            await client.get_text("/uz/docs/1")
    assert route.call_count == 1


@respx.mock(base_url="https://lex.uz")
async def test_redirect_to_404_page_is_not_found(
    respx_mock: respx.MockRouter, settings: Settings
) -> None:
    respx_mock.get("/uz/docs/999").mock(
        return_value=httpx.Response(302, headers={"Location": "/Pages/404.aspx"})
    )
    respx_mock.get("/Pages/404.aspx").mock(return_value=httpx.Response(200, text="Object moved"))
    async with LexClient(settings) as client:
        with pytest.raises(NotFoundError):
            await client.get_text("/uz/docs/999")


async def test_rate_limiter_spaces_requests() -> None:
    limiter = RateLimiter(0.05)
    start = time.monotonic()
    for _ in range(4):
        await limiter.wait()
    assert time.monotonic() - start >= 0.15


@respx.mock(base_url="https://lex.uz")
async def test_fetch_many_isolates_failures(
    respx_mock: respx.MockRouter, settings: Settings
) -> None:
    respx_mock.get("/uz/docs/1").mock(return_value=httpx.Response(200, text=DOC_HTML))
    respx_mock.get("/uz/docs/2").mock(return_value=httpx.Response(404))
    respx_mock.get("/uz/docs/3").mock(return_value=httpx.Response(200, text="<html></html>"))
    store = RawStore(settings.raw_dir)
    async with LexClient(settings) as client:
        report = await fetch_many(client, store, [1, 2, 3, 1])
        with pytest.raises(EmptyDocumentError):
            await fetch_document(client, store, 3)
    assert [r.doc_id for r in report.fetched] == [1]
    assert set(report.failed) == {2, 3}
    assert store.load(1) == DOC_HTML
    assert store.ids() == [1]


@respx.mock(base_url="https://lex.uz", assert_all_called=False)
async def test_fetch_many_skips_existing(respx_mock: respx.MockRouter, settings: Settings) -> None:
    route = respx_mock.get("/uz/docs/1").mock(return_value=httpx.Response(200, text=DOC_HTML))
    store = RawStore(settings.raw_dir)
    store.save(1, DOC_HTML)
    async with LexClient(settings) as client:
        report = await fetch_many(client, store, [1], skip_existing=True)
    assert report.skipped == [1]
    assert route.call_count == 0
