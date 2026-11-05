"""Polite async HTTP client for lex.uz: global rate limit + retries with backoff."""

import asyncio
import logging
import time
from typing import Self

import httpx
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lexrag.config import Settings, get_settings

log = logging.getLogger(__name__)


class LexError(Exception):
    pass


class NotFoundError(LexError):
    pass


class RetryableHTTPError(LexError):
    """429 / 5xx — the server is busy, try again later."""


class RateLimiter:
    """Guarantees at least `interval` seconds between request starts, across all coroutines."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            start = max(now, self._next_at)
            if start > now:
                await asyncio.sleep(start - now)
            self._next_at = start + self._interval


class LexClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        s = settings or get_settings()
        self._settings = s
        self._limiter = RateLimiter(s.request_interval_s)
        self._client = httpx.AsyncClient(
            base_url=s.base_url,
            headers={"User-Agent": s.user_agent, "Accept-Language": "uz"},
            timeout=s.request_timeout_s,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def get_text(self, path: str) -> str:
        return (await self._request("GET", path)).text

    async def post_form(self, path: str, data: dict[str, str]) -> str:
        return (await self._request("POST", path, data=data)).text

    async def get_bytes(self, path: str) -> bytes:
        return (await self._request("GET", path)).content

    async def _request(self, method: str, path: str, **kwargs: object) -> httpx.Response:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_attempts),
            wait=wait_exponential_jitter(multiplier=self._settings.retry_initial_wait_s, max=60),
            retry=retry_if_exception_type((httpx.TransportError, RetryableHTTPError)),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._limiter.wait()
                resp = await self._client.request(method, path, **kwargs)
                self._raise_for_status(resp, path)
                return resp
        raise AssertionError("unreachable")  # reraise=True always raises on exhaustion

    @staticmethod
    def _raise_for_status(resp: httpx.Response, path: str) -> None:
        # lex.uz answers a missing document with 302 → /Pages/404.aspx (status 200)
        if resp.status_code == 404 or "404.aspx" in resp.url.path.lower():
            raise NotFoundError(path)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableHTTPError(f"{resp.status_code} for {path}")
        if resp.status_code >= 400:
            raise LexError(f"{resp.status_code} for {path}")
