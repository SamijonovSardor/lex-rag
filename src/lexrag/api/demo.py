"""The public web page: GET / (one static HTML file) and POST /demo/ask.

The page needs no API key, so it is guarded three ways, all configurable:
  * per client IP: `demo_per_ip_per_hour` questions per sliding hour;
  * whole site: `demo_per_day` questions per UTC day;
  * budget: below `demo_min_budget_usd` left on the OpenRouter key the demo pauses, so a
    crowd of visitors can never spend the reserve the test API (/v1/...) runs on.
Counters live in memory: a restart resets them, which only ever errs towards fewer answers
in total than (restarts + 1) × the daily cap.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from functools import cache
from importlib.resources import files

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from lexrag.api.schemas import Options, to_result
from lexrag.config import Settings

log = logging.getLogger("lexrag.api.demo")
router = APIRouter()


class DemoQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DemoGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._per_ip: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._today = 0
        self._budget: tuple[float, float | None] = (0.0, None)  # (checked at, remaining $)

    def budget_left(self) -> float | None:
        """Remaining OpenRouter key balance in $, cached for 5 minutes; None = unknown or
        no limit on the key (then only the counters guard the spend)."""
        checked, value = self._budget
        if time.monotonic() - checked < 300:
            return value
        try:
            r = httpx.get(
                f"{self.settings.openrouter_base_url}/key",
                headers={"Authorization": f"Bearer {self.settings.openrouter_api_key}"},
                timeout=5.0,
            )
            value = r.json()["data"].get("limit_remaining")
            value = float(value) if value is not None else None
        except Exception:  # fail open: the daily cap still bounds the spend
            log.warning("could not read the OpenRouter balance", exc_info=True)
            value = None
        self._budget = (time.monotonic(), value)
        return value

    def admit(self, ip: str) -> tuple[str, str] | None:
        """None if the question may run, else (code, message for the page)."""
        s = self.settings
        if not s.demo_enabled:
            return "DEMO_OFF", "Demo oʻchirilgan."
        left = self.budget_left()
        if left is not None and left < s.demo_min_budget_usd:
            return "DEMO_PAUSED", "Demo vaqtincha toʻxtatilgan: soʻrovlar uchun budjet tugadi."
        now, day = time.monotonic(), datetime.now(UTC).date().isoformat()
        with self._lock:
            if day != self._day:
                self._day, self._today = day, 0
            if self._today >= s.demo_per_day:
                return (
                    "DEMO_DAILY_LIMIT",
                    "Bugungi savollar limiti tugadi. Ertaga qayta urinib koʻring.",
                )
            hits = self._per_ip[ip]
            while hits and now - hits[0] > 3600:
                hits.popleft()
            if len(hits) >= s.demo_per_ip_per_hour:
                wait = int(3600 - (now - hits[0])) // 60 + 1
                return (
                    "DEMO_RATE_LIMIT",
                    f"Juda koʻp savol. {wait} daqiqadan keyin qayta urinib koʻring.",
                )
            hits.append(now)
            self._today += 1
        return None


def client_ip(request: Request) -> str:
    # Render's proxy appends the real client to X-Forwarded-For; earlier entries can be forged.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


@cache
def page() -> str:
    return files("lexrag.api").joinpath("static/index.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    """Static HTML: no database, no key — also the keep-alive target."""
    return HTMLResponse(page())


@router.post("/demo/ask")
def ask(body: DemoQuestion, request: Request):
    from lexrag.api.app import ApiError, Utf8JSON  # app imports this module

    runtime = request.app.state.rt
    question = body.question.strip()
    if not question:
        raise ApiError(400, "VALIDATION_ERROR", "Savol boʻsh.")
    if len(question) > runtime.settings.demo_max_chars:
        limit = runtime.settings.demo_max_chars
        raise ApiError(400, "TOO_LONG", f"Savol {limit} belgidan oshmasin.")
    refused = runtime.demo.admit(client_ip(request))
    if refused:
        code, message = refused
        raise ApiError(503 if code in ("DEMO_PAUSED", "DEMO_OFF") else 429, code, message)
    fut = runtime.items.submit(lambda: runtime.pipeline().run("demo", question, top_k=5))
    result = to_result(fut.result(), Options(return_contexts=True, return_debug=False))
    return Utf8JSON(content=result.model_dump(mode="json"))
