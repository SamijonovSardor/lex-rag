"""Chat completions through OpenRouter, answering in a fixed JSON schema.

`seed` and `temperature` are sent on every call: with the same contexts the model should
return the same answer, so runs are comparable (the teacher's spec requires it). A model
that ignores `temperature` (reasoning models) still honours `seed` on a best-effort basis.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from lexrag.config import Settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    code = "LLM_ERROR"


class LLMTimeout(LLMError):
    code = "LLM_TIMEOUT"


class _Retryable(LLMError):
    pass


@dataclass
class Completion:
    data: dict[str, Any]
    input_tokens: int
    output_tokens: int
    model: str


class OpenRouterLLM:
    def __init__(
        self,
        settings: Settings,
        *,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = settings.llm_model
        self._client = httpx.Client(
            base_url=settings.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "X-Title": "lex-rag",
            },
            timeout=timeout_s,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    @retry(
        retry=retry_if_exception_type(_Retryable),
        wait=wait_exponential_jitter(multiplier=1, max=20),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )
    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        *,
        name: str,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int = 2000,
        reasoning_effort: str | None = None,
    ) -> Completion:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }
        if seed is not None:
            body["seed"] = seed
        if reasoning_effort:  # how long a reasoning model thinks; lower = faster
            body["reasoning"] = {"effort": reasoning_effort}
        try:
            r = self._client.post("/chat/completions", json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"no answer from {self.model} in time") from e
        except httpx.TransportError as e:
            raise _Retryable(str(e)) from e
        if r.status_code == 429 or r.status_code >= 500:
            raise _Retryable(f"{r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise LLMError(f"{r.status_code}: {r.text[:500]}")
        payload = r.json()
        if "choices" not in payload:  # upstream error reported inside a 200
            raise _Retryable(f"no choices: {str(payload)[:200]}")
        content = payload["choices"][0]["message"].get("content") or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"model returned invalid JSON: {content[:200]}") from e
        usage = payload.get("usage", {})
        return Completion(
            data=data,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=payload.get("model", self.model),
        )
