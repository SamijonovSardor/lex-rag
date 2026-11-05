import json

import httpx
import pytest
import respx
from tenacity import wait_none

from lexrag.config import Settings
from lexrag.index.embed import EmbeddingError, OpenRouterEmbedder

URL = "https://openrouter.ai/api/v1/embeddings"


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenRouterEmbedder._embed_batch.retry, "wait", wait_none())


@pytest.fixture
def embedder() -> OpenRouterEmbedder:
    return OpenRouterEmbedder(
        Settings(openrouter_api_key="test-key", embedding_dim=3, embed_batch_size=2)
    )


def reply(request: httpx.Request) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [{"index": i, "embedding": [float(len(t)), 0.0, 1.0]} for i, t in enumerate(texts)]
    return httpx.Response(
        200, json={"data": data[::-1], "usage": {"prompt_tokens": 5 * len(texts)}}
    )


@respx.mock
def test_batches_and_keeps_input_order(embedder: OpenRouterEmbedder) -> None:
    route = respx.post(URL).mock(side_effect=reply)
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]  # response rows came back reversed
    assert route.call_count == 2  # batch size 2
    sent = route.calls[0].request
    assert sent.headers["Authorization"] == "Bearer test-key"
    assert json.loads(sent.content)["model"] == "baai/bge-m3"
    assert embedder.tokens == 15


@respx.mock
def test_retries_rate_limits_and_server_errors(embedder: OpenRouterEmbedder) -> None:
    route = respx.post(URL).mock(side_effect=[httpx.Response(429), httpx.Response(502), reply])
    assert len(embedder.embed(["a"])) == 1
    assert route.call_count == 3


@respx.mock
def test_client_errors_fail_fast(embedder: OpenRouterEmbedder) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(400, json={"error": "bad"}))
    with pytest.raises(EmbeddingError, match="400"):
        embedder.embed(["a"])
    assert route.call_count == 1


@respx.mock
def test_wrong_dimension_is_an_error(embedder: OpenRouterEmbedder) -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    with pytest.raises(EmbeddingError, match="3-dim"):
        embedder.embed(["a"])


def test_missing_key_is_reported() -> None:
    with pytest.raises(EmbeddingError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbedder(Settings(openrouter_api_key=""))
