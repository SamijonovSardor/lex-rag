import json
from datetime import date

import httpx
import pytest
import respx
from tenacity import wait_none

from lexrag.config import Settings
from lexrag.models import ChunkParagraph
from lexrag.rag.answer import NO_ANSWER_TEXT, build_answer, find_quote, format_contexts, renumber
from lexrag.rag.llm import Completion, LLMError, LLMTimeout, OpenRouterLLM
from lexrag.rag.retrieve import Context


def ctx(chunk_id: str, *texts: tuple[int, str], article: str = "134-modda") -> Context:
    return Context(
        chunk_id=chunk_id,
        doc_id=-1,
        lex_id=1,
        doc_title="Mehnat kodeksi",
        doc_type="kodeks",
        url="https://lex.uz/docs/1",
        source_url="https://lex.uz/uz/docs/-1",
        edition_date=date(2026, 1, 1),
        doc_status="amalda",
        article=article,
        article_title="Yillik taʼtil",
        heading=f"{article}. Yillik taʼtil",
        paragraphs=[
            ChunkParagraph(elem_id=100 + i, qism=q, text=t) for i, (q, t) in enumerate(texts)
        ],
        amendments=[],
        similarity=0.8,
        aliases=[-1, 1, 7],
    )


CONTEXTS = [
    ctx(
        "lex:1#art-134#p1",
        (1, "Xodimlarga yillik asosiy taʼtil kamida yigirma bir kalendar kun etib beriladi."),
        (2, "Taʼtil ish yili davomida berilishi kerak."),
    ),
    ctx(
        "lex:1#art-135#p1",
        (1, "Qoʻshimcha taʼtil jamoa shartnomasida belgilanadi."),
        article="135-modda",
    ),
]
COMPLETION = Completion(data={}, input_tokens=100, output_tokens=20, model="m")


def answer_of(data: dict) -> object:
    return build_answer(data, CONTEXTS, "uz-Latn", COMPLETION)


def test_valid_citation_carries_source_text_and_part() -> None:
    a = answer_of(
        {
            "no_answer": False,
            "answer": "Yillik taʼtil kamida 21 kalendar kun [1]. U ish yilida beriladi [2].",
            "citations": [
                {
                    "ref": 1,
                    "context": 1,
                    "quote": "yillik asosiy ta'til kamida yigirma bir kalendar kun",
                },
                {"ref": 2, "context": 1, "quote": "Taʼtil ish yili davomida berilishi kerak."},
            ],
            "confidence": 0.9,
        }
    )
    assert not a.no_answer
    assert [c.part for c in a.citations] == ["1-qism", "2-qism"]
    # the model wrote ' and lower case: the citation keeps the source's own characters
    assert a.citations[0].quote == "yillik asosiy taʼtil kamida yigirma bir kalendar kun"
    assert a.citations[0].elem_id == 100
    assert a.used_contexts == {"lex:1#art-134#p1"}


def test_bad_citations_are_dropped_and_markers_renumbered() -> None:
    a = answer_of(
        {
            "no_answer": False,
            "answer": "Taʼtil 30 kun [1]. Qoʻshimcha taʼtil shartnomada [2]. Boshqa [3].",
            "citations": [
                {"ref": 1, "context": 1, "quote": "taʼtil kamida oʻttiz kalendar kun"},  # invented
                {
                    "ref": 2,
                    "context": 2,
                    "quote": "Qoʻshimcha taʼtil jamoa shartnomasida belgilanadi",
                },
                {"ref": 3, "context": 9, "quote": "nothing"},  # excerpt we never sent
            ],
            "confidence": 0.5,
        }
    )
    assert a.answer == "Taʼtil 30 kun. Qoʻshimcha taʼtil shartnomada [1]. Boshqa."
    assert [(c.ref, c.context.chunk_id) for c in a.citations] == [(1, "lex:1#art-135#p1")]
    assert [r["reason"] for r in a.rejected] == ["quote not in context", "unknown context"]


def test_answer_without_any_valid_citation_becomes_no_answer() -> None:
    a = answer_of(
        {
            "no_answer": False,
            "answer": "Taʼtil 30 kun [1].",
            "citations": [{"ref": 1, "context": 1, "quote": "oʻttiz kun"}],
            "confidence": 0.9,
        }
    )
    assert a.no_answer and a.citations == []
    assert a.answer == NO_ANSWER_TEXT["uz-Latn"]


def test_model_no_answer_passes_through_without_citations() -> None:
    a = answer_of({"no_answer": True, "answer": "Javob yoʻq.", "citations": [], "confidence": 0.1})
    assert a.no_answer and a.answer == "Javob yoʻq." and a.citations == []


def test_renumber_handles_grouped_and_spaced_markers() -> None:
    assert renumber("A [3]. B [1, 3]. C [2].", {3: 1, 1: 2}) == "A [1]. B [2][1]. C."


def test_find_quote_rejects_tiny_quotes() -> None:
    assert find_quote(CONTEXTS[0], "kun") is None


def test_quote_spanning_two_paragraphs_is_found() -> None:
    quote = "yigirma bir kalendar kun etib beriladi. Taʼtil ish yili davomida"
    text, qism, elem_id = find_quote(CONTEXTS[0], quote)
    assert text == quote
    assert (qism, elem_id) == (1, 100)  # the paragraph where the quote starts
    assert find_quote(CONTEXTS[0], "beriladi. Taʼtil ish yili tugagach") is None


def test_format_contexts_numbers_excerpts_and_parts() -> None:
    text = format_contexts(CONTEXTS)
    assert text.startswith("[1] Mehnat kodeksi — 134-modda. Yillik taʼtil (edition of 2026-01-01)")
    assert "(2-qism) Taʼtil ish yili" in text
    assert "\n\n[2] Mehnat kodeksi — 135-modda" in text


# ---- the OpenRouter chat client ------------------------------------------------------

URL = "https://openrouter.ai/api/v1/chat/completions"


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OpenRouterLLM.complete_json.retry, "wait", wait_none())


def chat_reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "openai/gpt-6-luna",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 7},
        },
    )


@respx.mock
def test_llm_sends_seed_temperature_and_schema() -> None:
    route = respx.post(URL).mock(return_value=chat_reply('{"ok": true}'))
    llm = OpenRouterLLM(Settings(openrouter_api_key="k"))
    c = llm.complete_json([{"role": "user", "content": "x"}], {"type": "object"}, name="t", seed=7)
    body = json.loads(route.calls[0].request.content)
    assert (body["seed"], body["temperature"], body["model"]) == (7, 0.0, "openai/gpt-6-luna")
    assert body["response_format"]["json_schema"]["strict"] is True
    assert (c.data, c.input_tokens, c.output_tokens) == ({"ok": True}, 50, 7)


@respx.mock
def test_llm_retries_then_reports_errors() -> None:
    llm = OpenRouterLLM(Settings(openrouter_api_key="k"))
    route = respx.post(URL).mock(side_effect=[httpx.Response(503), chat_reply("{}")])
    assert llm.complete_json([], {}, name="t").data == {}
    assert route.call_count == 2

    respx.post(URL).mock(return_value=chat_reply("not json"))
    with pytest.raises(LLMError, match="invalid JSON"):
        llm.complete_json([], {}, name="t")

    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(LLMTimeout):
        llm.complete_json([], {}, name="t")


def test_quote_with_our_part_labels_is_found() -> None:
    # The model copies the "(2-qism) " labels of the excerpt format into a long quote.
    quote = "yigirma bir kalendar kun etib beriladi.\n(2-qism) Taʼtil ish yili davomida"
    text, qism, _ = find_quote(CONTEXTS[0], quote)
    assert text == "yigirma bir kalendar kun etib beriladi. Taʼtil ish yili davomida"
    assert qism == 1
