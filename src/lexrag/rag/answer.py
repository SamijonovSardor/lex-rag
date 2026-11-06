"""Contexts → an answer whose every citation is checked against the source text.

The model is asked for JSON: the answer with [n] markers, and for each marker the excerpt
it relies on plus a quote copied from it. We never trust the citations as given:
  * the excerpt number must be one we sent (the spec: no link to a source the system did
    not see);
  * the quote must occur in that excerpt's text, character for character after
    normalizing apostrophes, spaces and case; the citation then carries the source's own
    words, not the model's copy;
  * the paragraph the quote sits in gives the article part ("2-qism").
Citations that fail are dropped with their markers. An answer left with no valid
citation is not given: the item becomes no_answer.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from lexrag.rag.llm import Completion, OpenRouterLLM
from lexrag.rag.retrieve import Context
from lexrag.text.lang import Language
from lexrag.text.normalize import clean_whitespace, normalize_apostrophes

LANGUAGE_NAMES: dict[str, str] = {
    "uz-Latn": "Uzbek, Latin script",
    "uz-Cyrl": "Uzbek, Cyrillic script",
    "ru": "Russian",
}
NO_ANSWER_TEXT: dict[str, str] = {
    "uz-Latn": "Berilgan qonun hujjatlarida bu savolga javob topilmadi.",
    "uz-Cyrl": "Берилган қонун ҳужжатларида бу саволга жавоб топилмади.",
    "ru": "В представленных нормативных актах ответ на этот вопрос не найден.",
}

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["no_answer", "answer", "citations", "confidence"],
    "properties": {
        "no_answer": {"type": "boolean"},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "context", "quote"],
                "properties": {
                    "ref": {"type": "integer"},
                    "context": {"type": "integer"},
                    "quote": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "number"},
    },
}

SYSTEM_PROMPT = """\
You answer questions about the legislation of the Republic of Uzbekistan. The user gives a \
question and numbered excerpts from lex.uz. Use ONLY these excerpts.

Rules:
1. State only what the excerpts say. No outside knowledge, even if you are sure of it.
2. One sentence = one legal rule. If you need two rules, write two sentences. End each \
such sentence (after its last word, before the period) with a marker like [1]; never put a \
marker at the start of a sentence. The number is the `ref` of an item in `citations`; use \
separate markers [1][2], not [1, 2]. Two sentences never share one citation: if both \
rest on the same excerpt, give each its own citation with its own quote.
3. Each citation gives the excerpt number in `context` and, in `quote`, an exact contiguous \
span of that excerpt's text (a sentence or clause, at most 300 characters) that by itself \
states the rule of the sentence citing it: its number, deadline, body or condition must be \
inside the quote. Copy it character for character in the excerpt's language. Never \
translate, shorten or paraphrase a quote. Do not quote table fragments that lose their \
meaning; quote the row that states the value.
4. Copy numbers, amounts, deadlines and terms exactly as the excerpt writes them.
5. If the excerpts do not answer the question, set no_answer to true, say in one sentence \
that the provided legislation does not answer it, and return no citations. Do not guess.
6. The answer must be in {language}. Answer only what the question asks: no background \
rules, related provisions or other excerpts it does not ask about. Usually one or two \
sentences in plain language; at most four when the question asks for several items.
7. confidence: your probability from 0 to 1 that the answer is correct and fully supported.
"""


@dataclass
class Citation:
    ref: int
    context: Context
    quote: str  # the source's own text
    part: str  # "2-qism"
    elem_id: int  # paragraph anchor on the Latin page


@dataclass
class Answer:
    answer: str
    language: Language
    no_answer: bool
    citations: list[Citation]
    confidence: float | None
    used_contexts: set[str]  # chunk_ids cited
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)  # citations that failed


def format_contexts(contexts: list[Context]) -> str:
    blocks = []
    for n, c in enumerate(contexts, start=1):
        head = f"[{n}] {c.doc_title}"
        if c.article:
            head += f" — {c.article}" + (f". {c.article_title}" if c.article_title else "")
        elif c.heading:
            head += f" — {c.heading}"
        if c.edition_date:
            head += f" (edition of {c.edition_date.isoformat()})"
        lines = [head, *(f"({p.qism}-qism) {p.text}" for p in c.paragraphs)]
        lines += [f"Note: {a}" for a in c.amendments[:2]]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _norm(text: str) -> str:
    return clean_whitespace(normalize_apostrophes(text)).replace("\n", " ").casefold()


_STRIP = " \t\"'«»“”„.…"


def find_quote(context: Context, quote: str) -> tuple[str, int, int] | None:
    """(source text, part, elem_id) of `quote` inside the context, or None."""
    q = _norm(quote).strip(_STRIP)
    if len(q) < 8:
        return None
    norms = [_norm(p.text) for p in context.paragraphs]
    aligned = all(len(n) == len(p.text) for n, p in zip(norms, context.paragraphs, strict=True))
    for p, hay in zip(context.paragraphs, norms, strict=True):
        i = hay.find(q)
        if i >= 0:  # if normalization kept offsets, cut the source's own characters
            return (p.text[i : i + len(q)] if aligned else quote.strip(_STRIP)), p.qism, p.elem_id
    # A quote may run over several paragraphs (a rule and its list of bands, a table): the
    # run-1 test lost 4 correct answers until this search was added.
    joined = " ".join(norms)
    i = joined.find(q)
    if i < 0:
        return None
    start = 0
    for p, hay in zip(context.paragraphs, norms, strict=True):
        if i < start + len(hay) + 1:
            source = " ".join(x.text for x in context.paragraphs)[i : i + len(q)]
            return (source if aligned else quote.strip(_STRIP)), p.qism, p.elem_id
        start += len(hay) + 1
    return None


_MARKER_RE = re.compile(r"\s?\[(\d+(?:\s*,\s*\d+)*)\]")


def renumber(answer: str, keep: dict[int, int]) -> str:
    """Rewrite [old] markers to [new]; drop markers of rejected citations."""

    def sub(m: re.Match[str]) -> str:
        refs = [keep[int(x)] for x in re.findall(r"\d+", m.group(1)) if int(x) in keep]
        return (" " if m.group(0).startswith(" ") else "") + "".join(f"[{r}]" for r in refs)

    return re.sub(r"\s+([.,;:])", r"\1", _MARKER_RE.sub(sub, answer)).strip()


def build_answer(
    data: dict[str, Any], contexts: list[Context], language: Language, completion: Completion
) -> Answer:
    base = dict(
        language=language,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        model=completion.model,
    )
    confidence = data.get("confidence")
    confidence = min(1.0, max(0.0, float(confidence))) if confidence is not None else None
    if data.get("no_answer"):
        return Answer(
            answer=data.get("answer") or NO_ANSWER_TEXT[language], no_answer=True,
            citations=[], confidence=confidence, used_contexts=set(), **base,
        )  # fmt: skip

    order = [
        int(x)
        for m in _MARKER_RE.finditer(data.get("answer", ""))
        for x in re.findall(r"\d+", m.group(1))
    ]
    citations: list[Citation] = []
    rejected: list[dict[str, Any]] = []
    keep: dict[int, int] = {}
    by_ref = {}
    for item in data.get("citations", []):
        by_ref.setdefault(int(item["ref"]), item)
    # Number kept citations in the order the answer first mentions them.
    for ref in [*dict.fromkeys(order), *(r for r in by_ref if r not in order)]:
        item = by_ref.get(ref)
        if item is None:
            continue
        n = int(item["context"])
        if not 1 <= n <= len(contexts):
            rejected.append({**item, "reason": "unknown context"})
            continue
        found = find_quote(contexts[n - 1], item["quote"])
        if found is None:
            rejected.append({**item, "reason": "quote not in context"})
            continue
        text, qism, elem_id = found
        keep[ref] = len(citations) + 1
        citations.append(
            Citation(
                ref=keep[ref], context=contexts[n - 1], quote=text,
                part=f"{qism}-qism", elem_id=elem_id,
            )
        )  # fmt: skip

    if not citations:
        return Answer(
            answer=NO_ANSWER_TEXT[language], no_answer=True, citations=[],
            confidence=confidence, used_contexts=set(), rejected=rejected, **base,
        )  # fmt: skip
    return Answer(
        answer=renumber(data["answer"], keep),
        no_answer=False,
        citations=citations,
        confidence=confidence,
        used_contexts={c.context.chunk_id for c in citations},
        rejected=rejected,
        **base,
    )


def generate(
    llm: OpenRouterLLM,
    question: str,
    contexts: list[Context],
    language: Language,
    *,
    temperature: float = 0.0,
    seed: int | None = 42,
    reasoning_effort: str | None = None,
) -> Answer:
    if not contexts:
        return Answer(
            answer=NO_ANSWER_TEXT[language], language=language, no_answer=True,
            citations=[], confidence=None, used_contexts=set(),
        )  # fmt: skip
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[language])},
        {
            "role": "user",
            "content": f"Question: {question}\n\nExcerpts:\n\n{format_contexts(contexts)}",
        },
    ]
    completion = llm.complete_json(
        messages,
        ANSWER_SCHEMA,
        name="legal_answer",
        temperature=temperature,
        seed=seed,
        reasoning_effort=reasoning_effort,
    )
    answer = build_answer(completion.data, contexts, language, completion)
    if language == "uz-Latn":  # the model mixes ’ ‘ ' : use the official oʻ gʻ and ʼ like the index
        answer.answer = normalize_apostrophes(answer.answer)
    return answer
