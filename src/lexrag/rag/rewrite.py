"""Russian question → Uzbek search query.

The index is Uzbek. bge-m3 can match a Russian question to an Uzbek article on its own, but
full-text search cannot (no shared words) and the reranker loses most of its accuracy across
languages: on the dev set, Hit@5 (article) for Russian questions fell from 60 % to 20 % once
the reranker was on. Translating the question first puts all three stages back on Uzbek
text. The rewrite is what the spec calls `debug.rewritten_query`.
"""

from lexrag.rag.llm import OpenRouterLLM

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {"query": {"type": "string"}},
}

PROMPT = """\
Translate this question about the law of Uzbekistan into Uzbek (Latin script), as a search \
query for Uzbek legislation. Use the terms Uzbek laws use (for example "da'vo muddati" for \
"исковая давность", "mehnat ta'tili" for "трудовой отпуск", "Fuqarolik kodeksi" for \
"Гражданский кодекс"). Keep every number, amount and name. Return only the translation.

Question: {question}"""


def to_uzbek_query(
    llm: OpenRouterLLM, question: str, seed: int | None = 42, effort: str | None = "minimal"
) -> tuple[str, int, int]:
    """(query, input tokens, output tokens)."""
    c = llm.complete_json(
        [{"role": "user", "content": PROMPT.format(question=question)}],
        SCHEMA,
        name="search_query",
        temperature=0.0,
        seed=seed,
        max_tokens=300,
        reasoning_effort=effort,
    )
    return c.data["query"].strip() or question, c.input_tokens, c.output_tokens
