"""Synthetic development set for tuning retrieval (Phase 4) — NOT the teacher's test.

Picks random articles from data/parsed and asks the LLM for a question a citizen would ask
that the article answers, in three languages, plus the expected answer. Questions written
from the article itself share its words, so scores on this set are optimistic; use it to
compare settings (reranker on/off, top-k), not to report quality.

    uv run python scripts/make_dev_set.py --n 40 --out eval/dev_synthetic.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from lexrag.config import get_settings
from lexrag.models import ActForm, DocRef
from lexrag.rag.llm import OpenRouterLLM
from lexrag.storage import ParsedStore, read_jsonl

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question_uz", "question_uz_cyrl", "question_ru", "expected_answer"],
    "properties": {
        "question_uz": {"type": "string"},
        "question_uz_cyrl": {"type": "string"},
        "question_ru": {"type": "string"},
        "expected_answer": {"type": "string"},
    },
}

PROMPT = """\
Below is one article of Uzbekistan's legislation. Write the question an ordinary citizen \
would ask whose answer is in this article.

- Ask about one concrete rule, right, deadline, amount or condition from the article; \
prefer a number or deadline if the article has one.
- Use everyday words. Do not copy distinctive phrases from the article and do not name the \
article number. Naming the law or code is allowed but not required.
- question_uz: Uzbek, Latin script. question_uz_cyrl: the same question in Uzbek Cyrillic. \
question_ru: the same question in Russian.
- expected_answer: the correct answer in one Uzbek sentence, taken from the article.

Law: {title}
{heading}
{text}
"""

UNANSWERABLE = [
    ("Toshkent metrosida bir martalik yoʻl haqi qancha turadi?", "uz-Latn"),
    ("Bazaviy hisoblash miqdori hozir necha soʻm?", "uz-Latn"),
    ("Qaysi bank omonatlarga eng yuqori foiz beradi?", "uz-Latn"),
    ("Сколько стоит загранпаспорт в Узбекистане в этом году?", "ru"),
    ("Ertaga Samarqandda ob-havo qanday boʻladi?", "uz-Latn"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("eval/dev_synthetic.jsonl"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    s = get_settings()
    parsed = ParsedStore(s.parsed_dir)
    catalog = {r.doc_id: r for r in read_jsonl(s.catalog_path, DocRef)}
    rng = random.Random(args.seed)

    codes = [i for i in parsed.ids() if catalog.get(i) and catalog[i].form != ActForm.LAW]
    laws = [i for i in parsed.ids() if i not in codes]
    picks: list[tuple[int, int]] = []  # (doc_id, unit position)
    while len(picks) < args.n:
        doc_id = rng.choice(codes if len(picks) % 5 < 3 else laws)  # 60 % codes
        doc = parsed.load(doc_id)
        arts = [u for u in doc.articles if 300 <= len(u.text) <= 4000 and u.number]
        if arts:
            u = rng.choice(arts)
            if (doc_id, u.position) not in picks:
                picks.append((doc_id, u.position))

    llm = OpenRouterLLM(s)
    rows = []
    for n, (doc_id, pos) in enumerate(picks, start=1):
        doc = parsed.load(doc_id)
        u = next(x for x in doc.units if x.position == pos)
        c = llm.complete_json(
            [
                {
                    "role": "user",
                    "content": PROMPT.format(title=doc.title, heading=u.heading, text=u.text),
                }
            ],
            SCHEMA,
            name="dev_question",
            temperature=0.7,
            seed=args.seed + n,
        )
        lang = "ru" if n % 4 == 0 else "uz-Cyrl" if n % 7 == 0 else "uz-Latn"
        q = {"ru": "question_ru", "uz-Cyrl": "question_uz_cyrl", "uz-Latn": "question_uz"}[lang]
        rows.append(
            {
                "id": f"d{n:03d}",
                "category": "raqam" if re.search(r"\d", c.data["expected_answer"]) else "faktik",
                "language": lang,
                "question": c.data[q],
                "expected_answer": c.data["expected_answer"],
                "expected_doc_id": doc_id,
                "expected_aliases": sorted({doc_id, *doc.twins.values()}),
                "expected_article": f"{u.number}-modda",
                "doc_title": doc.title,
                "answerable": True,
            }
        )
        print(f"{n:>3} {lang:<7} {u.number:>6}-modda  {c.data[q][:80]}")
    for k, (question, lang) in enumerate(UNANSWERABLE, start=len(rows) + 1):
        rows.append(
            {
                "id": f"d{k:03d}", "category": "javobsiz", "language": lang,
                "question": question, "expected_answer": "", "expected_doc_id": None,
                "expected_aliases": [], "expected_article": None, "doc_title": None,
                "answerable": False,
            }
        )  # fmt: skip
    llm.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
