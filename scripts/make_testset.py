"""Our own 100-question test set in the teacher's CSV format (spec section 6).

Same categories and counts as the teacher's set, mixed so every batch of 10 is a small
sample of the whole:

  faktik     35  one article, a concrete rule
  raqam      15  a number: deadline, amount, percent, BHM
  kop_modda  15  needs two neighbouring articles of one act
  ozgargan   10  an article amended by a later law: the rule as it reads now
  til        10  5 Russian + 5 Uzbek Cyrillic
  javobsiz   10  hand-written: not answered by the Constitution, codes and laws
  noaniq      5  colloquial or vague wording

Questions are written by the LLM from the article, told not to reuse its wording. This is
still easier than questions written by a person who never saw the article: read the scores
as an upper bound. Articles already used in eval/dev_synthetic.jsonl are excluded.

    uv run python scripts/make_testset.py --out eval/testset.csv
"""

import argparse
import csv
import itertools
import json
import random
import re
import sys
from pathlib import Path

from lexrag.config import get_settings
from lexrag.models import ActForm, DocRef, Document, Unit
from lexrag.rag.llm import OpenRouterLLM
from lexrag.storage import ParsedStore, read_jsonl

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question", "expected_answer"],
    "properties": {"question": {"type": "string"}, "expected_answer": {"type": "string"}},
}

BASE = """\
Below is legislation of Uzbekistan. Write ONE question an ordinary person would ask, and \
its correct answer.

Rules for the question: everyday words; do not copy distinctive phrases from the text; do \
not give the article number. {task}
Language of the question: {language}.
expected_answer: the correct answer in one or two Uzbek (Latin) sentences, taken only from \
the text.

{text}
"""

TASKS = {
    "faktik": "Ask about one concrete rule, right, duty or condition.",
    "raqam": "Ask for the exact number in it: a deadline, period, amount, percentage or BHM.",
    "kop_modda": "The full answer must need BOTH articles below; ask one question that "
    "combines them.",
    "ozgargan": "Ask about the rule as it is written now.",
    "noaniq": "Write it the way people talk: informal, vague, no legal terms, maybe a "
    "missing word.",
    "til": "Ask about one concrete rule, right, duty or condition.",
}
LANGUAGES = {"uz-Latn": "Uzbek, Latin script", "uz-Cyrl": "Uzbek, Cyrillic script", "ru": "Russian"}

UNANSWERABLE = [
    "Toshkent metrosida bir martalik yoʻl haqi hozir necha soʻm?",
    "Bugun dollar kursi qancha?",
    "Qaysi bank ipoteka kreditiga eng past foiz beradi?",
    "2026-yilgi Prezident saylovida kim gʻolib boʻldi?",
    "Samarqandda eng yaxshi oshxona qaysi?",
    "Chet elga ishga ketish uchun qaysi agentlik ishonchli?",
    "Ertaga Toshkentda yomgʻir yogʻadimi?",
    "Сколько стоит билет на скоростной поезд Ташкент — Самарканд?",
    "Futbol boʻyicha Oʻzbekiston terma jamoasining bosh murabbiyi kim?",
    "Telefonimdagi Click ilovasi ishlamay qoldi, nima qilay?",
]

NUMBER_RE = re.compile(
    r"\d|\b(bir|ikki|uch|toʻrt|besh|olti|yetti|sakkiz|toʻqqiz|oʻn|yigirma|oʻttiz|qirq|ellik|"
    r"yuz|ming|foiz|bazaviy hisoblash miqdori)\b",
    re.IGNORECASE,
)


def article_text(doc: Document, *units: Unit) -> str:
    parts = [f"Law: {doc.title}"]
    for u in units:
        parts.append(f"{u.heading}\n{u.text}")
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("eval/testset.csv"))
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    s = get_settings()
    parsed = ParsedStore(s.parsed_dir)
    catalog = {r.doc_id: r for r in read_jsonl(s.catalog_path, DocRef)}
    dev = Path("eval/dev_synthetic.jsonl")
    used = {
        (r["expected_doc_id"], r["expected_article"])
        for r in map(json.loads, dev.read_text(encoding="utf-8").splitlines())
        if r["answerable"]
    }
    rng = random.Random(args.seed)
    docs = {i: parsed.load(i) for i in parsed.ids()}
    codes = [i for i in docs if catalog.get(i) and catalog[i].form != ActForm.LAW]
    laws = [i for i in docs if i not in codes]

    def pick_doc() -> Document:
        return docs[rng.choice(codes if rng.random() < 0.6 else laws)]

    def usable(doc: Document, u: Unit) -> bool:
        return (
            bool(u.number)
            and 250 <= len(u.text) <= 4000
            and (doc.doc_id, f"{u.number}-modda") not in used
        )

    def pick(cond) -> tuple[Document, list[Unit]]:
        while True:
            doc = pick_doc()
            arts = [u for u in doc.articles if usable(doc, u) and cond(doc, u)]
            if arts:
                u = rng.choice(arts)
                used.add((doc.doc_id, f"{u.number}-modda"))
                return doc, [u]

    def pick_pair() -> tuple[Document, list[Unit]]:
        while True:
            doc = pick_doc()
            arts = doc.articles
            pairs = [
                (a, b) for a, b in itertools.pairwise(arts)
                if usable(doc, a) and usable(doc, b) and a.path == b.path
            ]  # fmt: skip
            if pairs:
                a, b = rng.choice(pairs)
                used.update({(doc.doc_id, f"{a.number}-modda"), (doc.doc_id, f"{b.number}-modda")})
                return doc, [a, b]

    plan: list[tuple[str, str]] = (
        [("faktik", "uz-Latn")] * 35
        + [("raqam", "uz-Latn")] * 15
        + [("kop_modda", "uz-Latn")] * 15
        + [("ozgargan", "uz-Latn")] * 10
        + [("til", "ru")] * 5
        + [("til", "uz-Cyrl")] * 5
        + [("noaniq", "uz-Latn")] * 5
    )
    llm = OpenRouterLLM(s)
    rows = []
    for n, (category, lang) in enumerate(plan, start=1):
        if category == "kop_modda":
            doc, units = pick_pair()
        elif category == "raqam":
            doc, units = pick(lambda d, u: bool(NUMBER_RE.search(u.text)))
        elif category == "ozgargan":
            doc, units = pick(lambda d, u: bool(u.amendments))
        else:
            doc, units = pick(lambda d, u: True)
        c = llm.complete_json(
            [
                {
                    "role": "user",
                    "content": BASE.format(
                        task=TASKS[category], language=LANGUAGES[lang],
                        text=article_text(doc, *units),
                    ),
                }
            ],
            SCHEMA,
            name="test_question",
            temperature=0.7,
            seed=args.seed + n,
            reasoning_effort="low",
        )  # fmt: skip
        rows.append(
            {
                "category": category,
                "question": c.data["question"].strip(),
                "expected_answer": c.data["expected_answer"].strip(),
                "expected_doc_id": f"lex:{doc.twins.get('uz_cyrl', doc.doc_id)}",
                "expected_article": ";".join(f"{u.number}-modda" for u in units),
                "answerable": "true",
                "language": lang,
            }
        )
        print(
            f"{n:>3} {category:<9} {lang:<7} {rows[-1]['expected_article']:<22} {rows[-1]['question'][:70]}"
        )
    llm.close()
    rows += [
        {
            "category": "javobsiz", "question": q, "expected_answer": "", "expected_doc_id": "",
            "expected_article": "", "answerable": "false",
            "language": "ru" if re.search("[а-я]", q) else "uz-Latn",
        }
        for q in UNANSWERABLE
    ]  # fmt: skip

    # Deal the categories round-robin into 10 batches of 10, then number q001..q100.
    rng.shuffle(rows)
    rows.sort(key=lambda r: r["category"])
    batches: list[list[dict]] = [[] for _ in range(10)]
    for i, row in enumerate(rows):
        batches[i % 10].append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "batch", "category", "question", "expected_answer", "expected_doc_id",
                    "expected_article", "answerable", "language"])  # fmt: skip
        n = 0
        for b, batch in enumerate(batches, start=1):
            rng.shuffle(batch)
            for row in batch:
                n += 1
                w.writerow([f"q{n:03d}", f"b{b:02d}", row["category"], row["question"],
                            row["expected_answer"], row["expected_doc_id"],
                            row["expected_article"], row["answerable"], row["language"]])  # fmt: skip
    print(f"wrote {n} questions in 10 batches to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
