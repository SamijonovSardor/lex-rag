"""Score a saved run against the test set with the metrics of spec section 6.

Deterministic: Hit@5 (document, article), MRR, fabricated sources, stale norms, no-answer
accuracy, latency, error rate.
LLM judge (one call per answered question): answer correctness 0/1/2, faithfulness (share of
claims supported by the contexts), citation accuracy, context precision.
The judge is gpt-6-luna, the same model that answers: it may be lenient with its own
answers. Treat judge scores as indicative.

    uv run python scripts/score_run.py eval/testset.csv data/runs/eval-2026-11-06
"""

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lexrag.config import get_settings
from lexrag.rag.llm import OpenRouterLLM

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["correctness", "claims_total", "claims_supported", "citations", "relevant"],
    "properties": {
        "correctness": {"type": "integer"},
        "claims_total": {"type": "integer"},
        "claims_supported": {"type": "integer"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ref", "supports"],
                "properties": {"ref": {"type": "integer"}, "supports": {"type": "boolean"}},
            },
        },
        "relevant": {"type": "array", "items": {"type": "boolean"}},
    },
}

JUDGE_PROMPT = """\
You grade an answer of a legal question-answering system about Uzbekistan's legislation.

Question: {question}
Reference answer (correct): {expected}

System answer: {answer}

Citations the system gave (ref → article → quoted source text):
{citations}

Retrieved contexts, in rank order:
{contexts}

Return:
- correctness: 2 = says what the reference says (wording may differ; numbers must match), \
1 = partly correct or incomplete, 0 = wrong or does not answer.
- claims_total: number of factual claims in the system answer; claims_supported: how many of \
them the retrieved contexts state.
- citations: for each citation ref, does its quoted text support the sentence of the answer \
that cites it?
- relevant: one boolean per retrieved context, in order: does it help answer the question?
"""


def norm_article(a: str) -> str:
    return a.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset", type=Path)
    ap.add_argument("run", type=Path)
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tests = {r["id"]: r for r in csv.DictReader(args.testset.open(encoding="utf-8"))}
    results = {}
    for f in sorted(args.run.glob("b*.json")):
        for r in json.loads(f.read_text(encoding="utf-8"))["results"]:
            results[r["id"]] = r
    missing = set(tests) - set(results)
    if missing:
        print(f"WARNING: {len(missing)} questions have no result: {sorted(missing)[:5]}")

    rows = []
    for qid, t in tests.items():
        r = results.get(qid)
        if r is None:
            continue
        expected_arts = [a for a in t["expected_article"].split(";") if a]
        ctx = r["contexts"][:5]

        def same_doc(c: dict, t: dict = t) -> bool:
            return t["expected_doc_id"] in c.get("doc_aliases", [c["doc_id"]])

        doc_hit = any(same_doc(c) for c in ctx)
        art_ranks = [
            i for i, c in enumerate(ctx, 1) if same_doc(c) and c["article"] in expected_arts
        ]
        rows.append(
            {
                "id": qid,
                "test": t,
                "result": r,
                "answerable": t["answerable"] == "true",
                "doc_hit": doc_hit,
                "art_hit": bool(art_ranks),
                "all_arts_hit": all(
                    any(same_doc(c) and c["article"] == a for c in ctx) for a in expected_arts
                ),
                "rr": 1 / art_ranks[0] if art_ranks else 0.0,
                "fabricated": sum(
                    c["chunk_id"] not in {x["chunk_id"] for x in r["contexts"]}
                    or not re.fullmatch(r"https://lex\.uz/docs/-?\d+", c["url"])
                    for c in r["citations"]
                ),
                "stale": sum(c["doc_status"] != "amalda" for c in r["citations"]),
            }
        )

    # ---- LLM judge -------------------------------------------------------------------
    to_judge = [x for x in rows if x["answerable"] and x["result"]["status"] == "ok"]
    if not args.no_judge:
        s = get_settings()
        llm = OpenRouterLLM(s)

        def judge(x: dict) -> dict:
            r = x["result"]
            cits = "\n".join(
                f"[{c['ref']}] {c['doc_title'][:60]} — {c['article']}: {c['quote']}"
                for c in r["citations"]
            )
            ctxs = "\n\n".join(
                f"#{c['rank']} {c['doc_title'][:60]} — {c['article']}\n{(c['text'] or '')[:1500]}"
                for c in r["contexts"][:5]
            )
            c = llm.complete_json(
                [{"role": "user", "content": JUDGE_PROMPT.format(
                    question=x["test"]["question"], expected=x["test"]["expected_answer"],
                    answer=r["answer"], citations=cits or "(none)", contexts=ctxs)}],
                JUDGE_SCHEMA, name="grade", seed=7, reasoning_effort="medium",
                max_tokens=8000,  # reasoning tokens count against the limit
            )  # fmt: skip
            return c.data

        def safe_judge(x: dict) -> dict | None:
            try:
                return judge(x)
            except Exception as e:  # one failed grade must not sink the report
                print(f"  judge failed on {x['id']}: {str(e)[:120]}")
                return None

        with ThreadPoolExecutor(4) as pool:
            for x, grade in zip(to_judge, pool.map(safe_judge, to_judge), strict=True):
                if grade is not None:
                    x["grade"] = grade
        llm.close()

    # ---- report ------------------------------------------------------------------------
    ans = [x for x in rows if x["answerable"]]
    unans = [x for x in rows if not x["answerable"]]
    graded = [x for x in ans if "grade" in x]

    def pct(n: int, d: int) -> str:
        return f"{n / d:6.1%}" if d else "   n/a"

    print(f"\n{len(rows)} questions, {len(ans)} answerable, {len(unans)} unanswerable\n")
    print("Retrieval (answerable)")
    print(f"  Hit@5 (doc)        {pct(sum(x['doc_hit'] for x in ans), len(ans))}   target ≥ 90%")
    print(f"  Hit@5 (article)    {pct(sum(x['art_hit'] for x in ans), len(ans))}   target ≥ 80%")
    print(f"  MRR                {sum(x['rr'] for x in ans) / len(ans):6.3f}   target ≥ 0.70")
    if graded:
        rel = [b for x in graded for b in x["grade"]["relevant"][:5]]
        print(f"  Context precision  {pct(sum(rel), len(rel))}   target ≥ 60%  (judge)")
        refused = [x for x in ans if x["result"]["status"] != "ok"]
        scores = [x["grade"]["correctness"] for x in graded] + [0] * len(refused)
        claims = sum(x["grade"]["claims_total"] for x in graded)
        supported = sum(x["grade"]["claims_supported"] for x in graded)
        cit = [c["supports"] for x in graded for c in x["grade"]["citations"]]
        print("Generation (judge)")
        print(f"  Answer correctness {statistics.mean(scores):6.2f}   target ≥ 1.6  (refusals = 0)")
        print(f"  Faithfulness       {pct(supported, claims)}   target ≥ 95%")
        print(f"  Citation accuracy  {pct(sum(cit), len(cit))}   target ≥ 90%")
    print("Sources")
    print(f"  Fabricated         {sum(x['fabricated'] for x in rows):6d}   target 0")
    print(f"  Stale norm         {sum(x['stale'] for x in rows):6d}   target 0")
    print("Safety")
    na = sum(x["result"]["no_answer"] for x in unans)
    print(f"  No-answer          {na:3d} / {len(unans)}   target ≥ 9/10")
    fr = [x["id"] for x in ans if x["result"]["no_answer"]]
    print(f"  Refused answerable {len(fr):3d}        {fr}")
    totals = [x["result"]["timing"]["total_ms"] for x in rows]
    q = statistics.quantiles(totals, n=20)
    errors = sum(x["result"]["status"] in ("error", "timeout") for x in rows)
    print("Operational")
    print(f"  Latency p50 / p95  {statistics.median(totals) / 1000:.1f} s / {q[18] / 1000:.1f} s"
          f"   target ≤ 5 s / ≤ 12 s")  # fmt: skip
    print(f"  Error rate         {pct(errors, len(rows))}   target ≤ 2%")

    print("\nBy category")
    by_cat = defaultdict(list)
    for x in rows:
        by_cat[x["test"]["category"]].append(x)
    for cat, xs in sorted(by_cat.items()):
        a = [x for x in xs if x["answerable"]]
        line = f"  {cat:<10} n={len(xs):<3}"
        if a:
            line += f" Hit@5 art {pct(sum(x['art_hit'] for x in a), len(a))}"
            g = [x["grade"]["correctness"] for x in a if "grade" in x]
            g += [0] * sum(x["result"]["status"] != "ok" for x in a)
            if g and graded:
                line += f"  correctness {statistics.mean(g):.2f}"
        else:
            line += f" no_answer {sum(x['result']['no_answer'] for x in xs)}/{len(xs)}"
        if cat == "kop_modda":
            line += f"  both articles in top-5 {pct(sum(x['all_arts_hit'] for x in a), len(a))}"
        print(line)

    if graded:  # spec: every wrong answer gets one cause
        causes = defaultdict(list)
        for x in ans:
            g = x.get("grade")
            correct = g and g["correctness"] == 2
            cites_ok = g and all(c["supports"] for c in g["citations"])
            if not x["art_hit"]:
                if not correct:
                    causes["retrieval"].append(x["id"])
            elif not correct:
                causes["generation"].append(x["id"])
            elif not cites_ok:
                causes["source"].append(x["id"])
        print("\nWhy answers were wrong (spec section 6)")
        for k in ("retrieval", "generation", "source"):
            print(f"  {k:<10} {len(causes[k]):3d}  {causes[k][:12]}")

    out = args.run / "scores.json"
    out.write_text(
        json.dumps([{k: v for k, v in x.items() if k not in ("test", "result")} for x in rows],
                   ensure_ascii=False, indent=1),
        encoding="utf-8",
    )  # fmt: skip
    print(f"\nper-question scores: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
