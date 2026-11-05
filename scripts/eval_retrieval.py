"""Retrieval metrics of the teacher's spec on a question set, for several settings.

Hit@5 (doc)     — expected act among the top-5 contexts
Hit@5 (article) — expected act AND article among the top-5
MRR             — mean of 1/rank of the expected article (0 if not in top-k)
A context matches the expected act through `aliases`, so the id form does not matter.

    uv run python scripts/eval_retrieval.py eval/dev_synthetic.jsonl --rerank cohere/rerank-5-fast
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from lexrag.config import get_settings
from lexrag.index import db
from lexrag.rag.pipeline import RagPipeline
from lexrag.text.lang import detect_language


def evaluate(rag: RagPipeline, rows: list[dict], k: int, aliases_of: dict[int, set[int]]):
    per_lang: dict[str, list[tuple[bool, bool, float]]] = defaultdict(list)
    misses, ms = [], []
    for row in rows:
        lang = row["language"] or detect_language(row["question"])
        r, _ = rag.retrieve(row["question"], lang, k)
        ms.append(r.timing_ms["retrieval_ms"])
        expected = set(row["expected_aliases"])
        rank_doc = next(
            (i for i, c in enumerate(r.contexts, 1) if aliases_of[c.doc_id] & expected), None
        )
        rank_art = next(
            (
                i
                for i, c in enumerate(r.contexts, 1)
                if aliases_of[c.doc_id] & expected and c.article == row["expected_article"]
            ),
            None,
        )
        per_lang[lang].append(
            (rank_doc is not None, rank_art is not None, 1 / rank_art if rank_art else 0.0)
        )
        if rank_art is None:
            got = ", ".join(f"{c.doc_title[:28]}/{c.article}" for c in r.contexts[:3])
            want = f"{row['doc_title'][:35]}/{row['expected_article']}"
            misses.append(f"  {row['id']} {lang:<7} want {want}  got: {got}")
    return per_lang, misses, ms


def report(name: str, per_lang, ms) -> None:
    allrows = [x for v in per_lang.values() for x in v]

    def line(label: str, xs) -> str:
        n = len(xs)
        doc, art, mrr = (sum(x[i] for x in xs) / n for i in range(3))
        return f"  {label:<8} n={n:<3} Hit@5 doc {doc:6.1%}   Hit@5 art {art:6.1%}   MRR {mrr:.3f}"

    print(f"\n== {name}   retrieval p50 {statistics.median(ms):.0f} ms, max {max(ms)} ms")
    print(line("all", allrows))
    for lang in sorted(per_lang):
        print(line(lang, per_lang[lang]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument(
        "--rerank",
        nargs="*",
        default=[],
        help="reranker models to compare; empty string = no reranker",
    )
    ap.add_argument("--misses", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(x) for x in args.path.read_text(encoding="utf-8").splitlines() if x]
    rows = [r for r in rows if r["answerable"]]
    s = get_settings()
    with db.connect(s.database_url) as conn:
        aliases_of = {d: set(a) for d, a in conn.execute("SELECT doc_id, aliases FROM documents")}
        for config in args.rerank or [""]:
            rag = RagPipeline(conn, s.model_copy(update={"rerank_model": config}))
            per_lang, misses, ms = evaluate(rag, rows, args.k, aliases_of)
            rag.close()
            report(config or "hybrid (no reranker)", per_lang, ms)
            if args.misses:
                print("\n".join(misses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
