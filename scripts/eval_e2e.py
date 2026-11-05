"""End-to-end run of the pipeline over a question set: what the API would return.

Reports what can be measured without a judge: status mix, no-answer behaviour, citation
checks, whether a cited article is the expected one, latency, tokens. Answer correctness
and faithfulness need an LLM judge (Phase 5).

    uv run python scripts/eval_e2e.py eval/dev_synthetic.jsonl --out data/runs/dev.jsonl
"""

import argparse
import json
import statistics
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from lexrag.api.schemas import Options, to_result
from lexrag.config import get_settings
from lexrag.index import db
from lexrag.rag.pipeline import RagPipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/runs/dev-e2e.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(x) for x in args.path.read_text(encoding="utf-8").splitlines() if x]
    s = get_settings()
    local = threading.local()
    pipelines: list[RagPipeline] = []

    def answer(row: dict) -> dict:
        if not hasattr(local, "rag"):
            local.rag = RagPipeline(db.connect(s.database_url), s)
            pipelines.append(local.rag)
        r = local.rag.run(row["id"], row["question"])
        return to_result(r, Options(return_debug=True)).model_dump(mode="json")

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(answer, rows))
    for p in pipelines:
        p.close()

    with db.connect(s.database_url) as conn:
        aliases = {
            f"lex:{lex}": set(a) for lex, a in conn.execute("SELECT lex_id, aliases FROM documents")
        }

    status = Counter(r["status"] for r in results)
    answerable = [(row, r) for row, r in zip(rows, results, strict=True) if row["answerable"]]
    unanswerable = [(row, r) for row, r in zip(rows, results, strict=True) if not row["answerable"]]
    cited_right = sum(
        any(
            aliases.get(c["doc_id"], set()) & set(row["expected_aliases"])
            and c["article"] == row["expected_article"]
            for c in r["citations"]
        )
        for row, r in answerable
    )
    rejected = sum(len((r.get("debug") or {}).get("rejected_citations", [])) for r in results)
    kept = sum(len(r["citations"]) for r in results)
    outside = sum(
        c["chunk_id"] not in {x["chunk_id"] for x in r["contexts"]}
        for r in results
        for c in r["citations"]
    )
    totals = [r["timing"]["total_ms"] for r in results if r["timing"].get("total_ms")]
    tin = sum((r["usage"] or {}).get("input_tokens", 0) for r in results)
    tout = sum((r["usage"] or {}).get("output_tokens", 0) for r in results)

    print(f"status: {dict(status)}")
    print(f"answerable {len(answerable)}: answered {sum(r['status'] == 'ok' for _, r in answerable)}, "
          f"cites the expected article {cited_right} ({cited_right / len(answerable):.0%})")  # fmt: skip
    print(f"unanswerable {len(unanswerable)}: no_answer "
          f"{sum(r['no_answer'] for _, r in unanswerable)}/{len(unanswerable)}")  # fmt: skip
    print(
        f"citations kept {kept}, rejected by the quote check {rejected}, outside contexts {outside}"
    )
    q = statistics.quantiles(totals, n=20)
    print(f"latency per question: p50 {statistics.median(totals) / 1000:.1f} s, "
          f"p95 {q[18] / 1000:.1f} s, max {max(totals) / 1000:.1f} s")  # fmt: skip
    print(f"LLM tokens: {tin:,} in, {tout:,} out "
          f"(~${tin * 0.1e-6 + tout * 0.5e-6:.3f} at gpt-6-luna prices)")  # fmt: skip

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row, r in zip(rows, results, strict=True):
            f.write(json.dumps({"question": row, "result": r}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
