"""The tester's side of the spec (sections 1, 3, 5): send the 100 questions as 10 batches,
one after another, and save every response as <out>/<run_id>/bNN.json.

Expected answers and sources never leave this script (the spec: they stay with the tester).

    uv run python scripts/run_test.py eval/testset.csv --base-url http://127.0.0.1:8000 \
        --run-id eval-2026-11-06
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

RETRY_WAITS = (5, 15, 45)  # spec section 5: 500/503/504 → up to 3 retries


def send(client: httpx.Client, body: dict) -> dict:
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            r = client.post("/v1/rag/batch", json=body)
        except httpx.TransportError as e:
            r = None
            print(f"    transport error: {e}")
        if r is not None and r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "1"))
            print(f"    429, waiting {wait} s")
            time.sleep(wait)
            continue
        if r is not None and r.status_code == 202:
            while True:  # async mode: poll until done
                time.sleep(5)
                g = client.get(f"/v1/rag/batch/{body['batch_id']}")
                if g.json().get("status") != "processing":
                    return g.json()
        if r is not None and r.status_code == 200:
            return r.json()
        if r is not None and r.status_code < 500:
            raise SystemExit(f"{r.status_code}: {r.text[:300]}")
        if attempt < len(RETRY_WAITS):
            print(f"    {r.status_code if r else 'no answer'}, retry in {RETRY_WAITS[attempt]} s")
            time.sleep(RETRY_WAITS[attempt])
    raise SystemExit(f"batch {body['batch_id']} failed after retries")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", type=Path, default=Path("data/runs"))
    ap.add_argument("--api-key", default=os.environ.get("LEXRAG_TEST_KEY", ""))
    ap.add_argument("--send-language", action="store_true", help="send items[].language")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    key = args.api_key
    if not key:  # local run: take the first key the server reads from .env
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("LEXRAG_API_KEYS="):
                key = line.split("=", 1)[1].split(",")[0].strip()
    rows = list(csv.DictReader(args.testset.open(encoding="utf-8")))
    batches: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        batches[row["batch"]].append(row)

    out = args.out / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {key}"}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=150.0) as client:
        print(json.dumps(client.get("/v1/meta").json(), ensure_ascii=False))
        for i, name in enumerate(sorted(batches), start=1):
            body = {
                "run_id": args.run_id,
                "batch_id": f"{args.run_id}-{name}",
                "batch_index": i,
                "total_batches": len(batches),
                "options": {
                    "top_k": 5, "return_contexts": True, "return_debug": True,
                    "temperature": 0, "seed": 42, "answer_language": "auto",
                },
                "items": [
                    {"id": r["id"], "question": r["question"],
                     **({"language": r["language"]} if args.send_language else {})}
                    for r in batches[name]
                ],
            }  # fmt: skip
            t = time.perf_counter()
            resp = send(client, body)
            retryable = [x["id"] for x in resp["results"] if (x["error"] or {}).get("retryable")]
            if retryable:  # spec: a retryable error → the same batch_id again
                print(f"    retrying {retryable}")
                resp = send(client, body)
            (out / f"{name}.json").write_text(
                json.dumps(resp, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            print(f"  {name}: {resp['summary']}  {time.perf_counter() - t:.1f} s")
    print(f"saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
