# lex-rag

Question answering over Uzbekistan legislation ([lex.uz](https://lex.uz)) with article-level citations.

Pipeline: **scrape lex.uz → Airflow (data-triggered DAGs) → hybrid retrieval (pgvector + full-text) → LLM answer with citations → web app.**

## Status

- [x] Phase 0 — lex.uz reconnaissance ([docs/recon.md](docs/recon.md))
- [x] Phase 1 — scraper, parser, Uzbek text normalization
- [x] Phase 2 — chunking, embeddings (`baai/bge-m3`), Postgres + pgvector ([docs/faza-2.md](docs/faza-2.md))
- [x] Phase 3 — Airflow 3 DAGs: daily RSS + weekly full refresh, data-triggered parse → index ([docs/faza-3.md](docs/faza-3.md))
- [x] Phase 4 — hybrid retrieval + Cohere rerank, answers with verified citations (`openai/gpt-6-luna`) ([docs/faza-4.md](docs/faza-4.md))
- [x] Phase 5 — our 100-question replica of the teacher's test + scoring ([docs/faza-5.md](docs/faza-5.md))
- [x] Phase 6 — API of the teacher's test spec ([docs/faza-6.md](docs/faza-6.md)) and a web page at `/`

Live: https://lex-rag-mdca.onrender.com (web page) · `/docs` (API) — deploy notes in [docs/deploy.md](docs/deploy.md)

## Run

```bash
uv sync
uv run pytest
uv run lexrag discover               # inventory → data/catalog.jsonl
uv run lexrag fetch --limit 20       # raw HTML → data/raw/
uv run lexrag parse                  # structured JSON → data/parsed/
docker compose up -d db              # Postgres + pgvector on 127.0.0.1:5434
uv run lexrag index --dry-run        # chunk stats only, no DB or API calls
uv run lexrag index                  # chunks + bge-m3 embeddings → Postgres (needs OPENROUTER_API_KEY)
uv run lexrag ask "Umumiy daʼvo muddati qancha?"   # one answer with citations
uv run uvicorn lexrag.api.app:app --port 8000      # the API + web page (keys in LEXRAG_API_KEYS)
uv run lexrag update --mode daily                  # what changed on lex.uz → fetch, parse, index
docker compose --profile airflow up -d --build     # Airflow 3 at http://127.0.0.1:8080 (~1.5 GB RAM)
uv run python scripts/eval_retrieval.py eval/dev_synthetic.jsonl --rerank cohere/rerank-5-pro
uv run python scripts/run_test.py eval/testset.csv --run-id my-run   # the teacher's test, against the API
uv run python scripts/score_run.py eval/testset.csv data/runs/my-run  # spec section 6 metrics
```
