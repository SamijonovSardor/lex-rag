from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEXRAG_", env_file=".env", extra="ignore")

    base_url: str = "https://lex.uz"
    user_agent: str = "Mozilla/5.0 (compatible; lex-rag/0.1; student research project)"
    # Minimum gap between two requests, shared by every coroutine of a client.
    # lex.uz is a public government service: we stay at ~1 req/s.
    request_interval_s: float = 1.0
    request_timeout_s: float = 60.0
    max_attempts: int = 5
    retry_initial_wait_s: float = 2.0  # exponential backoff with jitter, capped at 60 s

    data_dir: Path = Path("data")

    # Phase 2 — index. Port 5434: this machine's native PostgreSQL (no pgvector) holds 5432/5433.
    database_url: str = "postgresql://lexrag:lexrag@127.0.0.1:5434/lexrag"
    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "baai/bge-m3"
    embedding_dim: int = 1024
    embed_batch_size: int = 64
    # Which lex.uz id the API reports as `doc_id`: the original act (`uz_cyrl`, the id lex.uz
    # shows without a language prefix) or the Latin variant we index (`uz_latn`, negative).
    canonical_id: Literal["uz_cyrl", "uz_latn"] = "uz_cyrl"

    # Phase 4 — retrieval and answers
    llm_model: str = "openai/gpt-6-luna"
    # gpt-6-luna is a reasoning model: it ignores temperature, honours seed and effort.
    llm_reasoning_effort: str = "minimal"  # answers (dev set: same quality as "low", faster tail)
    rewrite_reasoning_effort: str = "minimal"  # Russian → Uzbek query
    rerank_model: str = "cohere/rerank-5-pro"  # empty = no reranking (docs/faza-4.md)
    retrieval_candidates: int = 50  # per search path (vector, full text) before fusion
    rerank_candidates: int = 30  # fused candidates the reranker reads
    neighbours_of: int = (
        0  # >0: adjacent articles of the top N join the rerank pool (tested: no gain)
    )

    # Phase 6 — the API of the teacher's spec
    api_keys: str = ""  # comma-separated bearer keys; one per tester
    api_workers: int = 4  # questions of a batch answered in parallel
    batch_sync_timeout_s: float = 110.0  # spec: ≤ 120 s sync, else 202 + GET
    rate_limit_per_s: float = 5.0  # batches per second per key (spec: at least 1)

    # Public demo page (no key). Spends the same OpenRouter balance as the test API, so it
    # is capped and pauses itself before it can eat the reserve the API needs.
    demo_enabled: bool = True
    demo_per_ip_per_hour: int = 10
    demo_per_day: int = 50
    demo_min_budget_usd: float = 0.15  # pause the demo below this OpenRouter key balance
    demo_max_chars: int = 500

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def parsed_dir(self) -> Path:
        return self.data_dir / "parsed"

    @property
    def catalog_path(self) -> Path:
        return self.data_dir / "catalog.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
