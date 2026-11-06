"""lex-rag on Airflow 3: a data-triggered pipeline (docs/faza-3.md).

    lexrag_daily    @daily     RSS → fetch new and amended acts ──┐
    lexrag_weekly   @weekly    full listing → fetch every act ────┴──▶ asset lexrag_raw
    lexrag_parse    on raw     raw HTML → parsed JSON, changed only ───▶ asset lexrag_parsed
    lexrag_index    on parsed  parsed → Postgres, incremental ─────────▶ asset lexrag_index

Each step hands the changed ids to the next in its asset event's `extra`. A step with
nothing to hand over is skipped, and a skipped task emits no event, so a quiet day ends
at the first step. All lex.uz traffic runs in the one-slot pool `lexuz`: daily and weekly
never hit the site at the same time (the client also keeps ~1 request/s).

The tasks only call lexrag.update; `lexrag update --mode daily|full` runs the same steps
without Airflow.
"""

import asyncio
from datetime import datetime
from typing import Any

from airflow.sdk import Asset, dag, task
from airflow.sdk.exceptions import AirflowSkipException

RAW = Asset("lexrag_raw")
PARSED = Asset("lexrag_parsed")
INDEX = Asset("lexrag_index")

DEFAULTS: dict[str, Any] = dict(
    start_date=datetime(2026, 11, 1), catchup=False, max_active_runs=1, tags=["lexrag"]
)


def _settings():
    from lexrag.config import Settings

    return Settings()  # LEXRAG_DATA_DIR, LEXRAG_DATABASE_URL, OPENROUTER_API_KEY from env


def _handed_over(context: dict, asset: Asset) -> tuple[list[int], list[int] | None]:
    """(changed ids, whole scope or None) from the events that triggered this run.
    A manual run takes them from its conf: {"doc_ids": [...], "scope": [...]}."""
    events = context["triggering_asset_events"].get(asset, []) if context.get(
        "triggering_asset_events") else []  # fmt: skip
    extras = [e.extra or {} for e in events] or [context["dag_run"].conf or {}]
    ids = sorted({i for x in extras for i in x.get("doc_ids", [])})
    scopes = [x["scope"] for x in extras if x.get("scope") is not None]
    return ids, (sorted(set(scopes[-1])) if scopes else None)


@dag(schedule="@daily", **DEFAULTS)
def lexrag_daily():
    @task(pool="lexuz", outlets=[RAW])
    def fetch_changes(**context) -> list[int]:
        from lexrag import update

        ids = asyncio.run(update.daily_changes(_settings()))
        if not ids:
            raise AirflowSkipException("nothing new for our scope in today's feed")
        context["outlet_events"][RAW].extra = {"doc_ids": ids}
        return ids

    fetch_changes()


@dag(schedule="@weekly", **DEFAULTS)
def lexrag_weekly():
    @task(pool="lexuz", outlets=[RAW])
    def fetch_everything(**context) -> int:
        from lexrag import update

        ids = asyncio.run(update.full_refresh(_settings()))
        context["outlet_events"][RAW].extra = {"doc_ids": ids, "scope": ids}
        return len(ids)

    fetch_everything()


@dag(schedule=[RAW], **DEFAULTS)
def lexrag_parse():
    @task(outlets=[PARSED])
    def parse(**context) -> list[int]:
        from lexrag import update

        ids, scope = _handed_over(context, RAW)
        changed = update.parse_changed(_settings(), ids)
        if not changed and scope is None:
            raise AirflowSkipException("fetched again, text unchanged: nothing to index")
        context["outlet_events"][PARSED].extra = {"doc_ids": changed, "scope": scope}
        return changed

    parse()


@dag(schedule=[PARSED], **DEFAULTS)
def lexrag_index():
    @task(outlets=[INDEX])
    def index(**context) -> dict[str, int]:
        from lexrag import update

        ids, scope = _handed_over(context, PARSED)
        report = update.index_changed(_settings(), scope or ids, prune=scope is not None)
        counts = {
            "docs_written": report.docs_written,
            "docs_deleted": report.docs_deleted,
            "chunks_embedded": report.chunks_embedded,
            "chunks_reused": report.chunks_reused,
        }
        context["outlet_events"][INDEX].extra = counts
        return counts

    index()


lexrag_daily()
lexrag_weekly()
lexrag_parse()
lexrag_index()
