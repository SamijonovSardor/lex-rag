"""Batch results in Postgres: idempotency key, async polling, 30-day retention.

The spec has two rules that meet here:
  * the same batch_id sent again returns the stored result, nothing is recomputed;
  * a batch whose answer had a retryable error is sent again with the same batch_id.
Both hold: a stored batch is returned as is, except that its items that failed with a
retryable error are answered again (and only those).
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  batch_id     text PRIMARY KEY,
  run_id       text NOT NULL,
  request_hash text NOT NULL,
  status       text NOT NULL,          -- processing | completed | failed
  response     jsonb,
  received_at  timestamptz NOT NULL,
  completed_at timestamptz
);
"""


@dataclass
class StoredBatch:
    batch_id: str
    run_id: str
    request_hash: str
    status: str
    response: dict[str, Any] | None
    received_at: datetime


class BatchStore:
    def __init__(self, pool: ConnectionPool) -> None:
        self.pool = pool
        with pool.connection() as conn:
            conn.execute(SCHEMA)

    def claim(
        self, batch_id: str, run_id: str, request_hash: str, received_at: datetime
    ) -> StoredBatch | None:
        """Register a new batch. Returns None if it is new, else the stored one."""
        with self.pool.connection() as conn:
            inserted = conn.execute(
                "INSERT INTO batches (batch_id, run_id, request_hash, status, received_at) "
                "VALUES (%s, %s, %s, 'processing', %s) ON CONFLICT (batch_id) DO NOTHING "
                "RETURNING batch_id",
                (batch_id, run_id, request_hash, received_at),
            ).fetchone()
        return None if inserted else self.get(batch_id)

    def restart(self, batch_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE batches SET status = 'processing', completed_at = NULL WHERE batch_id = %s",
                (batch_id,),
            )

    def finish(self, batch_id: str, status: str, response: dict[str, Any], at: datetime) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE batches SET status = %s, response = %s, completed_at = %s "
                "WHERE batch_id = %s",
                (
                    status,
                    Jsonb(response, dumps=lambda o: json.dumps(o, ensure_ascii=False)),
                    at,
                    batch_id,
                ),
            )

    def get(self, batch_id: str) -> StoredBatch | None:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT batch_id, run_id, request_hash, status, response, received_at "
                "FROM batches WHERE batch_id = %s",
                (batch_id,),
            ).fetchone()
        return StoredBatch(*row) if row else None

    def fail_abandoned(self) -> int:
        """At startup: batches left 'processing' by a previous process will never finish."""
        with self.pool.connection() as conn:
            return conn.execute(
                "UPDATE batches SET status = 'failed', completed_at = now() "
                "WHERE status = 'processing'"
            ).rowcount
