"""Neon's free compute sleeps after 5 idle minutes and closes every connection. The API's
long-lived connections must survive that: simulated here with pg_terminate_backend."""

import psycopg
import pytest

from lexrag.config import Settings
from lexrag.index import db
from lexrag.rag.pipeline import RagPipeline

URL = Settings().database_url


def kill(conn: psycopg.Connection) -> None:
    with db.connect(URL, timeout_s=3) as admin:
        admin.execute("SELECT pg_terminate_backend(%s)", (conn.info.backend_pid,))


@pytest.fixture
def conn():
    try:
        c = db.connect(URL, timeout_s=3)
    except psycopg.OperationalError as e:
        pytest.skip(f"no database: {e}")
    yield c
    c.close()


def test_pipeline_reconnects_after_the_database_drops_it(conn) -> None:
    rag = RagPipeline(conn, Settings(openrouter_api_key="k"), connect=lambda: db.connect(URL))
    kill(conn)
    rag.ensure_connection()
    fresh = rag.retriever.conn
    assert fresh is not conn and not fresh.closed
    assert fresh.execute("SHOW hnsw.ef_search").fetchone() == ("100",)  # session setting redone
    rag.close()
    fresh.close()


def test_without_a_factory_the_error_surfaces(conn) -> None:
    rag = RagPipeline(conn, Settings(openrouter_api_key="k"))
    kill(conn)
    with pytest.raises(psycopg.OperationalError):
        rag.ensure_connection()
    rag.close()
