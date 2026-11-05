import gzip
from functools import cache
from pathlib import Path

import pytest

from lexrag.config import Settings
from lexrag.models import Document
from lexrag.scraper.parse import parse_document

FIXTURES = Path(__file__).parent / "fixtures"

# Real lex.uz pages captured on 2026-10-30 (see docs/recon.md).
JK = -111453  # Jinoyat kodeksi, Latin — big, amended many times, articles like 18¹
KSUD = -8536927  # Konstitutsiyaviy sud toʻgʻrisidagi qonun (2026) — full requisites block
CONST_CYRL = 6445145  # Konstitutsiya, Cyrillic original
CONST_LATN = -6445145  # Konstitutsiya, official Latin twin


def fixture_text(name: str) -> str:
    return gzip.decompress((FIXTURES / name).read_bytes()).decode("utf-8")


@cache
def parsed(name: str, doc_id: int) -> Document:
    return parse_document(fixture_text(f"{name}.html.gz"), doc_id)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        request_interval_s=0.0,
        retry_initial_wait_s=0.0,
        max_attempts=3,
    )
