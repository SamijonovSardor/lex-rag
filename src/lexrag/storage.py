"""File storage for the bronze (raw HTML) and silver (parsed JSON) layers.

Writes are atomic (temp file + rename): an Airflow task killed mid-write never leaves
a half-written file that a downstream task would happily read.
"""

import gzip
import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from lexrag.models import Document


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class RawRecord:
    doc_id: int
    path: Path
    sha256: str  # of the HTML as served; parsed-text hash lives on Document.content_hash
    size: int
    fetched_at: datetime


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, doc_id: int) -> Path:
        return self.root / f"{doc_id}.html.gz"

    def exists(self, doc_id: int) -> bool:
        return self.path(doc_id).exists()

    def save(self, doc_id: int, html: str) -> RawRecord:
        data = html.encode("utf-8")
        path = self.path(doc_id)
        _atomic_write(path, gzip.compress(data, compresslevel=6))
        return RawRecord(
            doc_id=doc_id,
            path=path,
            sha256=hashlib.sha256(data).hexdigest(),
            size=len(data),
            fetched_at=datetime.now(UTC),
        )

    def load(self, doc_id: int) -> str:
        return gzip.decompress(self.path(doc_id).read_bytes()).decode("utf-8")

    def ids(self) -> list[int]:
        return sorted(int(p.name.removesuffix(".html.gz")) for p in self.root.glob("*.html.gz"))


class ParsedStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, doc_id: int) -> Path:
        return self.root / f"{doc_id}.json"

    def save(self, doc: Document) -> Path:
        path = self.path(doc.doc_id)
        _atomic_write(path, doc.model_dump_json(indent=1).encode("utf-8"))
        return path

    def load(self, doc_id: int) -> Document:
        return Document.model_validate_json(self.path(doc_id).read_bytes())

    def ids(self) -> list[int]:
        return sorted(int(p.stem) for p in self.root.glob("*.json"))


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> int:
    lines = [row.model_dump_json() for row in rows]
    _atomic_write(path, ("\n".join(lines) + "\n").encode("utf-8") if lines else b"")
    return len(lines)


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> Iterator[M]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)
