"""Document → chunks, the pieces we embed and retrieve.

One article = one chunk, so every citation names an exact article (the teacher's spec asks
whether chunking is by article or by length: by article). An article longer than MAX_CHARS
is cut at paragraph boundaries into pieces of about TARGET_CHARS, and each piece repeats
the previous piece's last paragraph, so a sentence that leans on the one before it is not
cut off from it.

Decrees and resolutions have no articles: their rule is the numbered point (band, "3. …").
In a document without articles each band is one chunk ("3-band"); its sub-items (a), 1)
stay with it, and the text before the first band (the preamble) is a chunk of its own.

Size of the corpus on 2026-10-30: median unit 771 characters, p90 2 350, p99 6 059,
longest 129 223 (a table). 556 of 18 646 units exceed 4 000.
"""

import hashlib
import re

from lexrag.models import Chunk, ChunkParagraph, Document, IndexedDocument, Unit

CHUNKER_VERSION = "2"  # bump when the output changes: forces a rewrite (vectors are reused)
MAX_CHARS = 4000
TARGET_CHARS = 2000

# A list band continues the current part (qism) of an article instead of opening a new one:
# "1)", "12-1)", "a)", "gʻ)", "—", "•". Seen in the corpus: 1) … 8), a) b) v) …
_BAND_RE = re.compile(r"^\s*(?:\d+(?:-\d+)?\)|[a-zʻ]{1,2}\)|[—–•]\s?|-\s)", re.IGNORECASE)
# A numbered point of a decree or resolution: "3. Quyidagilar:", "12-1. …" (not "1.1." or "2026.")
_POINT_RE = re.compile(r"^\s*(\d{1,3}(?:-\d{1,2})?)\.\s")
_SUPERSCRIPT_CHARS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPTS = str.maketrans(_SUPERSCRIPT_CHARS, "0123456789")
_SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


def number_parts(texts: list[str]) -> list[int]:
    """Part (qism) number of each paragraph of one article."""
    parts: list[int] = []
    qism = 0
    for i, text in enumerate(texts):
        if i == 0 or not _BAND_RE.match(text):
            qism += 1
        parts.append(qism)
    return parts


def article_slug(number: str) -> str:
    """'18¹' → '18.1', '223³' → '223.3': ASCII, so chunk ids survive URLs and logs."""
    plain = number.translate(_SUPERSCRIPTS)
    first_sup = next((i for i, ch in enumerate(number) if ch in _SUPERSCRIPT_CHARS), None)
    return plain if first_sup is None else f"{plain[:first_sup]}.{plain[first_sup:]}"


def split_points(paragraphs: list[ChunkParagraph]) -> list[tuple[str | None, list[ChunkParagraph]]]:
    """Paragraphs of a unit without articles → (point number or None, its paragraphs)."""
    groups: list[tuple[str | None, list[ChunkParagraph]]] = []
    for p in paragraphs:
        m = _POINT_RE.match(p.text)
        if m or not groups:
            groups.append((m[1] if m else None, []))
        groups[-1][1].append(p)
    return groups


def _split_long_paragraph(p: ChunkParagraph) -> list[ChunkParagraph]:
    """A single paragraph over MAX_CHARS (in practice a table) → slices at sentence ends."""
    if len(p.text) <= MAX_CHARS:
        return [p]
    slices: list[str] = []
    cur = ""
    for sentence in _SENTENCE_END_RE.split(p.text):
        while len(sentence) > TARGET_CHARS:  # no sentence end for a long stretch: hard cut
            cut = sentence.rfind(" ", 0, TARGET_CHARS)
            cut = cut if cut > TARGET_CHARS // 2 else TARGET_CHARS
            if cur:
                slices.append(cur)
                cur = ""
            slices.append(sentence[:cut])
            sentence = sentence[cut:].lstrip()
        if cur and len(cur) + 1 + len(sentence) > TARGET_CHARS:
            slices.append(cur)
            cur = ""
        cur = f"{cur} {sentence}" if cur else sentence
    if cur:
        slices.append(cur)
    return [ChunkParagraph(elem_id=p.elem_id, qism=p.qism, text=s) for s in slices]


def split_pieces(paragraphs: list[ChunkParagraph]) -> list[list[ChunkParagraph]]:
    if sum(len(p.text) for p in paragraphs) <= MAX_CHARS:
        return [paragraphs]
    flat = [s for p in paragraphs for s in _split_long_paragraph(p)]
    pieces: list[list[ChunkParagraph]] = []
    cur: list[ChunkParagraph] = []
    size = 0
    for p in flat:
        if cur and size + len(p.text) > TARGET_CHARS:
            pieces.append(cur)
            # Overlap: carry the last paragraph over, unless it is itself a big slice.
            cur = [cur[-1]] if len(cur[-1].text) <= TARGET_CHARS // 2 else []
            size = sum(len(x.text) for x in cur)
        cur.append(p)
        size += len(p.text)
    if cur:
        pieces.append(cur)
    return pieces


def embed_text(title: str, unit: Unit, paragraphs: list[ChunkParagraph]) -> str:
    lines = [title]
    if unit.path:
        lines.append(" / ".join(unit.path))
    if unit.heading and (not unit.path or unit.heading != unit.path[-1]):
        lines.append(unit.heading)
    lines.append("")
    lines.extend(p.text for p in paragraphs)
    return "\n".join(lines)


def embed_hash(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def chunk_document(doc: Document, meta: IndexedDocument, model: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    used_anchors: set[str] = set()
    by_points = not any(u.number for u in doc.units)  # a decree or resolution
    for unit in doc.units:
        if not any(p.text.strip() for p in unit.paragraphs):
            continue
        plain = [ChunkParagraph(elem_id=p.elem_id, qism=0, text=p.text) for p in unit.paragraphs]
        groups = split_points(plain) if by_points else [(None, plain)]
        for point, group in groups:
            parts = number_parts([p.text for p in group])
            paragraphs = [
                ChunkParagraph(elem_id=p.elem_id, qism=q, text=p.text)
                for p, q in zip(group, parts, strict=True)
            ]
            if unit.number:
                anchor, article = f"art-{article_slug(unit.number)}", f"{unit.number}-modda"
            elif point:
                anchor, article = f"band-{point}", f"{point}-band"
            else:
                anchor, article = f"u{unit.position}", None
            if anchor in used_anchors and not point:  # e.g. an act quoting "12¹-modda" twice
                anchor = f"u{unit.position}"
            if anchor in used_anchors:  # bands restart in an annex, even inside one unit
                base, n = f"u{unit.position}-{anchor}", 2
                anchor = base
                while anchor in used_anchors:
                    anchor, n = f"{base}~{n}", n + 1
            used_anchors.add(anchor)

            pieces = split_pieces(paragraphs)
            for i, piece in enumerate(pieces, start=1):
                text = embed_text(meta.title, unit, piece)
                chunks.append(
                    Chunk(
                        chunk_id=f"lex:{meta.lex_id}#{anchor}#p{i}",
                        doc_id=doc.doc_id,
                        unit_position=unit.position,
                        elem_id=group[0].elem_id if point else unit.elem_id,
                        kind=unit.kind,
                        article=article,
                        article_title=unit.title,
                        piece=i,
                        pieces=len(pieces),
                        heading=unit.heading,
                        path=unit.path,
                        paragraphs=piece,
                        amendments=unit.amendments,
                        embed_text=text,
                        embed_hash=embed_hash(model, text),
                    )
                )
    return chunks
