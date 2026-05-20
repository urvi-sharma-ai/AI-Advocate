from __future__ import annotations

import hashlib
import re

from ai_advocate.domain.models import Chunk, LegalDocument

# Rough token proxy: ~4 chars per token.
_CHAR_SOFT_LIMIT = 3600
_OVERLAP_CHARS = 200

_SECTION_START = re.compile(
    r"(?mi)(?:^)\s*((?:Section|SECTION)\s+(\d+[A-Za-z]?))[.\s\u2013\u2014\-:]*\s*(.*?)\s*\n"
)


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:{chunk_index}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _normalize_ws(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(lines).strip()


def _make_prefix(section_number: str | None, section_title: str | None) -> tuple[str, str | None]:
    if section_number:
        title = (section_title or "").strip()
        prefix_body = f"[S{section_number}] {title}".strip()
        section_label = f"Section {section_number}"
        return prefix_body, section_label
    return "[Preamble]", None


def _assemble_chunk_text(prefix_body: str, body: str) -> str:
    body = _normalize_ws(body)
    return f"{prefix_body}\n---\n{body}".strip()


def _pages_blob(pages: list[tuple[int, str]]) -> tuple[str, list[tuple[int, int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page_no, txt in pages:
        marker = f"\n\n--- Page {page_no} ---\n\n"
        blob_piece = marker + (txt or "")
        start = cursor + len(marker)
        end = start + len(txt or "")
        spans.append((start, end, page_no))
        parts.append(blob_piece)
        cursor += len(blob_piece)
    return "".join(parts), spans


def _page_window_for_span(
    start: int, end: int, spans: list[tuple[int, int, int]]
) -> tuple[int, int]:
    touched = [p for (s, e, p) in spans if end > s and start < e]
    if not touched:
        return 1, 1
    return min(touched), max(touched)


def _split_long_body(
    body: str,
    *,
    soft_limit: int = _CHAR_SOFT_LIMIT,
    overlap: int = _OVERLAP_CHARS,
) -> list[str]:
    body = body.strip()
    if len(body) <= soft_limit:
        return [body] if body else []

    chunks: list[str] = []
    step = max(soft_limit - overlap, 1)
    i = 0
    while i < len(body):
        chunk = body[i : i + soft_limit].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def _iter_section_spans(blob: str) -> list[tuple[str, str, str, int, int]]:
    """
    Return list of (section_number, title, body, start, end_exclusive) for each section match.
    Body includes text after the header line up to the next section header.
    """
    out: list[tuple[str, str, str, int, int]] = []
    matches = list(_SECTION_START.finditer(blob))
    if not matches and blob.strip():
        out.append(("", "", blob.strip(), 0, len(blob)))
        return out

    if matches and matches[0].start() > 0:
        pre = blob[: matches[0].start()].strip()
        if pre:
            out.append(("", "", pre, 0, matches[0].start()))

    for i, m in enumerate(matches):
        sec_no = m.group(2)
        title = (m.group(3) or "").strip()
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(blob)
        body = blob[block_start:block_end].strip()
        out.append((sec_no, title, body, m.start(), block_end))

    return out


def chunk_document(doc: LegalDocument, pages: list[tuple[int, str]]) -> list[Chunk]:
    blob, spans = _pages_blob(pages)

    sections = _iter_section_spans(blob)
    chunks: list[Chunk] = []
    chunk_index = 0

    for sec_no, title, body, span_start, span_end in sections:
        page_start, page_end = _page_window_for_span(span_start, span_end, spans)
        sec_number = sec_no or None
        sec_title = title or None
        prefix_body, section_label = _make_prefix(sec_number, sec_title)

        sub_bodies = _split_long_body(body)
        if not sub_bodies:
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.id, chunk_index),
                    text=_assemble_chunk_text(prefix_body, ""),
                    document_id=doc.id,
                    domain=doc.domain,
                    act_title=doc.act_title,
                    act_version=doc.act_version,
                    source_path=doc.source_path,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    section_number=sec_number,
                    section_title=sec_title,
                    section_label=section_label,
                    subsection_label=None,
                )
            )
            chunk_index += 1
            continue

        for sub in sub_bodies:
            chunks.append(
                Chunk(
                    id=_chunk_id(doc.id, chunk_index),
                    text=_assemble_chunk_text(prefix_body, sub),
                    document_id=doc.id,
                    domain=doc.domain,
                    act_title=doc.act_title,
                    act_version=doc.act_version,
                    source_path=doc.source_path,
                    chunk_index=chunk_index,
                    page_start=page_start,
                    page_end=page_end,
                    section_number=sec_number,
                    section_title=sec_title,
                    section_label=section_label,
                    subsection_label=None,
                )
            )
            chunk_index += 1

    return chunks
