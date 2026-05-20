from __future__ import annotations

from pathlib import Path

import pdfplumber

from ai_advocate.domain.exceptions import IngestionError
from ai_advocate.domain.models import LegalDocument


def load_pdf_pages(doc: LegalDocument) -> list[tuple[int, str]]:
    path = Path(doc.source_path)
    if path.name.startswith("."):
        return []

    try:
        with pdfplumber.open(path) as pdf:
            pages: list[tuple[int, str]] = []
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                pages.append((idx, text))
            return pages
    except Exception as exc:
        raise IngestionError(f"Failed to read PDF {path}: {exc}") from exc


def iter_pdf_paths(data_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(data_dir.rglob("*.pdf")):
        if ".DS_Store" in str(p):
            continue
        if any(part.startswith(".") for part in p.parts):
            continue
        paths.append(p)
    return paths
