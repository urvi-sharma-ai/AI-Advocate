from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from ai_advocate.domain.exceptions import IngestionError
from ai_advocate.domain.models import LegalDocument


def _slug(rel_file: str) -> str:
    return hashlib.sha256(rel_file.encode("utf-8")).hexdigest()[:16]


def parse_manifest(raw: dict[str, Any]) -> list[LegalDocument]:
    docs: list[LegalDocument] = []
    entries = raw.get("documents") or []
    if not isinstance(entries, list):
        raise IngestionError("`documents` must be a list in manifest.yaml")

    for i, row in enumerate(entries):
        if not isinstance(row, dict):
            raise IngestionError(f"manifest.documents[{i}] must be a mapping")
        rel = row.get("file") or row.get("path")
        if not rel:
            raise IngestionError(f"manifest.documents[{i}] missing `file`")

        act_title = row.get("act_title")
        domain = row.get("domain")
        act_version = row.get("act_version")
        if not act_title or domain not in ("Finance", "Land") or not act_version:
            raise IngestionError(
                f"manifest entry for {rel!r} needs act_title, domain (Finance|Land), act_version"
            )

        eff = row.get("effective_date")
        effective_date: date | None = None
        if eff:
            effective_date = date.fromisoformat(str(eff))

        docs.append(
            LegalDocument(
                id=_slug(str(rel)),
                act_title=str(act_title),
                domain=domain,
                relative_file=str(rel),
                source_path="",  # resolved later against data_dir
                act_version=str(act_version),
                effective_date=effective_date,
                supersedes=str(row["supersedes"]) if row.get("supersedes") else None,
            )
        )
    return docs


def load_manifest_yaml(path: Path, data_dir: Path) -> list[LegalDocument]:
    if not path.exists():
        raise IngestionError(f"Manifest not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    docs = parse_manifest(raw)
    resolved: list[LegalDocument] = []
    for doc in docs:
        full = (data_dir / doc.relative_file).resolve()
        if not full.exists():
            raise IngestionError(f"Manifest file not found on disk: {full}")
        resolved.append(doc.model_copy(update={"source_path": str(full)}))
    return resolved
