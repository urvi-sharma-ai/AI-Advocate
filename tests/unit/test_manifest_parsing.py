from __future__ import annotations

from pathlib import Path

from ai_advocate.infrastructure.pdf.manifest import load_manifest_yaml


def test_manifest_loads_all_docs() -> None:
    docs = load_manifest_yaml(Path("data/manifest.yaml"), Path("data"))
    assert len(docs) == 7
    assert {d.domain for d in docs} == {"Finance", "Land"}
    assert any(d.act_version == "IT_2025" for d in docs)
