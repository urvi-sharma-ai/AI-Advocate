from __future__ import annotations

from pathlib import Path

import pytest

from ai_advocate.infrastructure.config import Settings


def test_empty_manifest_path_env_resolves_to_data_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MANIFEST_PATH", raising=False)
    monkeypatch.setenv("MANIFEST_PATH", "")
    s = Settings()
    assert s.manifest_path is None
    assert s.resolved_manifest() == Path("data") / "manifest.yaml"


def test_dot_manifest_path_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MANIFEST_PATH", ".")
    s = Settings()
    assert s.manifest_path is None
    assert s.resolved_manifest() == Path("data") / "manifest.yaml"
