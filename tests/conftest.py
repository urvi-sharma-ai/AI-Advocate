from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_openai_env_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests should not call live OpenAI unless explicitly requested."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
