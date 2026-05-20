"""Ensure SQLite has FTS5 before ChromaDB initializes (uv Python often lacks it)."""

from __future__ import annotations

import sqlite3
import sys


def _stdlib_has_fts5() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts_probe USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


def ensure_sqlite_fts5() -> None:
    """
    ChromaDB 0.6+ migrations require SQLite FTS5 (see metadb/00003-full-text-tokenize).

    uv-managed CPython on macOS often ships sqlite3 without FTS5; pysqlite3 provides it.
    """
    if _stdlib_has_fts5():
        return

    try:
        import pysqlite3  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "SQLite FTS5 is required by ChromaDB but is missing from this Python build. "
            "Install the project dependency `pysqlite3` (already in pyproject.toml) via `uv sync`."
        ) from exc

    sys.modules["sqlite3"] = pysqlite3
