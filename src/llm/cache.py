"""SQLite-backed LLM response cache keyed by content hash."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path


class LLMCache:
    """Caches LLM responses in a local SQLite database.

    Cache key = SHA-256 of (provider, model, prompt, system, temperature).
    """

    def __init__(self, db_path: str | Path = ".llm_cache.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    @staticmethod
    def _make_key(provider: str, model: str, prompt: str, system: str, temperature: float) -> str:
        payload = json.dumps(
            {"provider": provider, "model": model, "prompt": prompt, "system": system, "temp": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(
        self, provider: str, model: str, prompt: str, system: str = "", temperature: float = 0.0
    ) -> tuple[str, int, int] | None:
        key = self._make_key(provider, model, prompt, system, temperature)
        row = self._conn.execute(
            "SELECT response, input_tokens, output_tokens FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return row[0], row[1], row[2]
        return None

    def put(
        self,
        provider: str,
        model: str,
        prompt: str,
        system: str,
        temperature: float,
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        key = self._make_key(provider, model, prompt, system, temperature)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache (key, response, input_tokens, output_tokens)
            VALUES (?, ?, ?, ?)
            """,
            (key, response, input_tokens, output_tokens),
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        self._conn.close()
