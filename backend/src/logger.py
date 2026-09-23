"""
Phase 6: Interaction logging to SQLite.

Every question + retrieved chunks + answer is written to logs/interactions.db.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

load_log = logging.getLogger(__name__)

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DB_PATH = LOG_DIR / "interactions.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    retrieved_json  TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    cited_sections  TEXT    NOT NULL,
    confidence      REAL    NOT NULL,
    latency_s       REAL    NOT NULL
);
"""


def _get_conn() -> sqlite3.Connection:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def log_interaction(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    answer: str,
    cited_sections: list[str],
    confidence: float,
    latency_s: float,
) -> None:
    """Append one interaction record to the SQLite log."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO interactions
               (ts, question, retrieved_json, answer, cited_sections, confidence, latency_s)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                json.dumps(retrieved_chunks),
                answer,
                json.dumps(cited_sections),
                confidence,
                latency_s,
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        load_log.warning(f"Failed to log interaction: {exc}")
