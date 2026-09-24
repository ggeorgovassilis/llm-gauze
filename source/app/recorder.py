"""Recording of every request/response exchange.

The recorder is the single source of truth that future error-detection and
remediation modules will read from. It is deliberately kept simple and
append-only: each exchange is written as one JSON line to a JSONL file.
"""

import json
import logging
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("bandaid.recorder")


class Recorder:
    """Append-only recorder writing JSONL records to disk."""

    def __init__(self, record_path: str | Path) -> None:
        self.path = Path(record_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, entry: dict) -> None:
        """Persist a single exchange record.

        `entry` is a dict of any JSON-serialisable content. Common fields are
        set by the caller (method, path, bodies, status, error, ...).
        """
        entry.setdefault("id", uuid.uuid4().hex)
        entry.setdefault("timestamp", time.time())
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        logger.info(
            "recorded exchange %s %s %s",
            entry.get("id"),
            entry.get("method"),
            entry.get("path"),
        )
