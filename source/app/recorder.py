"""Recording of every request/response exchange.

The recorder is the single source of truth that future error-detection and
remediation modules will read from. It is deliberately kept simple and
append-only: each exchange is written as one JSON line to a JSONL file.
"""

import asyncio
import datetime
import json
import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger("llm_gauze.recorder")


class Recorder:
    """Append-only recorder writing JSONL records to disk.

    Each run starts a fresh record file: if one already exists at the target
    path it is first rotated aside (timestamp appended to its name) so a
    restart never appends to — or overwrites — the previous run's data.
    """

    def __init__(self, record_path: str | Path) -> None:
        self.path = Path(record_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_existing()
        # Serialises the appends so concurrent records still land in call
        # order (asyncio.Lock is FIFO-fair across waiters).
        self._write_lock = asyncio.Lock()

    def _rotate_existing(self) -> None:
        """Move a pre-existing record file aside, appending a timestamp."""
        if not self.path.exists():
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        rotated = self.path.with_name(f"{self.path.stem}-{stamp}{self.path.suffix}")
        # Collision (same second) is vanishingly unlikely, but avoid clobbering.
        counter = 1
        while rotated.exists():
            rotated = self.path.with_name(f"{self.path.stem}-{stamp}-{counter}{self.path.suffix}")
            counter += 1
        self.path.rename(rotated)
        logger.info("rotated existing record file %s -> %s", self.path.name, rotated.name)

    async def record(self, entry: dict) -> None:
        """Persist a single exchange record.

        `entry` is a dict of any JSON-serialisable content. Common fields are
        set by the caller (method, path, bodies, status, error, ...).

        The blocking file write runs in a worker thread (``asyncio.to_thread``)
        so it never stalls the event loop. The ``asyncio.Lock`` serialises the
        writes, so records are appended to the JSONL file in the order
        ``record`` was called, preserving chronological order under
        concurrency.
        """
        entry.setdefault("id", uuid.uuid4().hex)
        entry.setdefault("timestamp", time.time())
        line = json.dumps(entry, ensure_ascii=False, default=str)
        async with self._write_lock:
            await asyncio.to_thread(self._append, line)

        logger.info(
            "recorded exchange %s %s %s",
            entry.get("id"),
            entry.get("method"),
            entry.get("path"),
        )

    def _append(self, line: str) -> None:
        """Append one serialised record to the JSONL file (worker thread).

        Callers serialise through ``self._write_lock``, so this is only ever
        run one write at a time.
        """
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
