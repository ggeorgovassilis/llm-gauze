"""Read recorder JSONL files back for analysis.

The recorder rotates ``records.jsonl`` aside on restart, appending a timestamp
to the rotated name (``records-20260821-175741.jsonl``).  An analysis must see
the whole history — the rotated files *and* the live one — in chronological
order, while tolerating a torn final line left by a hard kill.

The reader discovers the files next to a live record path, lazily merges their
records by ``timestamp`` (each file is already chronological), and skips any
line that is not a JSON object.  Zero-filled tails (a kill mid-write on some
filesystems) are stripped before parsing.
"""

from __future__ import annotations

import heapq
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

_TIMESTAMP_RE = re.compile(r"-(\d{8}-\d{6})(?:-(\d+))?$")


@dataclass(frozen=True)
class RecordFile:
    """One JSONL file participating in a recording."""

    path: Path
    #: True for the live ``records.jsonl``, which sorts after rotated files.
    live: bool = False


def _rotate_order(path: Path) -> tuple[str, int]:
    """A sortable key from a rotated file's timestamp suffix.

    Files that do not carry a parseable timestamp (hand-renamed or copied)
    sort first, then by name, rather than being silently ignored.
    """
    match = _TIMESTAMP_RE.search(path.stem)
    if match is None:
        return ("", 0)
    counter = int(match.group(2) or 0)
    return (match.group(1), counter)


def discover_record_files(target: str | Path, *, include_rotated: bool = True) -> list[RecordFile]:
    """Discover the live record file and its rotated siblings.

    ``target`` may be the live file itself or the directory that holds it.  The
    returned list is ordered oldest rotated file first and the live file last.
    """
    path = Path(target)
    if path.is_dir():
        live = path / "records.jsonl"
    else:
        live = path
    directory = live.parent
    files: list[Path] = []
    if include_rotated and directory.exists():
        prefix = f"{live.stem}-"
        for candidate in directory.iterdir():
            if candidate == live or not candidate.is_file():
                continue
            if candidate.suffix != live.suffix:
                continue
            if not candidate.stem.startswith(prefix):
                continue
            files.append(candidate)
    files.sort(key=lambda item: (_rotate_order(item), item.name))
    discovered = [RecordFile(path=item, live=False) for item in files]
    if live.exists():
        discovered.append(RecordFile(path=live, live=True))
    return discovered


def _parse_line(line: str) -> dict | None:
    """Parse one JSONL line, returning ``None`` for blank/corrupt/zero-filled data."""
    cleaned = line.strip().strip("\x00").strip()
    if not cleaned:
        return None
    try:
        record = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _timestamp(record: dict) -> float:
    value = record.get("timestamp")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_file(path: Path) -> Iterator[tuple[float, dict]]:
    """Yield ``(timestamp, record)`` pairs from one file in write order.

    ``errors="replace"`` keeps a binary or zero-filled tail from aborting the
    read; unparseable lines (including a truncated final line) are skipped.
    """
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                record = _parse_line(line)
                if record is not None:
                    yield _timestamp(record), record
    except OSError:
        # A file can vanish between discovery and opening; analysis continues
        # with whatever remains rather than failing wholesale.
        return


def iter_records(target: str | Path, *, include_rotated: bool = True) -> Iterator[dict]:
    """Stream the recording's records in chronological order.

    Records from rotated files and the live file are merged by ``timestamp``.
    The merge is lazy: no call reads a whole file into memory.
    """
    files = discover_record_files(target, include_rotated=include_rotated)
    streams = [_iter_file(record_file.path) for record_file in files]
    if not streams:
        return
    for _, record in heapq.merge(*streams, key=lambda item: item[0]):
        yield record


def read_records(target: str | Path, *, include_rotated: bool = True) -> list[dict]:
    """Read every record from ``target`` into a list (see :func:`iter_records`)."""
    return list(iter_records(target, include_rotated=include_rotated))
