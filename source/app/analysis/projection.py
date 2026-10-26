"""Project records down to the fields a question needs.

Full recorded bodies are large (a single ``request_body`` can carry a whole
conversation), and an agent usually needs a handful of fields.  Projection
keeps the pipeline's output small without discarding anything upstream: the
same stream can be projected differently for different questions.

Fields are dotted paths (``diagnosis.reason``).  A plain field keeps its name;
``alias=path`` renames it.  A missing path is emitted as ``null`` so downstream
consumers see a stable schema.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping

_MISSING = object()


def _lookup(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def project_record(record: Mapping[str, Any], fields: Iterable[str] | Mapping[str, str]) -> dict:
    """Reduce one record to the requested fields."""
    if isinstance(fields, Mapping):
        pairs = list(fields.items())
    else:
        pairs = [(field, field) for field in fields]
    projected = {}
    for name, path in pairs:
        value = _lookup(record, path)
        projected[name] = None if value is _MISSING else value
    return projected


def project(
    records: Iterable[Mapping[str, Any]], fields: Iterable[str] | Mapping[str, str]
) -> Iterator[dict]:
    """Yield each record reduced to ``fields``."""
    for record in records:
        yield project_record(record, fields)
