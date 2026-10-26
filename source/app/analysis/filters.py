"""Predicate filters over recorded exchanges.

A filter is a plain conjunction of the questions the agent asks repeatedly:
which outcome, which HTTP status, which transport error, which watchdog
verdict, which path/query, which time window, which remediation rung, and
whether the turn carried tool calls.  ``kind`` exposes the disambiguation from
:mod:`app.analysis.classify`, so ``kind=abort`` selects watchdog aborts while
``kind=pass`` selects the remediation passes that share their ``outcome``.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Callable, Iterable, Iterator

from app.analysis.classify import (
    diagnosis,
    diagnosis_reason,
    error_text,
    has_tool_calls,
    pass_rung,
    record_kind,
    record_rungs,
)

ERROR_OUTCOMES = frozenset({"http_error", "upstream_failure", "context_window_exceeded"})

_GLOB_CHARS = ("*", "?", "[")


def _values(value: Any) -> tuple:
    """Normalise a scalar-or-collection filter value to a tuple."""
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def _equals(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    try:
        return float(actual) == float(expected)
    except (TypeError, ValueError):
        return False


def _matches_scalar(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str) and any(char in expected for char in _GLOB_CHARS):
        return isinstance(actual, str) and fnmatch.fnmatchcase(actual, expected)
    return _equals(actual, expected)


def _matches_any(actual: Any, expected: Any) -> bool:
    values = _values(expected)
    if not values:
        return True
    return any(_matches_scalar(actual, value) for value in values)


def _contains(actual: Any, expected: Any) -> bool:
    """Case-insensitive substring match against a string (or any of several)."""
    if not isinstance(actual, str):
        return False
    haystack = actual.casefold()
    return any(str(value).casefold() in haystack for value in _values(expected))


def _matches_kind(record: dict, expected: Any) -> bool:
    values = _values(expected)
    if not values:
        return True
    kind = record_kind(record)
    for value in values:
        if value == "error" and record.get("outcome") in ERROR_OUTCOMES:
            return True
        if value == "success" and record.get("outcome") == "success":
            return True
        if kind == value:
            return True
    return False


def _matches_rung(record: dict, expected: Any) -> bool:
    values = {str(value) for value in _values(expected)}
    if not values:
        return True
    present = set(record_rungs(record))
    rung = pass_rung(record)
    if rung is not None:
        present.add(rung)
    return bool(present & values)


def _timestamp(record: dict) -> float | None:
    value = record.get("timestamp")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_predicate(
    *,
    outcome: Any = None,
    status: Any = None,
    attempt: Any = None,
    error: Any = None,
    reason: Any = None,
    abort_kind: Any = None,
    path: Any = None,
    query: Any = None,
    since: float | None = None,
    until: float | None = None,
    rung: Any = None,
    has_tool_calls: bool | None = None,
    request_id: Any = None,
    kind: Any = None,
    retryable: bool | None = None,
    predicate: Callable[[dict], bool] | None = None,
) -> Callable[[dict], bool]:
    """Build a predicate from the supported filter criteria.

    Omitted criteria match everything; supplied criteria are ANDed.  Passing a
    collection for a criterion ORs its values (``outcome=["http_error", ...]``).
    """

    def matches(record: dict) -> bool:
        if outcome is not None and not _matches_any(record.get("outcome"), outcome):
            return False
        if status is not None and not _matches_any(record.get("status"), status):
            return False
        if attempt is not None and not _matches_any(record.get("attempt"), attempt):
            return False
        if error is not None:
            haystack = f"{error_text(record)} {diagnosis_reason(record) or ''}"
            if not _contains(haystack, error):
                return False
        if reason is not None and not _contains(diagnosis_reason(record), reason):
            return False
        if abort_kind is not None and not _matches_any(record.get("abort_kind"), abort_kind):
            return False
        if path is not None and not _matches_any(record.get("path"), path):
            return False
        if query is not None and not _matches_any(record.get("query"), query):
            return False
        timestamp = _timestamp(record)
        if since is not None and (timestamp is None or timestamp < since):
            return False
        if until is not None and (timestamp is None or timestamp > until):
            return False
        if rung is not None and not _matches_rung(record, rung):
            return False
        if has_tool_calls is not None and has_tool_calls != _record_has_tool_calls(record):
            return False
        if request_id is not None and not _matches_any(record.get("request_id"), request_id):
            return False
        if kind is not None and not _matches_kind(record, kind):
            return False
        if retryable is not None:
            record_diagnosis = diagnosis(record)
            if "retryable" not in record_diagnosis:
                return False
            if bool(record_diagnosis.get("retryable")) != retryable:
                return False
        if predicate is not None and not predicate(record):
            return False
        return True

    return matches


def _record_has_tool_calls(record: dict) -> bool:
    return has_tool_calls(record)


def filter_records(records: Iterable[dict], **criteria: Any) -> Iterator[dict]:
    """Yield records matching every supplied criterion."""
    matches = build_predicate(**criteria)
    for record in records:
        if matches(record):
            yield record
