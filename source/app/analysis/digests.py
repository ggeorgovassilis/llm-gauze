"""Ready-made digests for the questions that recur when reading recordings.

Each digest consumes a stream of records (already read/filtered/projected as
needed) and emits one compact, machine-readable summary:

``stuck``
    Why is a run stuck?  Abort timeline, per-request abort/resubmission/retry
    counts, stall false-positives, and retry storms.

``endpoint``
    What problems talking to the LLM endpoint?  Transport-error and HTTP-status
    breakdowns, retryable classification, and the per-request retry
    distribution.

``remediation``
    Which errors occurred, which rung fired, and did the ladder succeed or hit
    a cap?  Errors joined with pass records and the final ladder summary.

``narrative``
    What happened in domain terms?  The ordered tool/MCP call sequence (calls
    joined to their results) and a prose-ish reconstruction of the turn.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Callable, Iterable

from app.analysis.classify import (
    content_text,
    diagnosis,
    error_entry,
    is_abort,
    is_attempt,
    is_final,
    is_pass,
    nested,
    normalise_tool_call,
    pass_rung,
    record_kind,
    record_rungs,
    request_messages,
    response_messages,
)

_MAX_SUMMARY_TEXT = 200


def _bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _counts(counter: Counter) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter, key=lambda item: (-counter[item], item))}


def _timestamp(record: dict) -> float:
    value = record.get("timestamp")
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clip_text(value: Any, limit: int) -> Any:
    """Truncate long strings (and nested argument values) for compact output."""
    if not isinstance(value, str):
        return value
    if limit <= 0 or len(value) <= limit:
        return value
    return value[:limit] + "…"


def _chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _changes(value: Any) -> list:
    """Normalise a record's ``tool_repair``/``think_cleanup`` tail to a list."""
    if isinstance(value, list):
        return value
    return [value] if value else []


def _unique_changes(values: Iterable[Any]) -> list:
    """Flatten repeated changes across attempts, preserving first occurrence."""
    seen: set[str] = set()
    unique = []
    for value in values:
        signature = json.dumps(value, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(value)
    return unique


def _group_records(records: Iterable[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        request_id = record.get("request_id")
        key = request_id if isinstance(request_id, str) and request_id else "unknown"
        groups.setdefault(key, []).append(record)
    return groups


def _attempt_count(group: list[dict]) -> int:
    attempts = [record for record in group if is_attempt(record)]
    highest = 0
    for record in attempts:
        try:
            highest = max(highest, int(record.get("attempt") or 1))
        except (TypeError, ValueError):
            continue
    return max(len(attempts), highest)


def _request_summary(request_id: str, group: list[dict]) -> dict:
    aborts = [record for record in group if is_abort(record)]
    passes = [record for record in group if is_pass(record)]
    attempts = _attempt_count(group)
    rung_counts: Counter = Counter()
    for record in passes:
        rung_counts[pass_rung(record) or "unknown"] += 1
    abort_kinds: Counter = Counter()
    for record in aborts:
        abort_kinds[_bucket(record.get("abort_kind"))] += 1
    last = group[-1] if group else {}
    return {
        "request_id": request_id,
        "started": _timestamp(group[0]) if group else None,
        "ended": _timestamp(last) if group else None,
        "last_outcome": last.get("outcome"),
        "aborts": len(aborts),
        "abort_kinds": _counts(abort_kinds),
        "attempts": attempts,
        "retries": max(attempts - 1, 0),
        "resubmissions": sum(rung_counts.values()),
        "rungs": _counts(rung_counts),
        "ended_with_abort": bool(aborts) and record_kind(last) == "abort",
    }


def stuck(
    records: Iterable[dict], *, retry_storm_threshold: int = 1, include_idle: bool = False
) -> dict:
    """Why is this run stuck?  Abort timeline + per-request counts.

    By default only requests with an abort, a resubmission or a retry are
    listed under ``requests`` (the totals still count every request); pass
    ``include_idle=True`` to list idle requests too.
    """
    records = list(records)
    groups = _group_records(records)

    timeline: list[dict] = []
    abort_kinds: Counter = Counter()
    stalled_with_output: list[dict] = []
    for record in records:
        if not is_abort(record):
            continue
        abort_kind = record.get("abort_kind")
        abort_kinds[_bucket(abort_kind)] += 1
        entry = {
            "timestamp": _timestamp(record),
            "request_id": record.get("request_id") or "unknown",
            "attempt": record.get("attempt"),
            "abort_kind": abort_kind,
            "abort_stream": record.get("abort_stream"),
            "abort_reason": record.get("abort_reason"),
            "abort_details": record.get("abort_details"),
            "partial_content_chars": _chars(record.get("partial_content")),
            "partial_reasoning_chars": _chars(record.get("partial_reasoning")),
            "partial_tool_calls": len(record.get("partial_tool_calls") or []),
        }
        timeline.append(entry)
        if abort_kind == "stalled":
            signals = [
                field
                for field in ("partial_content", "partial_reasoning", "partial_tool_calls")
                if _has_value(record.get(field))
            ]
            if signals:
                stalled_with_output.append({**entry, "signals": signals})

    requests = [_request_summary(request_id, group) for request_id, group in groups.items()]
    if not include_idle:
        requests = [
            request
            for request in requests
            if request["aborts"] or request["resubmissions"] or request["retries"]
        ]
    retry_storms = sorted(
        (request for request in requests if request["retries"] >= retry_storm_threshold),
        key=lambda request: (-request["retries"], request["request_id"]),
    )
    passes = sum(1 for record in records if is_pass(record))
    return {
        "digest": "stuck",
        "records": len(records),
        "totals": {
            "requests": len(groups),
            "aborts": sum(abort_kinds.values()),
            "passes": passes,
            "attempts": sum(1 for record in records if is_attempt(record)),
        },
        "abort_kinds": _counts(abort_kinds),
        "timeline": timeline,
        "stall_false_positives": stalled_with_output,
        "retry_storms": retry_storms,
        "requests": requests,
    }


def endpoint(records: Iterable[dict], *, include_idle: bool = False) -> dict:
    """What problems talking to the LLM endpoint?

    By default only requests with a retry or an error are listed under
    ``requests``; pass ``include_idle=True`` to list every request.
    """
    records = list(records)
    attempts = [record for record in records if is_attempt(record)]
    errors = [record for record in attempts if record.get("outcome") != "success"]

    errors_by_outcome: Counter = Counter()
    errors_by_type: Counter = Counter()
    upstream_by_reason: Counter = Counter()
    http_by_status: Counter = Counter()
    http_by_reason: Counter = Counter()
    context_by_reason: Counter = Counter()
    retryable_counts: Counter = Counter()
    for record in errors:
        outcome = _bucket(record.get("outcome"))
        errors_by_outcome[outcome] += 1
        error = error_entry(record)
        if error:
            errors_by_type[_bucket(error.get("type"))] += 1
        reason = diagnosis(record).get("reason")
        if record.get("outcome") == "upstream_failure" and reason:
            upstream_by_reason[_bucket(reason)] += 1
        if record.get("outcome") == "http_error":
            http_by_status[_bucket(record.get("status"))] += 1
            if reason:
                http_by_reason[_bucket(reason)] += 1
        if record.get("outcome") == "context_window_exceeded" and reason:
            context_by_reason[_bucket(reason)] += 1
        if diagnosis(record):
            retryable_counts[_bucket(bool(diagnosis(record).get("retryable")))] += 1

    groups = _group_records(records)
    retry_distribution: Counter = Counter()
    requests: list[dict] = []
    for request_id, group in groups.items():
        attempt_count = _attempt_count(group)
        if not attempt_count:
            continue
        retry_distribution[_bucket(max(attempt_count - 1, 0))] += 1
        group_errors = [
            record for record in group if is_attempt(record) and record.get("outcome") != "success"
        ]
        requests.append(
            {
                "request_id": request_id,
                "attempts": attempt_count,
                "retries": max(attempt_count - 1, 0),
                "error_outcomes": _counts(
                    Counter(_bucket(record.get("outcome")) for record in group_errors)
                ),
                "statuses": _counts(
                    Counter(_bucket(record.get("status")) for record in group_errors)
                ),
                "reasons": _counts(
                    Counter(
                        _bucket(diagnosis(record).get("reason"))
                        for record in group_errors
                        if diagnosis(record).get("reason")
                    )
                ),
            }
        )

    if not include_idle:
        requests = [
            request for request in requests if request["retries"] or request["error_outcomes"]
        ]

    return {
        "digest": "endpoint",
        "records": len(records),
        "totals": {
            "attempts": len(attempts),
            "errors": len(errors),
            "successful_attempts": len(attempts) - len(errors),
        },
        "error_outcomes": _counts(errors_by_outcome),
        "upstream_failures": {
            "count": sum(1 for record in errors if record.get("outcome") == "upstream_failure"),
            "by_error_type": _counts(errors_by_type),
            "by_reason": _counts(upstream_by_reason),
        },
        "http_errors": {
            "count": sum(1 for record in errors if record.get("outcome") == "http_error"),
            "by_status": _counts(http_by_status),
            "by_reason": _counts(http_by_reason),
        },
        "context_window_exceeded": sum(
            1 for record in errors if record.get("outcome") == "context_window_exceeded"
        ),
        "context_window_reasons": _counts(context_by_reason),
        "retryable": _counts(retryable_counts),
        "retry_distribution": _counts(retry_distribution),
        "requests": sorted(
            requests, key=lambda request: (-request["retries"], request["request_id"])
        ),
    }


def _error_summary(record: dict) -> dict:
    return {
        "timestamp": _timestamp(record),
        "outcome": record.get("outcome"),
        "attempt": record.get("attempt"),
        "status": record.get("status"),
        "error_type": error_entry(record).get("type"),
        "reason": diagnosis(record).get("reason"),
        "retryable": diagnosis(record).get("retryable"),
    }


def _pass_summary(record: dict) -> dict:
    rung = pass_rung(record)
    block = nested(record, rung) if rung else {}
    entry = {
        "timestamp": _timestamp(record),
        "attempt": record.get("attempt"),
        "rung": rung,
        "mode": block.get("mode"),
        "resource_chars": block.get("resource_chars"),
        "resource_lines": block.get("resource_lines"),
        "has_pre_remediation_body": record.get("pre_remediation_body") is not None,
        "has_resubmitted_body": block.get("resubmitted_body") is not None,
    }
    return entry


def _final_summary(record: dict) -> dict:
    rungs = {}
    for rung in record_rungs(record):
        block = nested(record, rung)
        rungs[rung] = {
            "attempts": block.get("attempts"),
            "outcome": block.get("outcome"),
        }
    return {
        "rounds": record.get("rounds", 0) or 0,
        "remediation_seconds": record.get("remediation_seconds"),
        "remediation_exhausted": record.get("remediation_exhausted"),
        "rungs": rungs,
    }


def remediation(records: Iterable[dict], *, include_idle: bool = False) -> dict:
    """Which errors occurred and how were they remedied?

    By default only requests that had an error, a remediation pass or a
    non-trivial final ladder summary are listed; totals still count every
    request.  Pass ``include_idle=True`` to list every request.
    """
    records = list(records)
    groups = _group_records(records)

    error_reasons: Counter = Counter()
    rung_counts: Counter = Counter()
    cap_trips: Counter = Counter({"rounds": 0, "time": 0})
    exhausted_rungs: Counter = Counter()

    all_requests: list[dict] = []
    for request_id, group in groups.items():
        errors = [
            record for record in group if is_attempt(record) and record.get("outcome") != "success"
        ]
        passes = [record for record in group if is_pass(record)]
        final_record = next((record for record in reversed(group) if is_final(record)), None)

        for record in errors:
            error_reasons[_bucket(diagnosis(record).get("reason"))] += 1
        for record in passes:
            rung_counts[pass_rung(record) or "unknown"] += 1

        final = _final_summary(final_record) if final_record is not None else None
        if final is not None:
            exhausted = final.get("remediation_exhausted")
            if exhausted in ("rounds", "time"):
                cap_trips[str(exhausted)] += 1
            for rung, block in final["rungs"].items():
                if block.get("outcome") == "exhausted":
                    exhausted_rungs[rung] += 1

        chain = [
            f"{_bucket(record.get('outcome'))}: "
            f"{diagnosis(record).get('reason') or record.get('status') or 'error'}"
            for record in errors
        ]
        chain.extend(pass_rung(record) or "unknown" for record in passes)
        if final_record is not None:
            chain.append("success")
        elif errors:
            chain.append(_bucket(group[-1].get("outcome")))

        if final_record is not None:
            status = "remediated" if errors or passes else "clean"
        elif errors:
            status = "unresolved"
        else:
            status = "clean"

        all_requests.append(
            {
                "request_id": request_id,
                "status": status,
                "errors": [_error_summary(record) for record in errors],
                "passes": [_pass_summary(record) for record in passes],
                "final": final,
                "chain": chain,
            }
        )

    totals = {
        "requests": len(groups),
        "errors": sum(len(request["errors"]) for request in all_requests),
        "passes": sum(len(request["passes"]) for request in all_requests),
        "resolved": sum(
            1 for request in all_requests if request["status"] in ("clean", "remediated")
        ),
        "unresolved": sum(1 for request in all_requests if request["status"] == "unresolved"),
    }
    requests = all_requests
    if not include_idle:
        requests = [
            request
            for request in all_requests
            if request["errors"]
            or request["passes"]
            or (
                request["final"] is not None
                and (
                    request["final"]["rounds"]
                    or request["final"]["remediation_exhausted"]
                    or any(
                        block.get("outcome") == "exhausted"
                        for block in request["final"]["rungs"].values()
                    )
                )
            )
        ]
    return {
        "digest": "remediation",
        "records": len(records),
        "totals": totals,
        "error_reasons": _counts(error_reasons),
        "rung_counts": _counts(rung_counts),
        "cap_trips": {key: cap_trips[key] for key in ("rounds", "time")},
        "exhausted_rungs": _counts(exhausted_rungs),
        "requests": requests,
    }


def _conversation_messages(group: list[dict]) -> list[dict]:
    """Merge a request's message history across attempts, preserving order."""
    messages: list[dict] = []
    seen: set[str] = set()

    def add(candidates: Iterable[dict]) -> None:
        for message in candidates:
            signature = json.dumps(message, sort_keys=True, default=str)
            if signature in seen:
                continue
            seen.add(signature)
            messages.append(message)

    for record in group:
        add(request_messages(record))
    for record in group:
        add(response_messages(record))
    return messages


def _ordered_tool_calls(messages: list[dict], limit: int) -> list[dict]:
    """Calls in conversation order, each joined to its result when present."""
    calls: list[dict] = []
    by_id: dict[str, dict] = {}
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list):
                continue
            for raw_call in raw_calls:
                call = normalise_tool_call(raw_call)
                if call is None:
                    continue
                entry = {
                    "id": call["id"],
                    "name": call["name"],
                    "arguments": _clip_text(call["arguments"], limit),
                    "result": None,
                }
                calls.append(entry)
                if isinstance(call["id"], str):
                    by_id[call["id"]] = entry
        elif role == "tool":
            call_id = message.get("tool_call_id")
            result = _clip_text(content_text(message.get("content")), limit)
            target = by_id.get(call_id) if isinstance(call_id, str) else None
            if target is not None:
                target["result"] = result
            else:
                calls.append(
                    {
                        "id": call_id,
                        "name": message.get("name"),
                        "arguments": None,
                        "result": result,
                    }
                )
    return calls


def _conversation_events(messages: list[dict], limit: int) -> list[dict]:
    events = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            events.append(
                {"kind": "user", "content": _clip_text(content_text(message.get("content")), limit)}
            )
        elif role == "assistant":
            raw_calls = message.get("tool_calls")
            calls = []
            if isinstance(raw_calls, list):
                for raw_call in raw_calls:
                    call = normalise_tool_call(raw_call)
                    if call is not None:
                        calls.append(
                            {
                                "name": call["name"],
                                "arguments": _clip_text(call["arguments"], limit),
                            }
                        )
            events.append(
                {
                    "kind": "assistant",
                    "content": _clip_text(content_text(message.get("content")), limit),
                    "tool_calls": calls,
                }
            )
        elif role == "tool":
            events.append(
                {
                    "kind": "tool_result",
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message.get("name"),
                    "content": _clip_text(content_text(message.get("content")), limit),
                }
            )
    return events


def _record_events(group: list[dict]) -> list[dict]:
    events = []
    for record in group:
        kind = record_kind(record)
        if kind == "abort":
            events.append(
                {
                    "timestamp": _timestamp(record),
                    "kind": "abort",
                    "abort_kind": record.get("abort_kind"),
                    "abort_reason": record.get("abort_reason"),
                    "partial_content": _clip_text(record.get("partial_content"), _MAX_SUMMARY_TEXT),
                }
            )
        elif kind == "pass":
            events.append(
                {
                    "timestamp": _timestamp(record),
                    "kind": "remediation_pass",
                    "rung": pass_rung(record),
                }
            )
        elif kind == "attempt" and record.get("outcome") != "success":
            events.append(
                {
                    "timestamp": _timestamp(record),
                    "kind": "error",
                    "outcome": record.get("outcome"),
                    "status": record.get("status"),
                    "reason": diagnosis(record).get("reason"),
                }
            )
        elif kind == "final":
            events.append(
                {
                    "timestamp": _timestamp(record),
                    "kind": "final",
                    "rounds": record.get("rounds", 0) or 0,
                    "remediation_exhausted": record.get("remediation_exhausted"),
                }
            )
    return events


def _narrative_summary(
    user_messages: list[str],
    tool_calls: list[dict],
    final_record: dict | None,
    last_record: dict,
) -> str:
    question = _clip_text(user_messages[-1], _MAX_SUMMARY_TEXT) if user_messages else ""
    names = []
    for call in tool_calls:
        name = call.get("name")
        if name and name not in names:
            names.append(name)
    parts = []
    if question:
        parts.append(f'User asked "{question}"')
    else:
        parts.append("No user message recorded")
    if names:
        parts.append(f"{len(tool_calls)} tool call(s): {', '.join(names)}")
    else:
        parts.append("no tool calls")
    outcome = last_record.get("outcome")
    if final_record is not None:
        rounds = final_record.get("rounds", 0) or 0
        suffix = f" after {rounds} remediation round(s)" if rounds else ""
        parts.append(f"outcome: {outcome}{suffix}")
    else:
        parts.append(f"outcome: {outcome}")
    return "; ".join(parts)


def _turn_messages(messages: list[dict]) -> list[dict]:
    """The messages belonging to the turn being analysed (from the last user message)."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    return messages[last_user:] if last_user >= 0 else messages


def narrative(
    records: Iterable[dict], *, max_text: int = 500, include_history: bool = False
) -> dict:
    """What happened in domain terms?  Ordered tool calls + reconstruction.

    By default the reconstruction covers the current turn (from the last user
    message) so a whole-recording narrative stays compact; ``include_history``
    includes the full message history carried in ``request_body``.
    """
    records = list(records)
    groups = _group_records(records)
    requests: list[dict] = []
    all_calls = []
    seen_calls: set[str] = set()
    for request_id, group in groups.items():
        messages = _conversation_messages(group)
        turn = _turn_messages(messages)
        tool_calls = _ordered_tool_calls(turn, max_text)
        for record in group:
            if not is_abort(record):
                continue
            partial = record.get("partial_tool_calls")
            if not isinstance(partial, list):
                continue
            for raw_call in partial:
                call = normalise_tool_call(raw_call)
                if call is None:
                    continue
                tool_calls.append(
                    {
                        "id": call["id"],
                        "name": call["name"],
                        "arguments": _clip_text(call["arguments"], max_text),
                        "result": None,
                        "partial": True,
                    }
                )
        for call in tool_calls:
            signature = json.dumps(
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "arguments": call.get("arguments"),
                },
                sort_keys=True,
                default=str,
            )
            if signature in seen_calls:
                continue
            seen_calls.add(signature)
            all_calls.append({"request_id": request_id, **call})
        user_messages = [
            content_text(message.get("content")) or ""
            for message in messages
            if message.get("role") == "user"
        ]
        final_record = next((record for record in reversed(group) if is_final(record)), None)
        requests.append(
            {
                "request_id": request_id,
                "started": _timestamp(group[0]) if group else None,
                "ended": _timestamp(group[-1]) if group else None,
                "summary": _narrative_summary(
                    user_messages, tool_calls, final_record, group[-1] if group else {}
                ),
                "conversation": _conversation_events(
                    messages if include_history else turn, max_text
                ),
                "tool_calls": tool_calls,
                "timeline": _record_events(group),
                "tool_repairs": _unique_changes(
                    change for record in group for change in _changes(record.get("tool_repair"))
                ),
                "message_overflow": [
                    record.get("message_overflow")
                    for record in group
                    if record.get("message_overflow")
                ],
            }
        )
    return {
        "digest": "narrative",
        "records": len(records),
        "totals": {
            "requests": len(groups),
            "tool_calls": len(all_calls),
        },
        "requests": requests,
        "tool_calls": all_calls,
    }


DIGESTS: dict[str, Callable[..., dict]] = {
    "stuck": stuck,
    "endpoint": endpoint,
    "remediation": remediation,
    "narrative": narrative,
}


def run_digest(name: str, records: Iterable[dict], **options: Any) -> dict:
    """Run a named digest over ``records``."""
    try:
        digest = DIGESTS[name]
    except KeyError:
        raise ValueError(
            f"unknown digest {name!r}; choose from {', '.join(sorted(DIGESTS))}"
        ) from None
    return digest(records, **options)
