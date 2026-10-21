"""Record classification and body decoding for the analysis toolkit.

The recorder's ``outcome`` discriminator is deliberately overloaded (see
``docs/record-schema.md``): ``success`` is a plain successful attempt, an
extract/coast/empty-stop pass, *or* the final delivery; ``loop_aborted`` is a
loop abort *or* a loop-retry pass; ``runaway_reasoning_aborted`` is a runaway
abort *or* a runaway pass.  Every analysis stage routes through the helpers in
this module instead of key-sniffing, so the disambiguation rules live in one
place.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator

#: Outcomes that describe a single upstream attempt.
ATTEMPT_OUTCOMES = frozenset(
    {"success", "http_error", "upstream_failure", "context_window_exceeded"}
)

#: Outcomes that describe a content-watchdog abort when ``abort_kind`` is set.
ABORT_OUTCOMES = frozenset({"loop_aborted", "stalled", "runaway_reasoning_aborted"})

#: The five remediation rungs, in ladder order.
RUNGS: tuple[str, ...] = ("extract", "coast", "empty_stop", "runaway", "loop_retry")

#: Record kinds produced by :func:`record_kind`.
KINDS = ("attempt", "abort", "pass", "final", "unknown")


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def nested(record: dict, rung: str) -> dict:
    """Return a record's nested rung block, or an empty dict."""
    return _as_dict(record.get(rung))


def record_rungs(record: dict) -> list[str]:
    """All rung blocks present on a record (pass records *and* final summaries)."""
    return [rung for rung in RUNGS if isinstance(record.get(rung), dict)]


def triggered_rungs(record: dict) -> list[str]:
    """Rungs whose nested block says ``outcome: "triggered"``."""
    return [rung for rung in RUNGS if nested(record, rung).get("outcome") == "triggered"]


def is_abort(record: dict) -> bool:
    """True for a watchdog abort record (loop/stall/runaway)."""
    return record.get("abort_kind") is not None


def is_pass(record: dict) -> bool:
    """True for a remediation-pass record (one rung fired).

    Pass records never deliver a body: the turn was discarded and re-submitted.
    The final-success record also carries nested rung blocks, but it delivers a
    ``response_body``, which is what disambiguates the two.
    """
    if record.get("response_body") is not None:
        return False
    return bool(triggered_rungs(record)) or "pre_remediation_body" in record


def is_final(record: dict) -> bool:
    """True for the terminal success record that carries the ladder summary."""
    if record.get("outcome") != "success" or record.get("response_body") is None:
        return False
    if any(key in record for key in ("rounds", "remediation_seconds", "remediation_exhausted")):
        return True
    return any(
        "attempts" in nested(record, rung) or "outcome" in nested(record, rung)
        for rung in RUNGS
        if isinstance(record.get(rung), dict)
    )


def is_attempt(record: dict) -> bool:
    """True for a raw upstream-attempt record."""
    return (
        record.get("outcome") in ATTEMPT_OUTCOMES and not is_pass(record) and not is_final(record)
    )


def record_kind(record: dict) -> str:
    """Classify a record as ``attempt``/``abort``/``pass``/``final``/``unknown``."""
    if is_abort(record):
        return "abort"
    if is_pass(record):
        return "pass"
    if is_final(record):
        return "final"
    if is_attempt(record):
        return "attempt"
    return "unknown"


def pass_rung(record: dict) -> str | None:
    """The rung a pass record belongs to, or ``None``."""
    if not is_pass(record):
        return None
    rungs = triggered_rungs(record)
    return rungs[0] if rungs else None


def diagnosis(record: dict) -> dict:
    """The ``diagnosis`` tail (``{retryable, reason, code?}``) or ``{}``."""
    return _as_dict(record.get("diagnosis"))


def diagnosis_reason(record: dict) -> str | None:
    """``diagnosis.reason`` as a string, when present."""
    reason = diagnosis(record).get("reason")
    return reason if isinstance(reason, str) else None


def error_entry(record: dict) -> dict:
    """The ``error`` tail (``{type, message, traceback?}``) or ``{}``."""
    return _as_dict(record.get("error"))


def error_text(record: dict) -> str:
    """A searchable ``type: message`` string for a record's transport error."""
    error = error_entry(record)
    if not error:
        return ""
    return f"{error.get('type', '')}: {error.get('message', '')}".strip(": ")


def payload_objects(value: Any) -> Iterator[dict]:
    """Yield JSON objects from a JSON document or an SSE text stream.

    Recorded bodies are normally JSON strings, but a streamed response can be
    retained as raw SSE.  Tolerating both keeps the narrative digest working
    for every recording generation.
    """
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(value, str) or not value.strip():
        return
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        decoded = None
    if decoded is not None:
        if isinstance(decoded, dict):
            yield decoded
        elif isinstance(decoded, list):
            for item in decoded:
                if isinstance(item, dict):
                    yield item
        return
    for line in value.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            item = json.loads(data)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(item, dict):
            yield item


def decode_body(value: Any) -> dict:
    """Decode a recorded body into a JSON object (empty dict when unparseable)."""
    for item in payload_objects(value):
        return item
    return {}


def _message_lists(value: Any) -> Iterator[dict]:
    """Yield messages from a request body, a response body, or plain text."""
    for item in payload_objects(value):
        messages = item.get("messages")
        if isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict):
                    yield message
        choices = item.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if isinstance(message, dict):
                    yield message
        if "role" in item:
            yield item


def request_messages(record: dict) -> list[dict]:
    """The client's message history from ``request_body``."""
    payload = decode_body(record.get("request_body"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]


def response_messages(record: dict) -> list[dict]:
    """Assistant messages carried by ``response_body`` (or SSE payloads)."""
    messages = []
    for message in _message_lists(record.get("response_body")):
        if message.get("role") == "assistant" or "tool_calls" in message:
            messages.append(message)
    return messages


def content_text(content: Any) -> str | None:
    """Flatten OpenAI message content (a string or a list of content parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts) if parts else None
    return None


def normalise_tool_call(call: Any) -> dict | None:
    """Normalise an OpenAI tool call to ``{id, name, arguments}``."""
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
        call_id = call.get("id")
    else:
        name = call.get("name")
        arguments = call.get("arguments")
        call_id = call.get("id") or call.get("tool_call_id")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    return {"id": call_id, "name": name, "arguments": arguments}


def tool_calls(record: dict) -> list[dict]:
    """Every tool call visible on a record, in conversation order.

    Scans the client's message history, the delivered response, the discarded
    ``pre_remediation_body``, and the preserved ``partial_tool_calls`` of an
    abort record.
    """
    calls: list[dict] = []
    for key in ("request_body", "response_body", "pre_remediation_body"):
        for message in _message_lists(record.get(key)):
            raw_calls = message.get("tool_calls")
            if not isinstance(raw_calls, list):
                continue
            for raw_call in raw_calls:
                call = normalise_tool_call(raw_call)
                if call is not None:
                    calls.append(call)
    partial = record.get("partial_tool_calls")
    if isinstance(partial, list):
        for raw_call in partial:
            call = normalise_tool_call(raw_call)
            if call is not None:
                calls.append(call)
    return calls


def has_tool_calls(record: dict) -> bool:
    """True when any tool call is visible on the record."""
    return bool(tool_calls(record))


def iter_pairs(records: Iterable[dict]) -> Iterator[tuple[str, dict]]:
    """Yield ``(request_id, record)`` pairs, using ``unknown`` when absent."""
    for record in records:
        request_id = record.get("request_id")
        yield (request_id if isinstance(request_id, str) and request_id else "unknown"), record
