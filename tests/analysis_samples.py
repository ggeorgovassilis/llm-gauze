"""Realistic record fixtures for the analysis-toolkit tests.

The records deliberately mirror the shapes written by ``app/proxy.py`` (and
documented in ``docs/record-schema.md``), including the overloaded outcomes
that the analysis stages must disambiguate: success attempts vs
extract/coast/empty-stop passes vs the final success, and loop/runaway aborts
vs their pass records.
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def request_body(messages):
    return json.dumps({"model": "test-model", "messages": messages})


def response_body(content="done", tool_calls=None, reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return json.dumps(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        }
    )


def tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def base(request_id, timestamp, outcome, *, attempt=1, **extra):
    record = {
        "id": f"{request_id}-{timestamp}-{attempt}-{outcome}",
        "timestamp": timestamp,
        "request_id": request_id,
        "method": "POST",
        "path": "/v1/chat/completions",
        "query": "",
        "request_headers": {"content-type": "application/json"},
        "request_body": request_body([SYSTEM, {"role": "user", "content": "hello"}]),
        "attempt": attempt,
        "max_attempts": 3,
        "duration": 0.25,
        "outcome": outcome,
    }
    record.update(extra)
    return record


def attempt(request_id, timestamp, outcome, **extra):
    extra.setdefault("status", 200 if outcome == "success" else None)
    extra.setdefault("diagnosis", None)
    return base(request_id, timestamp, outcome, **extra)


def abort(request_id, timestamp, kind, outcome, *, partial_content=None, **extra):
    return base(
        request_id,
        timestamp,
        outcome,
        abort_kind=kind,
        abort_stream="content" if kind == "loop" else None,
        abort_reason=f"{kind} detected",
        abort_details={"window": 2048},
        partial_content=partial_content,
        partial_reasoning=None,
        partial_tool_calls=None,
        response_body=None,
        **extra,
    )


def pass_record(request_id, timestamp, rung, outcome, **extra):
    block = {"attempt": 1, "outcome": "triggered"}
    if rung in ("runaway", "loop_retry"):
        block["resubmitted_body"] = {"messages": [{"role": "user", "content": "retry"}]}
    if rung == "extract":
        block.update({"mode": "read", "resource_chars": 120, "resource_lines": 4})
    return base(
        request_id,
        timestamp,
        outcome,
        response_body=None,
        pre_remediation_body=response_body("discarded"),
        streamed=True,
        think_cleanup=None,
        diagnosis=None,
        **{rung: block},
        **extra,
    )


def final_success(
    request_id,
    timestamp,
    *,
    rounds=1,
    remediation_seconds=0.5,
    remediation_exhausted=None,
    extract=None,
    coast=None,
    empty_stop=None,
    response=None,
    **extra,
):
    return base(
        request_id,
        timestamp,
        "success",
        response_body=response or response_body("final answer"),
        streamed=True,
        think_cleanup=None,
        tool_repair=None,
        extract=extract,
        coast=coast,
        empty_stop=empty_stop,
        rounds=rounds,
        remediation_seconds=remediation_seconds,
        remediation_exhausted=remediation_exhausted,
        diagnosis=None,
        **extra,
    )


def tool_conversation_request(request_id, timestamp):
    messages = [
        SYSTEM,
        {"role": "user", "content": "find a cheap GPU"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "search", {"q": "gpu"})],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "search", "content": "found 3"},
        {"role": "user", "content": "buy the first one"},
    ]
    return base(
        request_id,
        timestamp,
        "success",
        request_body=request_body(messages),
        status=200,
        response_body=response_body(
            "Bought it", tool_calls=[tool_call("call_2", "buy", {"id": 1})]
        ),
        streamed=False,
        think_cleanup=None,
        tool_repair=[{"kind": "repaired_arguments", "index": 0}],
        extract=None,
        coast=None,
        empty_stop=None,
        message_overflow=None,
        diagnosis=None,
    )


def sample_records():
    """A recording exercising every record kind and both overloaded outcomes."""
    records = []

    # A retried request: transport error then success.
    records.append(
        attempt(
            "req-retry",
            1_700_000_000.0,
            "upstream_failure",
            status=None,
            error={"type": "ConnectError", "message": "All connection attempts failed"},
            diagnosis={
                "retryable": True,
                "reason": "transient upstream error ConnectError: All connection attempts failed",
            },
            response_body=None,
        )
    )
    records.append(
        attempt(
            "req-retry",
            1_700_000_001.0,
            "success",
            attempt=2,
            response_body=response_body("recovered"),
        )
    )

    # A retryable HTTP error.
    records.append(
        attempt(
            "req-http",
            1_700_000_002.0,
            "http_error",
            status=503,
            response_body=response_body("unavailable"),
            diagnosis={"retryable": True, "reason": "retryable HTTP status 503"},
        )
    )

    # Context-window overflow.
    records.append(
        attempt(
            "req-context",
            1_700_000_003.0,
            "context_window_exceeded",
            status=400,
            response_body=None,
            diagnosis={"retryable": False, "reason": "context window exceeded"},
        )
    )

    # A stalled abort that nevertheless produced output (false positive).
    records.append(
        abort(
            "req-stall",
            1_700_000_004.0,
            "stalled",
            "stalled",
            partial_content="partial answer before silence",
        )
    )

    # A loop abort followed by a loop-retry pass (same outcome, disambiguated).
    records.append(
        abort(
            "req-loop",
            1_700_000_005.0,
            "loop",
            "loop_aborted",
            partial_content="same same same",
        )
    )
    records.append(pass_record("req-loop", 1_700_000_006.0, "loop_retry", "loop_aborted"))

    # A runaway-reasoning pass (not an abort: no abort_kind).
    records.append(
        pass_record("req-runaway", 1_700_000_007.0, "runaway", "runaway_reasoning_aborted")
    )

    # An extraction pass then the final success.
    records.append(pass_record("req-extract", 1_700_000_008.0, "extract", "success"))
    records.append(
        final_success(
            "req-extract",
            1_700_000_009.0,
            rounds=1,
            extract={"attempts": 2, "outcome": "triggered"},
        )
    )

    # A coast pass and an empty-stop pass in a request that hits the rounds cap.
    records.append(pass_record("req-cap", 1_700_000_010.0, "coast", "success"))
    records.append(pass_record("req-cap", 1_700_000_011.0, "empty_stop", "success"))
    records.append(
        final_success(
            "req-cap",
            1_700_000_012.0,
            rounds=2,
            remediation_exhausted="rounds",
            extract={"attempts": 1, "outcome": "exhausted"},
            coast={"attempts": 1, "outcome": "triggered"},
            empty_stop={"attempts": 1, "outcome": "triggered"},
        )
    )

    # A plain success (nullable nested keys, as the proxy writes them).
    records.append(
        attempt(
            "req-plain",
            1_700_000_013.0,
            "success",
            response_body=response_body("plain"),
            extract=None,
            coast=None,
            empty_stop=None,
        )
    )

    # A tool-using conversation, with duplicated history across two attempts.
    records.append(tool_conversation_request("req-tools", 1_700_000_014.0))
    duplicate = tool_conversation_request("req-tools", 1_700_000_015.0)
    duplicate["attempt"] = 2
    records.append(duplicate)

    return records


def write_recording(directory, records, *, rotated=()):
    """Write records to ``directory/records.jsonl`` and optional rotated files."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    live = directory / "records.jsonl"
    live.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    for name, older in rotated:
        (directory / name).write_text(
            "".join(json.dumps(record) + "\n" for record in older), encoding="utf-8"
        )
    return live
