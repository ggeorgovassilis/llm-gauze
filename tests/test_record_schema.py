"""Stabilised recorder record schema (#92).

Every recorded exchange must carry a stable ``outcome`` discriminator drawn
from the same vocabulary telemetry already emits on ``requests_total``, and
``response_body`` must mean exactly one thing: the body actually delivered to
the client, or ``null`` when nothing was delivered. The pre-remediation turn
that used to be written into ``response_body`` on the nudge/coast pass records
now lives in ``pre_remediation_body``.

These tests cover:

* the pure ``_attempt_outcome`` classifier;
* the buffered ``forward`` path (success / http_error / upstream_failure, and
  the retried attempt discarding its body);
* the streaming verdict-abort record;
* the nudge/coast remediation-pass records;
* the runaway/loop-retry remediation-pass records.

Run via the test wrapper (``./scripts/test.sh tests/test_record_schema.py``):
these tests use pytest fixtures and are not runnable as a plain script.
"""

import asyncio
import json
import random

import httpx
from app.proxy import Proxy, _attempt_outcome
from app.recorder import Recorder
from app.remediation.base import Diagnosis
from app.remediation.context import CONTEXT_WINDOW_CODE
from mock_upstream import MockUpstream, chunk, make_request, settings_override

_chunk = chunk

# The outcome vocabulary persisted on every record, mirroring the telemetry
# labels on ``requests_total``.
_OUTCOME_VOCABULARY = {
    "success",
    "http_error",
    "upstream_failure",
    "loop_aborted",
    "stalled",
    "runaway_reasoning_aborted",
    "context_window_exceeded",
}


def _read_records(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _http(status, body=b'{"ok": true}', content_type="application/json"):
    return httpx.Response(status, content=body, headers={"content-type": content_type})


# --- pure classifier --------------------------------------------------


def test_attempt_outcome_success():
    assert _attempt_outcome(200, None, None) == "success"


def test_attempt_outcome_http_error():
    assert _attempt_outcome(500, None, None) == "http_error"


def test_attempt_outcome_transport_error():
    assert _attempt_outcome(None, {"type": "ConnectError"}, None) == "upstream_failure"


def test_attempt_outcome_error_beats_status():
    assert _attempt_outcome(200, {"type": "ReadError"}, None) == "upstream_failure"


def test_attempt_outcome_missing_status_is_upstream_failure():
    assert _attempt_outcome(None, None, None) == "upstream_failure"


def test_attempt_outcome_context_window_beats_status():
    diagnosis = Diagnosis(
        retryable=False,
        reason="context window exceeded",
        code=CONTEXT_WINDOW_CODE,
    )
    assert _attempt_outcome(500, None, diagnosis) == "context_window_exceeded"


# --- buffered forward path --------------------------------------------


def _run_buffered(handler, *, record_path, body=None, **overrides):
    transport = httpx.MockTransport(handler)
    with settings_override(
        loop_detection_enabled=False,
        retry_max_attempts=3,
        retry_backoff_initial=0.0,
        retry_backoff_jitter=False,
        **overrides,
    ):
        recorder = Recorder(record_path)
        proxy = Proxy(recorder)
        proxy.client = httpx.AsyncClient(base_url="http://upstream.test", transport=transport)
        request_body = body if body is not None else json.dumps({"prompt": "hi"}).encode()

        async def run():
            request = await make_request(request_body, path="/v1/completions")
            return await proxy.forward(request, "v1/completions")

        return asyncio.run(run()), _read_records(record_path)


def test_buffered_success_record(tmp_path):
    async def handler(request):
        return _http(200)

    resp, records = _run_buffered(handler, record_path=tmp_path / "record_schema_success.jsonl")
    assert resp.status_code == 200
    assert len(records) == 1, records
    assert records[0]["outcome"] == "success"
    assert records[0]["outcome"] in _OUTCOME_VOCABULARY
    assert records[0]["response_body"] == '{"ok": true}'


def test_buffered_http_error_record(tmp_path):
    async def handler(request):
        return _http(404, body=b'{"error": "not found"}')

    resp, records = _run_buffered(handler, record_path=tmp_path / "record_schema_http_error.jsonl")
    assert resp.status_code == 404
    assert len(records) == 1, records
    assert records[0]["outcome"] == "http_error"
    assert records[0]["response_body"] == '{"error": "not found"}'


def test_buffered_upstream_failure_record(tmp_path):
    async def handler(request):
        raise httpx.ConnectError("connection refused")

    resp, records = _run_buffered(
        handler, record_path=tmp_path / "record_schema_upstream_failure.jsonl"
    )
    assert resp.status_code == 502
    assert len(records) == 3, records
    assert all(r["outcome"] == "upstream_failure" for r in records), records
    # Nothing was ever delivered, so no attempt carries a response body.
    assert all(r["response_body"] is None for r in records), records


def test_buffered_retry_discards_intermediate_body(tmp_path):
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _http(500, body=b'{"error": "boom"}')
        return _http(200)

    resp, records = _run_buffered(handler, record_path=tmp_path / "record_schema_retry.jsonl")
    assert resp.status_code == 200
    assert [r["outcome"] for r in records] == ["http_error", "success"], records
    # The retried attempt's body was discarded, not delivered.
    assert records[0]["response_body"] is None, records[0]
    assert records[1]["response_body"] == '{"ok": true}', records[1]


# --- streaming verdict-abort record -----------------------------------


def test_streaming_abort_record_has_verdict_outcome(tmp_path):
    """A stall abort records the verdict outcome and a null response_body."""
    steps = [(0.03, b": ping\n\n") for _ in range(30)]
    record_path = tmp_path / "record_schema_stall.jsonl"
    with MockUpstream([{"steps": steps}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            stall_detection_enabled=True,
            stall_ttft_seconds=0.2,
            stall_gap_seconds=0.3,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()

            async def run():
                return await proxy._forward_streaming(
                    request_id="schema-stall",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "schema-stall",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )

            asyncio.run(run())

    records = _read_records(record_path)
    aborts = [r for r in records if r.get("abort_kind")]
    assert len(aborts) == 1, records
    assert aborts[0]["abort_kind"] == "stalled", aborts[0]
    assert aborts[0]["outcome"] == "stalled", aborts[0]
    assert aborts[0]["outcome"] in _OUTCOME_VOCABULARY
    assert aborts[0]["response_body"] is None, aborts[0]


# --- nudge / coast remediation-pass records ---------------------------


def test_nudge_pass_moves_turn_to_pre_remediation_body(tmp_path):
    empty = [
        _chunk({"role": "assistant", "reasoning_content": "thinking hard"}),
        _chunk({}, finish_reason="stop"),
    ]
    answer = [
        _chunk({"role": "assistant", "content": "the real answer"}),
        _chunk({}, finish_reason="stop"),
    ]
    record_path = tmp_path / "record_schema_nudge.jsonl"
    with MockUpstream([{"chunks": empty}, {"chunks": answer}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=True,
            think_empty_response_placeholder="PLACEHOLDER",
            think_nudge_enabled=True,
            think_nudge_text="PLEASE_REPLY_VISIBLY",
            think_nudge_max_attempts=2,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()

            async def run():
                return await proxy._forward_streaming(
                    request_id="schema-nudge",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "schema-nudge",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )

            asyncio.run(run())

    records = _read_records(record_path)
    triggered = [r for r in records if r.get("nudge", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    # The re-submitted (discarded) turn moved to `pre_remediation_body`;
    # `response_body` is null because nothing was delivered to the client.
    assert triggered[0]["outcome"] == "success", triggered[0]
    assert triggered[0]["response_body"] is None, triggered[0]
    assert "reasoning_content" in triggered[0]["pre_remediation_body"], triggered[0]

    # The final record carries the delivered body.
    final = records[-1]
    assert final["outcome"] == "success", final
    assert "the real answer" in final["response_body"], final


def test_coast_pass_moves_turn_to_pre_remediation_body(tmp_path):
    record_path = tmp_path / "record_schema_coast.jsonl"
    coasted = [
        _chunk(
            {
                "role": "assistant",
                "content": "Chunk 5 done: 12 products.",
                "reasoning_content": "Chunk 5 done: 12 products.",
            }
        ),
        _chunk({}, finish_reason="stop"),
    ]
    tool_call = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_next",
                        "type": "function",
                        "function": {"name": "next_chunk", "arguments": "{}"},
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    request = {
        "model": "test",
        "messages": [
            {"role": "user", "content": "vet the products"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "next_chunk", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "next_chunk", "description": "next chunk"},
            }
        ],
    }
    with MockUpstream([{"chunks": coasted}, {"chunks": tool_call}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=True,
            think_nudge_enabled=False,
            coast_detection_enabled=True,
            coast_nudge_text="PLEASE_MAKE_THE_TOOL_CALL",
            coast_max_attempts=2,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(request).encode()

            async def run():
                return await proxy._forward_streaming(
                    request_id="schema-coast",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "schema-coast",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )

            asyncio.run(run())

    records = _read_records(record_path)
    triggered = [r for r in records if r.get("coast", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    assert triggered[0]["outcome"] == "success", triggered[0]
    assert triggered[0]["response_body"] is None, triggered[0]
    assert "Chunk 5 done" in triggered[0]["pre_remediation_body"], triggered[0]

    final = records[-1]
    assert final["outcome"] == "success", final
    assert "next_chunk" in final["response_body"], final


# --- runaway / loop-retry remediation-pass records --------------------


def test_runaway_pass_records_runaway_outcome(tmp_path):
    """A runaway-reasoning nudge records ``outcome: runaway_reasoning_aborted``."""
    rng = random.Random(1234)
    flood = " ".join("".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=6)) for _ in range(2500))
    runaway_turn = [
        _chunk({"reasoning_content": flood}),
        _chunk({}, finish_reason="length"),
    ]
    answer_turn = [
        _chunk({"role": "assistant", "content": "The answer is 42."}),
        _chunk({}, finish_reason="stop"),
    ]
    record_path = tmp_path / "record_schema_runaway.jsonl"
    with MockUpstream([{"chunks": runaway_turn}, {"chunks": answer_turn}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=False,
            think_nudge_enabled=False,
            coast_detection_enabled=False,
            loop_retry_enabled=False,
            runaway_reasoning_enabled=True,
            runaway_reasoning_nudge_text="STOP_THINKING_ANSWER_NOW",
            runaway_reasoning_max_attempts=2,
            runaway_reasoning_token_threshold=2000,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "q"}]}
            ).encode()

            async def run():
                return await proxy._forward_streaming(
                    request_id="schema-runaway",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "schema-runaway",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )

            asyncio.run(run())

    records = _read_records(record_path)
    triggered = [r for r in records if r.get("runaway", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    assert triggered[0]["outcome"] == "runaway_reasoning_aborted", triggered[0]
    assert triggered[0]["outcome"] in _OUTCOME_VOCABULARY
    assert triggered[0]["response_body"] is None, triggered[0]


def test_loop_retry_pass_records_loop_aborted_outcome(tmp_path):
    """A varied-sampling re-submission records ``outcome: loop_aborted``."""
    sentence = "Let me carefully reconsider the whole approach before continuing."
    loop_turn = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": sentence + ". "}) for _ in range(40)],
        _chunk({}, finish_reason="stop"),
    ]
    answer_turn = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "The capital of France is Paris."}),
        _chunk({}, finish_reason="stop"),
    ]
    record_path = tmp_path / "record_schema_loop_retry.jsonl"
    with MockUpstream([{"chunks": loop_turn}, {"chunks": answer_turn}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            loop_retry_enabled=True,
            loop_retry_max_attempts=2,
            loop_window_bytes=2000,
            loop_min_output_fraction=0.5,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()

            async def run():
                return await proxy._forward_streaming(
                    request_id="schema-loop-retry",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "schema-loop-retry",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )

            asyncio.run(run())

    records = _read_records(record_path)
    triggered = [r for r in records if r.get("loop_retry", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    assert triggered[0]["outcome"] == "loop_aborted", triggered[0]
    assert triggered[0]["outcome"] in _OUTCOME_VOCABULARY
    assert triggered[0]["response_body"] is None, triggered[0]
