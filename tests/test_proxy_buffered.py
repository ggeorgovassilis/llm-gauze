"""Tests for the buffered (non-streaming) forward path and its pure helpers.

Follow-up from the test audit (#39), finding F3: only the streaming path
(`_forward_streaming`) was exercised. This covers the buffered `Proxy.forward`
retry loop for non-chat-completion routes — successful passthrough,
retry-then-success, exhausted-retry 502, and HTTP-error passthrough — plus the
pure helpers that were previously uncovered: `_decode`, `_diagnosis_entry`,
`_is_chat_completion`, `_ensure_stream`, `_filtered_headers`, and the
synthesised branch of `_context_window_error_response`.
"""

import asyncio
import json
import logging

import httpx
import pytest
from app.config import settings
from app.proxy import (
    Proxy,
    _apply_overflow_guard,
    _client_wants_stream,
    _context_window_error_response,
    _decode,
    _diagnosis_entry,
    _ensure_stream,
    _filtered_headers,
    _is_chat_completion,
    _merge_tool_calls,
    _reconstruct_sse_lines,
)
from app.recorder import Recorder
from app.remediation.base import Diagnosis
from app.remediation.context import CONTEXT_WINDOW_CODE
from mock_upstream import MockUpstream, make_request, settings_override, sse
from mock_upstream import chunk as _chunk


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "spec",
    [
        {"status": 200, "content_type": "text/plain", "body": b"Not an API endpoint"},
        {"status": 200, "content_type": "application/json", "body": b'{"choices":[]}'},
        {"steps": [(0, b"data: [DONE]\n\n")]},
        {"chunks": [{"choices": [], "usage": {"completion_tokens": 0}}]},
        {"steps": [(0, b"data: invalid\n\ndata: []\n\n")]},
    ],
)
def test_streaming_non_sse_success_is_not_fabricated(tmp_path, stream, spec):
    async def run():
        with MockUpstream([spec]) as upstream:
            with settings_override(
                llm_base_url=upstream.url,
                extract_enabled=True,
                coast_detection_enabled=True,
                think_cleanup_enabled=True,
                retry_max_attempts=3,
            ):
                proxy = Proxy(Recorder(tmp_path / "records.jsonl"))
                try:
                    response = await proxy.forward(
                        await make_request(json.dumps({"stream": stream, "messages": []}).encode()),
                        "v1/chat/completions",
                    )
                finally:
                    await proxy.aclose()
            assert upstream.request_count == 1
        assert response.status_code == 502
        assert response.headers["content-type"] == "application/json"
        assert "choices" not in json.loads(response.body)
        assert json.loads(response.body)["error"]["code"] == "invalid_upstream_response"

    asyncio.run(run())


@pytest.mark.parametrize("http_status", [200, 401])
def test_streaming_upstream_errors_preserved(tmp_path, http_status):
    error = {"error": {"code": "unauthorised", "message": "Upstream rejected the request"}}
    spec = (
        {"chunks": [error]}
        if http_status == 200
        else {"status": http_status, "body": json.dumps(error).encode()}
    )

    async def run():
        with MockUpstream([spec]) as upstream:
            with settings_override(llm_base_url=upstream.url, think_cleanup_enabled=True):
                proxy = Proxy(Recorder(tmp_path / "records.jsonl"))
                try:
                    response = await proxy.forward(
                        await make_request(b'{"stream":true,"messages":[]}'),
                        "v1/chat/completions",
                    )
                finally:
                    await proxy.aclose()
            assert upstream.request_count == 1
        assert response.status_code == (502 if http_status == 200 else http_status)
        assert json.loads(response.body) == error

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_streaming_valid_reasoning_tools_and_usage_preserved(tmp_path, stream):
    chunks = [
        _chunk({"role": "assistant", "reasoning_content": "Let me check."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ]
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
        {"choices": [], "usage": {"completion_tokens": 12}},
    ]

    async def run():
        with MockUpstream([{"chunks": chunks}]) as upstream:
            with settings_override(llm_base_url=upstream.url, think_cleanup_enabled=True):
                proxy = Proxy(Recorder(tmp_path / "records.jsonl"))
                try:
                    response = await proxy.forward(
                        await make_request(json.dumps({"stream": stream, "messages": []}).encode()),
                        "v1/chat/completions",
                    )
                finally:
                    await proxy.aclose()
            assert upstream.request_count == 1
        assert response.status_code == 200
        if stream:
            assert response.body == b"".join(sse(frame) for frame in chunks) + b"data: [DONE]\n\n"
        else:
            result = json.loads(response.body)
            assert result["choices"][0]["finish_reason"] == "tool_calls"
            assert result["choices"][0]["message"]["reasoning_content"] == "Let me check."
            assert result["choices"][0]["message"]["tool_calls"][0]["function"] == {
                "name": "lookup",
                "arguments": "{}",
            }
            assert result["usage"] == {"completion_tokens": 12}

    asyncio.run(run())


# --- pure helper unit tests -------------------------------------------


def test_decode_empty_returns_none():
    assert _decode(b"") is None


def test_decode_utf8_body():
    assert _decode(b"hello") == "hello"


def test_decode_invalid_utf8_replaced():
    assert _decode(b"\xff\xfe") == "\ufffd\ufffd"


def test_diagnosis_entry_none_returns_none():
    assert _diagnosis_entry(None) is None


def test_diagnosis_entry_without_code():
    diagnosis = Diagnosis(retryable=True, reason="retryable HTTP status 500")
    assert _diagnosis_entry(diagnosis) == {
        "retryable": True,
        "reason": "retryable HTTP status 500",
    }


def test_diagnosis_entry_with_code():
    diagnosis = Diagnosis(
        retryable=False,
        reason="context window exceeded",
        code=CONTEXT_WINDOW_CODE,
    )
    entry = _diagnosis_entry(diagnosis)
    assert entry == {
        "retryable": False,
        "reason": "context window exceeded",
        "code": CONTEXT_WINDOW_CODE,
    }


def test_is_chat_completion_matches():
    assert _is_chat_completion("/v1/chat/completions") is True
    assert _is_chat_completion("/v1/chat/completions/") is True
    assert _is_chat_completion("/v1/completions") is False
    assert _is_chat_completion("/v1/chat/completions/stream") is False


def test_ensure_stream_empty_body():
    body, ctype = _ensure_stream(b"")
    assert json.loads(body) == {"stream": True}
    assert ctype == "application/json"


def test_ensure_stream_adds_stream_true():
    body, ctype = _ensure_stream(b'{"model": "test"}')
    assert json.loads(body) == {"model": "test", "stream": True}
    assert ctype == "application/json"


def test_ensure_stream_invalid_json_untouched():
    raw = b"not json"
    body, ctype = _ensure_stream(raw)
    assert body == raw
    assert ctype == "application/json"


def test_ensure_stream_non_dict_untouched():
    raw = b"[1, 2, 3]"
    body, ctype = _ensure_stream(raw)
    assert body == raw
    assert ctype == "application/json"


def test_filtered_headers_strips_hop_by_hop_and_owned():
    headers = {
        "content-type": "application/json",
        "content-length": "42",
        "connection": "keep-alive",
        "transfer-encoding": "chunked",
        "x-custom": "yes",
    }
    out, ctype = _filtered_headers(headers)
    assert ctype == "application/json"
    assert out == {"x-custom": "yes"}


def test_context_window_error_response_passthrough():
    diagnosis = Diagnosis(
        retryable=False,
        reason="context window exceeded",
        code=CONTEXT_WINDOW_CODE,
    )
    resp = _context_window_error_response(
        status=500,
        body=b'{"error": "boom"}',
        headers={"content-type": "application/json"},
        diagnosis=diagnosis,
    )
    assert resp.status_code == 500
    assert resp.body == b'{"error": "boom"}'


def test_context_window_error_response_synthesises_when_no_body():
    diagnosis = Diagnosis(
        retryable=False,
        reason="context window exceeded",
        code=CONTEXT_WINDOW_CODE,
    )
    resp = _context_window_error_response(
        status=None,
        body=None,
        headers=None,
        diagnosis=diagnosis,
    )
    assert resp.status_code == settings.context_window_abort_status
    data = json.loads(resp.body)
    assert data["error"]["type"] == CONTEXT_WINDOW_CODE, data
    assert data["error"]["message"] == "context window exceeded"


def test_client_wants_stream_empty_body():
    assert _client_wants_stream(b"") is False


def test_client_wants_stream_invalid_json():
    assert _client_wants_stream(b"not json") is False


def test_client_wants_stream_non_dict():
    assert _client_wants_stream(b"[1, 2, 3]") is False


def test_client_wants_stream_true():
    assert _client_wants_stream(b'{"stream": true}') is True


def test_apply_overflow_guard_invalid_json():
    with settings_override(message_overflow_enabled=True):
        changes, body = _apply_overflow_guard(b"not json")
    assert changes == []
    assert body == b"not json"


def test_apply_overflow_guard_non_dict():
    with settings_override(message_overflow_enabled=True):
        changes, body = _apply_overflow_guard(b"[1, 2, 3]")
    assert changes == []
    assert body == b"[1, 2, 3]"


def test_apply_overflow_guard_no_changes():
    payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
    with settings_override(message_overflow_enabled=True):
        changes, body = _apply_overflow_guard(payload)
    assert changes == []
    assert body == payload


def test_merge_tool_calls_skips_non_dict_fragments():
    acc = []
    _merge_tool_calls(acc, [None, "not-a-dict", 42])
    assert acc == []


def test_reconstruct_sse_lines_includes_tool_calls():
    lines = _reconstruct_sse_lines(
        meta={},
        content="",
        reasoning="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{}"},
            }
        ],
    )
    assert lines[-1] == "data: [DONE]"
    assert "get_weather" in "\n".join(lines)


def _parse_sse_chunks(lines):
    """Parse `_reconstruct_sse_lines` output into the chunk dicts (sans [DONE])."""
    return [json.loads(line.removeprefix("data: ")) for line in lines[:-1]]


def test_reconstruct_sse_lines_full_envelope():
    meta = {
        "id": "chatcmpl-123",
        "created": 1700000000,
        "model": "test-model",
        "role": "assistant",
        "finish_reason": "stop",
    }
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        },
    ]
    lines = _reconstruct_sse_lines(
        meta=meta, content="hello", reasoning="thinking", tool_calls=calls
    )
    assert lines[-1] == "data: [DONE]"
    chunks = _parse_sse_chunks(lines)

    # Fixed chunk order: role, reasoning_content, content, one per tool call,
    # then the terminal empty-delta chunk.
    assert [chunk["choices"][0]["delta"] for chunk in chunks] == [
        {"role": "assistant"},
        {"reasoning_content": "thinking"},
        {"content": "hello"},
        {"tool_calls": [{**calls[0], "index": 0}]},
        {"tool_calls": [{**calls[1], "index": 1}]},
        {},
    ]

    # Every chunk shares the same fixed envelope: one base id/created/model,
    # one choices[0] with index 0, and finish_reason null on non-terminal chunks.
    for chunk in chunks[:-1]:
        assert chunk["id"] == "chatcmpl-123"
        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["created"] == 1700000000
        assert chunk["model"] == "test-model"
        assert chunk["choices"][0]["index"] == 0
        assert chunk["choices"][0]["finish_reason"] is None

    terminal = chunks[-1]
    assert terminal["id"] == "chatcmpl-123"
    assert terminal["choices"][0]["delta"] == {}
    assert terminal["choices"][0]["finish_reason"] == "stop"


def test_reconstruct_sse_lines_defaults():
    lines = _reconstruct_sse_lines(meta={}, content="", reasoning="", tool_calls=[])
    assert lines[-1] == "data: [DONE]"
    chunks = _parse_sse_chunks(lines)
    assert len(chunks) == 2  # role chunk + terminal chunk

    role_chunk, terminal = chunks
    assert role_chunk["choices"][0]["delta"] == {"role": "assistant"}
    assert role_chunk["choices"][0]["finish_reason"] is None
    assert role_chunk["model"] == ""
    assert isinstance(role_chunk["created"], int)
    assert role_chunk["id"]  # freshly generated UUID hex
    assert role_chunk["id"] == terminal["id"]  # base id reused across chunks

    assert terminal["choices"][0]["delta"] == {}
    assert terminal["choices"][0]["finish_reason"] == "stop"


# --- buffered forward path integration --------------------------------


def _http(status: int, body: bytes = b'{"ok": true}', content_type: str = "application/json"):
    return httpx.Response(status, content=body, headers={"content-type": content_type})


def _run_forward(handler, *, body: bytes | None = None, **overrides):
    """Drive `Proxy.forward` on a non-chat route against a mock httpx transport."""
    transport = httpx.MockTransport(handler)
    with settings_override(
        loop_detection_enabled=False,
        retry_max_attempts=3,
        retry_backoff_initial=0.0,
        retry_backoff_jitter=False,
        **overrides,
    ):
        recorder = Recorder("/tmp/proxy_buffered.jsonl")
        proxy = Proxy(recorder)
        # Swap in the mock transport so transport errors (connection refused,
        # etc.) can be scripted deterministically without a live server.
        proxy.client = httpx.AsyncClient(base_url="http://upstream.test", transport=transport)
        request_body = body if body is not None else json.dumps({"prompt": "hi"}).encode()

        async def run():
            request = await make_request(request_body, path="/v1/completions")
            return await proxy.forward(request, "v1/completions")

        return asyncio.run(run())


def test_forward_success_passthrough():
    async def handler(request):
        return _http(200, body=b'{"ok": true}')

    resp = _run_forward(handler)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert json.loads(resp.body) == {"ok": True}


def test_forward_retry_then_success():
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _http(500, body=b'{"error": "boom"}')
        return _http(200, body=b'{"ok": true}')

    resp = _run_forward(handler)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert calls["n"] == 2


def test_forward_exhausted_retry_returns_502():
    async def handler(request):
        raise httpx.ConnectError("connection refused")

    resp = _run_forward(handler)
    assert resp.status_code == 502, (resp.status_code, resp.body)
    assert json.loads(resp.body) == {"error": "upstream failure"}


def test_forward_http_error_passthrough():
    async def handler(request):
        return _http(404, body=b'{"error": "not found"}')

    resp = _run_forward(handler)
    assert resp.status_code == 404, (resp.status_code, resp.body)
    assert json.loads(resp.body) == {"error": "not found"}


def test_forward_logs_completion_summary(caplog):
    async def handler(request):
        return _http(200, body=b'{"ok": true}')

    with caplog.at_level(logging.INFO, logger="llm_gauze.proxy"):
        resp = _run_forward(handler)
    assert resp.status_code == 200

    summaries = [
        r
        for r in caplog.records
        if r.name == "llm_gauze.proxy" and "outcome=success" in r.getMessage()
    ]
    assert summaries, [r.getMessage() for r in caplog.records]
    msg = summaries[0].getMessage()
    assert "POST /v1/completions" in msg
    assert "status=200" in msg
    assert "duration=" in msg
    assert "outcome=success" in msg


def test_forward_completion_duration_spans_retries(caplog):
    """The completion line's `duration` must cover the whole request.

    It starts when the request arrives and ends when the final response is
    produced, so it includes the failed first attempt, the backoff sleep, and
    the successful retry — not just the last attempt.
    """
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        await asyncio.sleep(0.1)  # a slow, failing first attempt
        if calls["n"] == 1:
            return _http(500, body=b'{"error": "boom"}')
        return _http(200, body=b'{"ok": true}')

    with caplog.at_level(logging.INFO, logger="llm_gauze.proxy"):
        resp = _run_forward(handler)
    assert resp.status_code == 200
    assert calls["n"] == 2

    summaries = [
        r
        for r in caplog.records
        if r.name == "llm_gauze.proxy" and "outcome=success" in r.getMessage()
    ]
    assert summaries, [r.getMessage() for r in caplog.records]
    msg = summaries[0].getMessage()
    duration = float(msg.split("duration=")[1].split(" ")[0].rstrip("s"))
    # The logged duration is wall-clock and must at least cover the slow first
    # attempt plus the retry backoff; a last-attempt-only duration would be ~0.
    assert duration >= 0.15, msg
