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

import httpx
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
from mock_upstream import make_request, settings_override

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
