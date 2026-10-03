"""Context-window overflow detection — unit + integration.

Unit tests cover `ContextWindowDetector` classification (body, exception
message, case-insensitivity, deferral). Integration tests drive the real proxy
paths (buffered `forward` and `_forward_streaming`) against a mock upstream
that returns the canonical llama.cpp context-overflow error with a *retryable*
status — proving the gateway fails fast (one attempt) and passes the upstream's
error through verbatim instead of retrying a doomed request or translating it.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import Diagnosis
from app.remediation.context import (
    CONTEXT_WINDOW_CODE,
    ContextWindowDetector,
)
from app.remediation.retry import RetryableDetector
from mock_upstream import MockUpstream, make_request, settings_override

# The canonical llama.cpp / LiteLLM error body when the context window fills.
_LLAMACPP_ERROR = {
    "error": {
        "message": ("the request was aborted because it would have exceeded the context window"),
        "type": "invalid_request_error",
        "code": 400,
    }
}


def _detector(statuses=(500,)):
    return ContextWindowDetector(RetryableDetector(set(statuses)))


# --- unit tests -------------------------------------------------------


def test_detects_context_overflow_in_body():
    d = _detector()
    diagnosis = d.diagnose_status(500, body=json.dumps(_LLAMACPP_ERROR).encode())
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code == CONTEXT_WINDOW_CODE, diagnosis


def test_detects_context_overflow_in_exception_message():
    d = _detector()
    diagnosis = d.diagnose_exception(RuntimeError(_LLAMACPP_ERROR["error"]["message"]))
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code == CONTEXT_WINDOW_CODE, diagnosis


def test_markers_are_case_insensitive():
    d = _detector()
    diagnosis = d.diagnose_status(500, body=b'{"error": "Exceeded The Context Window"}')
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code == CONTEXT_WINDOW_CODE, diagnosis


def test_detects_litellm_phrasing():
    # LiteLLM wraps the provider's overflow with its own wording ("exceeds
    # the available context size") rather than llama.cpp's phrasing.
    d = _detector()
    body = (
        '{"error": {"message": "litellm.ContextWindowExceededError: '
        "request (33101 tokens) exceeds the available context size "
        '(32768 tokens), try increasing it"}}'
    )
    diagnosis = d.diagnose_status(400, body=body.encode())
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code == CONTEXT_WINDOW_CODE, diagnosis


def test_defers_when_no_marker():
    d = _detector()
    diagnosis = d.diagnose_status(500, body=b'{"error": "internal error"}')
    assert diagnosis.retryable is True, diagnosis  # 500 is retryable
    assert diagnosis.code is None, diagnosis


def test_defers_non_retryable_status():
    d = _detector()
    diagnosis = d.diagnose_status(404, body=b"not found")
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code is None, diagnosis


def test_diagnosis_str_returns_reason():
    diagnosis = Diagnosis(
        retryable=False,
        reason="context window exceeded",
        code=CONTEXT_WINDOW_CODE,
    )
    assert str(diagnosis) == "context window exceeded"


def test_diagnose_status_without_body_defers():
    # `body` defaults to None; `_decode` must treat it as empty (no marker).
    d = _detector()
    diagnosis = d.diagnose_status(500)
    assert diagnosis.retryable is True, diagnosis
    assert diagnosis.code is None, diagnosis


def test_defers_exception_without_marker():
    # Exception without a context-window signature defers to the fallback.
    d = _detector()
    diagnosis = d.diagnose_exception(RuntimeError("plain transient failure"))
    assert diagnosis.retryable is False, diagnosis
    assert diagnosis.code is None, diagnosis


# --- integration harness ---------------------------------------------


def _context_mock(status=500, payload=None):
    payload = payload if payload is not None else json.dumps(_LLAMACPP_ERROR).encode()
    return MockUpstream([{"status": status, "content_type": "application/json", "body": payload}])


def _context_settings(mock):
    return settings_override(
        llm_base_url=mock.url,
        context_window_detection_enabled=True,
        context_window_abort_status=413,
        retry_max_attempts=3,
        retry_backoff_initial=0.0,
        loop_detection_enabled=True,
    )


def test_buffered_forward_fails_fast():
    """Non-chat path: retryable 500 + context body -> one attempt, passed
    through verbatim (upstream status + body) instead of retried."""
    with _context_mock(status=500) as mock, _context_settings(mock):
        recorder = Recorder("/tmp/context_buffered.jsonl")

        async def run():
            proxy = Proxy(recorder)
            body = json.dumps({"prompt": "hello", "max_tokens": 16}).encode()
            request = await make_request(body, path="/v1/completions")
            return await proxy.forward(request, "v1/completions")

        resp = asyncio.run(run())
        assert resp.status_code == 500, (resp.status_code, resp.body)
        # The upstream's own error is forwarded verbatim, not translated.
        assert json.loads(resp.body) == _LLAMACPP_ERROR, resp.body
        assert mock.request_count == 1


def test_streaming_fails_fast():
    """Chat path (HTTP error branch): retryable 500 + context body -> passed
    through verbatim (upstream status + body) instead of retried."""
    with _context_mock(status=500) as mock, _context_settings(mock):
        recorder = Recorder("/tmp/context_streaming.jsonl")

        async def run():
            proxy = Proxy(recorder)
            body = json.dumps(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode()
            return await proxy._forward_streaming(
                request_id="ctx-itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "ctx-itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        resp = asyncio.run(run())
        assert resp.status_code == 500, (resp.status_code, resp.body)
        # The upstream's own error is forwarded verbatim, not translated.
        assert json.loads(resp.body) == _LLAMACPP_ERROR, resp.body
        assert mock.request_count == 1
