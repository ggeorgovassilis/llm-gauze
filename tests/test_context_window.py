"""Context-window overflow detection — unit + integration.

Unit tests cover `ContextWindowDetector` classification (body, exception
message, case-insensitivity, deferral). Integration tests drive the real proxy
paths (buffered `forward` and `_forward_streaming`) against a mock upstream
that returns the canonical llama.cpp context-overflow error with a *retryable*
status — proving the gateway fails fast (one attempt) and passes the upstream's
error through verbatim instead of retrying a doomed request or translating it.

    docker compose exec -T gateway python - < tests/test_context_window.py
"""

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.context import (
    CONTEXT_WINDOW_CODE,
    ContextWindowDetector,
)
from app.remediation.retry import RetryableDetector

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


# --- integration harness ---------------------------------------------


class _ContextHandler(BaseHTTPRequestHandler):
    # Class-level config shared across requests.
    status = 500
    payload = json.dumps(_LLAMACPP_ERROR).encode()
    requests = 0

    def do_POST(self):
        type(self).requests += 1
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, *args):  # silence request logging
        pass


class _MockServer:
    def __init__(self, status=500, payload=None):
        self.server = HTTPServer(("127.0.0.1", 0), _ContextHandler)
        _ContextHandler.status = status
        _ContextHandler.payload = (
            payload if payload is not None else json.dumps(_LLAMACPP_ERROR).encode()
        )
        _ContextHandler.requests = 0
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()


def _configure(mock):
    settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
    settings.context_window_detection_enabled = True
    settings.context_window_abort_status = 413
    settings.retry_max_attempts = 3
    settings.retry_backoff_initial = 0.0
    settings.loop_detection_enabled = True


async def _make_request(body: bytes):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/completions",
        "raw_path": b"/v1/completions",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"host", b"127.0.0.1"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(scope, receive)


def test_buffered_forward_fails_fast():
    """Non-chat path: retryable 500 + context body -> one attempt, passed
    through verbatim (upstream status + body) instead of retried."""
    mock = _MockServer(status=500)
    try:
        _configure(mock)
        recorder = Recorder("/tmp/context_buffered.jsonl")

        async def run():
            proxy = Proxy(recorder)
            body = json.dumps({"prompt": "hello", "max_tokens": 16}).encode()
            request = await _make_request(body)
            return await proxy.forward(request, "v1/completions")

        resp = asyncio.run(run())
        assert resp.status_code == 500, (resp.status_code, resp.body)
        # The upstream's own error is forwarded verbatim, not translated.
        assert json.loads(resp.body) == _LLAMACPP_ERROR, resp.body
        assert _ContextHandler.requests == 1, _ContextHandler.requests
    finally:
        mock.stop()


def test_streaming_fails_fast():
    """Chat path (HTTP error branch): retryable 500 + context body -> passed
    through verbatim (upstream status + body) instead of retried."""
    mock = _MockServer(status=500)
    try:
        _configure(mock)
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
        assert _ContextHandler.requests == 1, _ContextHandler.requests
    finally:
        mock.stop()


def _run_all() -> int:
    tests = [
        value
        for key, value in sorted(globals().items())
        if key.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
