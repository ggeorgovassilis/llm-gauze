"""End-to-end test of the stalled-stream detection path.

Spins up a mock SSE upstream that returns headers and then either stays silent
(only keepalive/comment frames) or produces tokens. Verifies the silent case is
aborted with a ``stalled_detected`` verdict (instead of hanging) and the
flowing case is not aborted.

    docker compose exec -T gateway python - < tests/test_stall_integration.py
"""

import asyncio
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder


def _chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


class _MockHandler(BaseHTTPRequestHandler):
    # Each step is (seconds_to_sleep_before_write, payload_bytes).
    steps: list = []

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delay, payload in type(self).steps:
            time.sleep(delay)
            try:
                self.wfile.write(payload)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, *args):  # silence request logging
        pass


class _MockServer:
    def __init__(self, steps):
        self.server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        _MockHandler.steps = steps
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def stop(self):
        self.server.shutdown()


def _run_forward(steps):
    mock = _MockServer(steps)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.stall_detection_enabled = True
            settings.stall_ttft_seconds = 0.2
            settings.stall_gap_seconds = 0.3
            recorder = Recorder("/tmp/stall_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode()
            return await proxy._forward_streaming(
                request_id="stall-itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "stall-itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        return asyncio.run(run())
    finally:
        mock.stop()


def test_stall_aborts_silent_stream():
    """Headers + keepalives only (no content token) -> stall abort, no hang."""
    # ~0.9s of keepalive/comment frames, far beyond the 0.2s TTFT budget.
    steps = [(0.03, b": ping\n\n") for _ in range(30)]
    resp = _run_forward(steps)
    data = json.loads(resp.body)
    assert resp.status_code == settings.stall_abort_status, (
        resp.status_code,
        resp.body,
    )
    assert data["error"]["type"] == "stalled_detected", data
    assert "token" in data["error"]["reason"], data


def test_stall_does_not_abort_flowing_stream():
    """Tokens arriving within the gap budget must not trip the watchdog."""
    steps = [
        (0.0, b": ping\n\n"),  # keepalive must NOT reset the timer
        (0.05, _sse(_chunk({"role": "assistant", "content": ""}))),
        (0.05, _sse(_chunk({"content": "Hello "}))),
        (0.05, _sse(_chunk({"content": "world"}))),
        (0.05, _sse(_chunk({}, finish_reason="stop"))),
    ]
    resp = _run_forward(steps)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["object"] == "chat.completion", data
    assert data["choices"][0]["message"]["content"] == "Hello world", data


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
