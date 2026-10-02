"""Message-overflow protection (oversized tool results) — unit + integration.

Unit tests cover `MessageOverflowGuard.process` (deterministic trigger, warn +
truncate, warn-only, non-tool messages untouched, non-string content, threshold
boundary, input immutability). Integration tests drive the real `Proxy.forward`
through both the buffered path (loop detection off) and the streaming path
(loop detection on), asserting the mock upstream received the warned/truncated
body and the recorder captured the intervention — plus a disabled-switch no-op.

    docker compose exec -T gateway python - < tests/test_overflow.py
"""

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.overflow import MessageOverflowGuard

_WARNING = "PLEASE_WORK_AROUND_THIS"


def _chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-overflow",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# --- unit tests -------------------------------------------------------


def _tool_messages(content):
    return {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": content},
        ],
    }


def test_oversized_tool_result_truncated_and_warned():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    content = "first line\n" + "x" * 1000
    out, changes = guard.process(_tool_messages(content))
    assert len(changes) == 1, changes
    assert changes[0] == {
        "kind": "tool_overflow",
        "index": 1,
        "size": len(content),
        "truncated": True,
    }, changes
    rewritten = out["messages"][1]["content"]
    assert rewritten == f"{_WARNING}\nfirst line", rewritten
    # Input is not mutated.
    assert "x" * 1000 in _tool_messages(content)["messages"][1]["content"]


def test_warn_only_keeps_full_content():
    guard = MessageOverflowGuard(threshold=10, truncate=False, warning=_WARNING)
    content = "first line\n" + "x" * 1000
    out, changes = guard.process(_tool_messages(content))
    assert len(changes) == 1, changes
    assert changes[0]["truncated"] is False
    assert out["messages"][1]["content"] == f"{_WARNING}\n{content}", out


def test_single_line_blob_capped_at_threshold():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    content = "y" * 5000  # no newline -> minified-style blob
    out, _ = guard.process(_tool_messages(content))
    rewritten = out["messages"][1]["content"]
    assert rewritten == f"{_WARNING}\n{'y' * 10}...", rewritten


def test_threshold_boundary_not_flagged():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    out, changes = guard.process(_tool_messages("a" * 10))
    assert changes == [], changes
    assert out["messages"][1]["content"] == "a" * 10


def test_non_tool_messages_untouched():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "big" * 100},
            {"role": "user", "content": "huge" * 100},
            {"role": "assistant", "content": "vast" * 100},
        ],
    }
    out, changes = guard.process(body)
    assert changes == [], changes
    assert out is body, out


def test_non_string_tool_content_skipped():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [{"role": "tool", "tool_call_id": "c1", "content": ["list"]}],
    }
    out, changes = guard.process(body)
    assert changes == [], changes
    assert out is body, out


def test_multiple_tool_messages_flag_only_oversized():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [
            {"role": "tool", "tool_call_id": "c1", "content": "small"},
            {"role": "tool", "tool_call_id": "c2", "content": "b" * 100},
            {"role": "tool", "tool_call_id": "c3", "content": "c" * 100},
        ],
    }
    out, changes = guard.process(body)
    assert [c["index"] for c in changes] == [1, 2], changes
    assert out["messages"][0]["content"] == "small"
    assert out["messages"][1]["content"].startswith(_WARNING)
    assert out["messages"][2]["content"].startswith(_WARNING)


# --- integration harness ---------------------------------------------


class _CaptureHandler(BaseHTTPRequestHandler):
    respond_sse = False
    captured_body = None
    requests = 0

    def do_POST(self):
        type(self).requests += 1
        length = int(self.headers.get("Content-Length", 0))
        type(self).captured_body = self.rfile.read(length)
        if type(self).respond_sse:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in [
                _chunk({"role": "assistant", "content": "ok"}),
                _chunk({}, finish_reason="stop"),
            ]:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            payload = json.dumps(
                {
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "test",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def log_message(self, *args):  # silence request logging
        pass


class _MockServer:
    def __init__(self, respond_sse=False):
        self.server = HTTPServer(("127.0.0.1", 0), _CaptureHandler)
        _CaptureHandler.respond_sse = respond_sse
        _CaptureHandler.captured_body = None
        _CaptureHandler.requests = 0
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()


async def _make_request(body: bytes):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
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


def _run_forward(loop_enabled, content, enabled=True):
    mock = _MockServer(respond_sse=loop_enabled)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = loop_enabled
            settings.message_overflow_enabled = enabled
            settings.message_overflow_threshold = 10
            settings.message_overflow_truncate = True
            settings.message_overflow_warning = _WARNING
            recorder = Recorder("/tmp/overflow_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {
                    "model": "test",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "tool",
                            "tool_call_id": "c1",
                            "content": content,
                        },
                    ],
                }
            ).encode()
            request = await _make_request(body)
            return await proxy.forward(request, "v1/chat/completions")

        resp = asyncio.run(run())
        records = [
            json.loads(line) for line in open("/tmp/overflow_integration.jsonl") if line.strip()
        ]
        captured = json.loads(_CaptureHandler.captured_body)
        return resp, records, captured
    finally:
        mock.stop()


def test_buffered_forward_truncates_before_upstream():
    resp, records, captured = _run_forward(False, "first line\n" + "x" * 1000)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    tool_content = captured["messages"][1]["content"]
    assert tool_content == f"{_WARNING}\nfirst line", tool_content
    final = records[-1]
    assert final["message_overflow"][0]["index"] == 1, final
    assert final["message_overflow"][0]["size"] == len("first line\n" + "x" * 1000)


def test_streaming_forward_truncates_before_upstream():
    resp, records, captured = _run_forward(True, "first line\n" + "x" * 1000)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    tool_content = captured["messages"][1]["content"]
    assert tool_content == f"{_WARNING}\nfirst line", tool_content
    assert records[-1]["message_overflow"][0]["index"] == 1, records[-1]


def test_disabled_is_noop():
    resp, records, captured = _run_forward(False, "first line\n" + "x" * 1000, enabled=False)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    # The upstream received the full, unmodified tool result.
    assert captured["messages"][1]["content"] == "first line\n" + "x" * 1000
    assert records[-1].get("message_overflow") is None, records[-1]


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
