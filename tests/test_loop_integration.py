"""End-to-end test of the loop-detection streaming path.

Spins up a mock SSE upstream inside the process and drives the real
``Proxy._forward_streaming`` against it, verifying both the loop-abort and the
non-loop reconstruction paths. Runs in the container:

    docker compose exec -T gateway python - < tests/test_loop_integration.py
"""

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder


def _chunk(delta: dict, finish_reason=None, usage=None) -> dict:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


class _MockHandler(BaseHTTPRequestHandler):
    chunks: list = []

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in type(self).chunks:
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, *args):  # silence request logging
        pass


class _MockServer:
    def __init__(self, chunks):
        self.server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        _MockHandler.chunks = chunks
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def stop(self):
        self.server.shutdown()


def _run_forward(chunks):
    mock = _MockServer(chunks)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            recorder = Recorder("/tmp/loop_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            return await proxy._forward_streaming(
                request_id="itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        return asyncio.run(run())
    finally:
        mock.stop()


def test_loop_abort():
    sentence = "Let me carefully reconsider the whole approach before continuing."
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": sentence + ". "}) for _ in range(8)],
        _chunk(
            {},
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 8, "total_tokens": 9},
        ),
    ]
    resp = _run_forward(chunks)
    data = json.loads(resp.body)
    assert resp.status_code == settings.loop_abort_status, (resp.status_code, resp.body)
    assert data["error"]["type"] == "loop_detected", data


def test_non_loop_reconstruction():
    sentences = [
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
        "Python is a popular programming language.",
    ]
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": s + " "}) for s in sentences],
        _chunk(
            {},
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
        ),
    ]
    resp = _run_forward(chunks)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["object"] == "chat.completion", data
    content = data["choices"][0]["message"]["content"]
    assert content == " ".join(sentences) + " ", repr(content)


def test_stale_content_length_header():
    """Regression: the body is rewritten (stream forced on), so the inbound
    Content-Length no longer matches. Forwarding it verbatim made h11 raise
    ``LocalProtocolError: Too much data for declared Content-Length``; the
    proxy must drop it and let httpx recompute it instead.
    """
    # No ``stream`` key on purpose: `_ensure_stream` *adds* ``"stream": true``,
    # making the rewritten body longer than the inbound Content-Length.
    body = json.dumps(
        {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    sentences = [
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
    ]
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": s}) for s in sentences],
        _chunk({}, finish_reason="stop"),
    ]
    mock = _MockServer(chunks)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            recorder = Recorder("/tmp/loop_integration_cl.jsonl")
            proxy = Proxy(recorder)
            return await proxy._forward_streaming(
                request_id="itest-cl",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={
                    "content-type": "application/json",
                    "content-length": str(len(body)),
                },
                base_entry={
                    "request_id": "itest-cl",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        resp = asyncio.run(run())
    finally:
        mock.stop()

    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["object"] == "chat.completion", data
    assert data["choices"][0]["message"]["content"] == "".join(sentences), data


def test_tool_calls_reconstruction():
    """Regression: tool-using turns must carry their ``tool_calls`` through.

    Streaming tool calls arrive as fragments (first carries id/name, later
    ones append ``arguments``). Dropping them left a message with
    ``finish_reason: tool_calls`` but no content and no tool_calls, which the
    client rejected as "Response contained no choices".
    """
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        ),
        _chunk(
            {"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}
        ),
        _chunk(
            {
                "tool_calls": [
                    {"index": 0, "function": {"arguments": 'ation": "Paris"}'}}
                ]
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp = _run_forward(chunks)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert data["choices"][0]["finish_reason"] == "tool_calls", data
    assert "content" not in message, data
    assert message["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            },
        }
    ], data


def test_streaming_passthrough():
    """Regression: a ``stream: true`` client must get SSE back, not JSON.

    Copilot sends ``stream: true`` and parses the response as SSE ``data:``
    events. Answering with a plain JSON body yielded zero completions and
    failed with "Response contained no choices".
    """
    sentences = [
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
    ]
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": s + " "}) for s in sentences],
        _chunk({}, finish_reason="stop"),
    ]
    body = json.dumps(
        {
            "model": "test",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
    ).encode()
    mock = _MockServer(chunks)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            recorder = Recorder("/tmp/loop_integration_stream.jsonl")
            proxy = Proxy(recorder)
            return await proxy._forward_streaming(
                request_id="itest-stream",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "itest-stream",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        resp = asyncio.run(run())
    finally:
        mock.stop()

    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert resp.media_type == "text/event-stream", resp.media_type
    text = resp.body.decode("utf-8")
    assert "data:" in text, text
    assert "[DONE]" in text, text
    assert "chat.completion.chunk" in text, text


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
