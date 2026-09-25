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
