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
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()


class _SequenceHandler(BaseHTTPRequestHandler):
    """Serves a *different* chunk-list per request, capturing request bodies.

    Used to verify loop remediation: the first exchange loops, the second is
    re-submitted with varied sampling and produces a clean turn.
    """

    responses: list = []
    requests: list = []

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        body = self.rfile.read(length) if length else b""
        type(self).requests.append(json.loads(body.decode("utf-8")))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = type(self).responses.pop(0) if type(self).responses else []
        for chunk in chunks:
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, *args):  # silence request logging
        pass


class _SequenceServer:
    def __init__(self, responses):
        self.server = HTTPServer(("127.0.0.1", 0), _SequenceHandler)
        _SequenceHandler.responses = list(responses)
        _SequenceHandler.requests = []
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()


def _run_forward(chunks):
    mock = _MockServer(chunks)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            # Loop remediation is exercised by its own dedicated tests; keep
            # the shared abort tests on the pure detect-and-abort path.
            settings.loop_retry_enabled = False
            # Small window so the loop test arms on a short payload.
            settings.loop_window_bytes = 2000
            settings.loop_min_output_fraction = 0.5
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
    # Enough repetitions to cross the (small) arming gate set in _run_forward.
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": sentence + ". "}) for _ in range(40)],
        _chunk(
            {},
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 40, "total_tokens": 41},
        ),
    ]
    resp = _run_forward(chunks)
    data = json.loads(resp.body)
    assert resp.status_code == settings.loop_abort_status, (resp.status_code, resp.body)
    assert data["error"]["type"] == "loop_detected", data


def test_finish_reason_length_is_a_loop():
    # Regression: a *drift* loop (near-identical, not verbatim) does not trip
    # the compression-ratio detector, so the model runs to the output window
    # limit and the upstream returns ``finish_reason: "length"`` (llama.cpp
    # ``truncated = 1``). That truncation is itself a runaway signal and must
    # be treated as a loop, not passed through as a truncated 200.
    sentences = [
        "Let me carefully reconsider whether the total should be zero.",
        "Let me carefully reconsider whether the count should be zero.",
        "Let me carefully reconsider whether the sum should be zero.",
        "Let me carefully reconsider whether the value should be zero.",
        "Let me carefully reconsider whether the index should be zero.",
    ]
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": s + " "}) for s in sentences],
        _chunk(
            {},
            finish_reason="length",
            usage={"prompt_tokens": 4164, "completion_tokens": 4028, "total_tokens": 8192},
        ),
    ]
    resp = _run_forward(chunks)
    data = json.loads(resp.body)
    assert resp.status_code == settings.loop_abort_status, (resp.status_code, resp.body)
    assert data["error"]["type"] == "loop_detected", data
    assert "window exhausted" in data["error"]["reason"], data


def _loop_chunks():
    sentence = "Let me carefully reconsider the whole approach before continuing."
    return [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": sentence + ". "}) for _ in range(40)],
        _chunk({}, finish_reason="stop"),
    ]


def _run_forward_sequence(responses):
    mock = _SequenceServer(responses)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            settings.loop_retry_enabled = True
            settings.loop_retry_max_attempts = 2
            settings.loop_window_bytes = 2000
            settings.loop_min_output_fraction = 0.5
            recorder = Recorder("/tmp/loop_retry_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()
            return await proxy._forward_streaming(
                request_id="itest-retry",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "itest-retry",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        return asyncio.run(run())
    finally:
        mock.stop()


def test_loop_retry_breaks_loop():
    """A looped first pass is re-submitted with varied sampling and succeeds."""
    good = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "The capital of France is Paris."}),
        _chunk({}, finish_reason="stop"),
    ]
    resp = _run_forward_sequence([_loop_chunks(), good])
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["object"] == "chat.completion", data
    assert "Paris" in data["choices"][0]["message"]["content"], data
    # Two upstream exchanges: the loop, then the varied-sampling re-submission.
    assert len(_SequenceHandler.requests) == 2, _SequenceHandler.requests
    second = _SequenceHandler.requests[1]
    assert second["temperature"] == settings.loop_retry_temperature, second
    assert second["repeat_penalty"] == settings.loop_retry_repeat_penalty, second
    assert second["presence_penalty"] == settings.loop_retry_presence_penalty, second
    assert second["frequency_penalty"] == settings.loop_retry_frequency_penalty, second
    # The recorded loop_retry entry must capture the exact re-submitted body,
    # so the overridden sampling is auditable against the client's original.
    recorded = [
        line
        for line in open("/tmp/loop_retry_integration.jsonl")
        if json.loads(line).get("loop_retry")
    ]
    assert recorded, "expected a loop_retry record"
    entry = json.loads(recorded[-1])
    rb = entry["loop_retry"]["resubmitted_body"]
    assert rb["temperature"] == settings.loop_retry_temperature, rb
    assert rb["repeat_penalty"] == settings.loop_retry_repeat_penalty, rb
    assert rb["presence_penalty"] == settings.loop_retry_presence_penalty, rb
    assert rb["frequency_penalty"] == settings.loop_retry_frequency_penalty, rb


def test_loop_retry_exhausts_to_abort():
    """When every re-submission still loops, the request aborts."""
    resp = _run_forward_sequence([_loop_chunks(), _loop_chunks(), _loop_chunks()])
    assert resp.status_code == settings.loop_abort_status, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["error"]["type"] == "loop_detected", data
    # Original attempt + loop_retry_max_attempts re-submissions.
    assert len(_SequenceHandler.requests) == 3, _SequenceHandler.requests


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
    body = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
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
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"loc'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ation": "Paris"}'}}]}),
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
