"""Think-tag cleanup — unit + integration.

Unit tests cover `ThinkContentGuard.clean` (relocation of complete and
unmatched tags, case-insensitivity, placeholder emission, no-op pass-through).
Integration tests drive the real `Proxy._forward_streaming` against a mock SSE
upstream that leaks a `<think>` block, proving both the non-streaming
(reconstructed) and streaming (SSE) client shapes return non-empty cleaned
content and preserve the relocated reasoning.

    docker compose exec -T gateway python - < tests/test_think_cleanup.py
"""

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.think import ThinkContentGuard

_PLACEHOLDER = "THE PLACEHOLDER"


def _guard(tags=None, placeholder=_PLACEHOLDER):
    return ThinkContentGuard(tags=tags, placeholder=placeholder)


# --- unit tests -------------------------------------------------------


def test_relocates_and_keeps_visible_answer():
    # A think block followed by real content: the block is relocated, the
    # visible answer stays put, no placeholder is needed.
    content, reasoning, changes = _guard().clean(
        "<think>scratch</think> the real answer"
    )
    assert content == " the real answer", repr(content)
    assert reasoning == "scratch", repr(reasoning)
    assert changes == [{"kind": "relocated_think", "chars": 7, "blocks": 1}], changes


def test_relocates_unmatched_opening_tag():
    # A truncated think block: everything after the tag is reasoning, so the
    # visible content is empty and the placeholder fires.
    content, reasoning, changes = _guard().clean("<think>truncated reasoning")
    assert content == _PLACEHOLDER, repr(content)
    assert reasoning == "truncated reasoning", repr(reasoning)
    kinds = {c["kind"] for c in changes}
    assert kinds == {"relocated_think", "empty_content_placeholder"}, changes


def test_tags_are_case_insensitive():
    content, reasoning, _ = _guard().clean("<THINK>hi</Think>")
    assert content == _PLACEHOLDER, repr(content)
    assert reasoning == "hi", repr(reasoning)


def test_reasoning_tag_variant():
    content, reasoning, _ = _guard().clean("<reasoning>why</reasoning>")
    assert reasoning == "why", repr(reasoning)
    assert content == _PLACEHOLDER, repr(content)


def test_placeholder_when_empty_content_and_reasoning():
    # Reasoning arrived via the proper channel (no think tag) but the model
    # produced no visible content: the placeholder still guarantees non-empty.
    content, reasoning, changes = _guard().clean("", "some reasoning")
    assert content == _PLACEHOLDER, repr(content)
    assert reasoning == "some reasoning", repr(reasoning)
    assert changes == [{"kind": "empty_content_placeholder"}], changes


def test_no_change_when_plain_content():
    content, reasoning, changes = _guard().clean("hello", "")
    assert content == "hello", repr(content)
    assert reasoning == "", repr(reasoning)
    assert changes == [], changes


def test_no_placeholder_when_both_empty():
    content, reasoning, changes = _guard().clean("", "")
    assert content == "", repr(content)
    assert changes == [], changes


def test_preserves_existing_reasoning():
    content, reasoning, _ = _guard().clean(
        "<think>extra</think>", "existing"
    )
    assert reasoning == "existing\nextra", repr(reasoning)
    assert content == _PLACEHOLDER, repr(content)


def test_custom_tags():
    content, reasoning, changes = _guard(tags=("ponder",)).clean(
        "<ponder>x</ponder>"
    )
    assert reasoning == "x", repr(reasoning)
    assert content == _PLACEHOLDER, repr(content)
    kinds = {c["kind"] for c in changes}
    assert "relocated_think" in kinds, changes


def test_think_only_emits_placeholder_not_empty():
    # The exact observed failure: the model replies only with a think tag.
    content, reasoning, changes = _guard().clean(
        "<think>step one, step two</think>"
    )
    assert content == _PLACEHOLDER, repr(content)
    assert reasoning == "step one, step two", repr(reasoning)
    kinds = {c["kind"] for c in changes}
    assert kinds == {"relocated_think", "empty_content_placeholder"}, changes


# --- integration harness ---------------------------------------------


def _chunk(delta: dict, finish_reason=None, usage=None) -> dict:
    chunk = {
        "id": "chatcmpl-think",
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


def _run_forward(chunks, stream=False):
    mock = _MockServer(chunks)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            settings.think_cleanup_enabled = True
            settings.think_empty_response_placeholder = _PLACEHOLDER
            recorder = Recorder("/tmp/think_cleanup_integration.jsonl")
            proxy = Proxy(recorder)
            req = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            if stream:
                req["stream"] = True
            body = json.dumps(req).encode()
            return await proxy._forward_streaming(
                request_id="think-itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "think-itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        return asyncio.run(run())
    finally:
        mock.stop()


def test_reconstructed_think_only_returns_placeholder():
    chunks = [
        _chunk({"role": "assistant", "content": "<think>"}),
        _chunk({"content": "step one"}),
        _chunk({"content": "</think>"}),
        _chunk({}, finish_reason="stop"),
    ]
    resp = _run_forward(chunks, stream=False)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert message["content"] == _PLACEHOLDER, message
    assert message["reasoning_content"] == "step one", message


def test_streaming_think_only_returns_placeholder_sse():
    chunks = [
        _chunk({"role": "assistant", "content": "<think>"}),
        _chunk({"content": "step one"}),
        _chunk({"content": "</think>"}),
        _chunk({}, finish_reason="stop"),
    ]
    resp = _run_forward(chunks, stream=True)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert resp.media_type == "text/event-stream", resp.media_type
    text = resp.body.decode("utf-8")
    # Rebuilt stream must carry the placeholder and relocated reasoning, and
    # must NOT leak the raw think tags.
    assert _PLACEHOLDER in text, text
    assert "step one" in text, text
    assert "<think>" not in text, text
    assert "data: [DONE]" in text, text


def test_streaming_no_change_passthrough():
    """Regression: a clean stream is still passed through verbatim (raw SSE)."""
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "plain answer"}),
        _chunk({}, finish_reason="stop"),
    ]
    resp = _run_forward(chunks, stream=True)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    text = resp.body.decode("utf-8")
    assert "plain answer" in text, text
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
