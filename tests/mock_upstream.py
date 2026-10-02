"""Shared in-process mock OpenAI-compatible upstream + small test helpers.

The integration tests drive the real ``Proxy`` against a :class:`MockUpstream`
so no test depends on a live (and non-deterministic) LLM. ``MockUpstream`` is
an :class:`http.server.HTTPServer` on an ephemeral port that, per request,
serves a response spec. Three spec shapes are supported:

* ``{"chunks": [...]}`` — SSE ``data:`` frames, one per chunk dict. When a
  request arrives after the last spec, the final spec repeats, so a one-element
  list yields a fixed response for every request.
* ``{"steps": [(delay_seconds, payload_bytes), ...]}`` — raw bytes written to
  the SSE stream, with a sleep before each write (used by the stall detector).
* ``{"status": int, "content_type": str, "body": bytes}`` — a fixed HTTP reply
  (used by the context-window error tests).

Every request body is captured on :attr:`MockUpstream.requests`.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings


def chunk(delta: dict, finish_reason=None, usage=None) -> dict:
    """Build an OpenAI ``chat.completion.chunk`` dict (see the proxy's
    reconstruction logic). ``delta`` is the per-chunk ``choices[0].delta``."""
    c = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        c["usage"] = usage
    return c


def sse(data: dict) -> bytes:
    """Encode a chunk dict as a single SSE ``data:`` frame."""
    return f"data: {json.dumps(data)}\n\n".encode()


async def make_request(body: bytes, path: str = "/v1/chat/completions"):
    """Build a starlette ``Request`` for driving the buffered ``Proxy.forward``."""
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
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


class _Handler(BaseHTTPRequestHandler):
    """Serves the response spec configured on the owning :class:`MockUpstream`."""

    def do_POST(self):
        upstream = type(self).upstream
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        upstream.requests.append(body)
        spec = upstream.spec_for(len(upstream.requests) - 1)

        if "chunks" in spec:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in spec["chunks"]:
                try:
                    self.wfile.write(sse(c))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        elif "steps" in spec:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delay, payload in spec["steps"]:
                time.sleep(delay)
                try:
                    self.wfile.write(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break
        else:
            self.send_response(spec["status"])
            self.send_header("Content-Type", spec.get("content_type", "application/json"))
            self.send_header("Content-Length", str(len(spec["body"])))
            self.end_headers()
            self.wfile.write(spec["body"])

    def log_message(self, *args):  # silence request logging
        pass


class MockUpstream:
    """An in-process HTTP server standing in for the OpenAI-compatible LLM.

    ``specs`` is a list of per-request response specs (see module docstring).
    Use it as a context manager (``with MockUpstream([...]) as mock:``) or call
    :meth:`stop` explicitly. ``mock.url`` is the ``http://127.0.0.1:<port>``
    base URL to point ``settings.llm_base_url`` at.
    """

    def __init__(self, specs):
        self.specs = specs
        self.requests = []  # raw request bodies, one per POST, in order
        self.server = HTTPServer(("127.0.0.1", 0), _Handler)
        _Handler.upstream = self
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def spec_for(self, index: int) -> dict:
        return self.specs[min(index, len(self.specs) - 1)]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def request_count(self) -> int:
        return len(self.requests)

    def json_bodies(self) -> list:
        """Decode each captured request body as JSON."""
        return [json.loads(b.decode("utf-8")) for b in self.requests]

    def stop(self):
        self.server.shutdown()
        self.thread.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


class settings_override:
    """Context manager that overrides ``settings.*`` fields and restores them.

    Use instead of assigning ``settings.<field> = ...`` directly so no override
    leaks into the next test::

        with settings_override(llm_base_url=mock.url, loop_detection_enabled=False):
            ...
    """

    def __init__(self, **overrides):
        self._overrides = overrides
        self._originals = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._originals[key] = getattr(settings, key)
            setattr(settings, key, value)
        return self

    def __exit__(self, *exc):
        for key, value in self._originals.items():
            setattr(settings, key, value)
        return False
