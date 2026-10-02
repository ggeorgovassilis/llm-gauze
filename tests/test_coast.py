"""Coast detection (announced-but-absent tool call) — unit + integration.

Unit tests cover `CoastPolicy.should_nudge` (the deterministic trigger) and
`CoastPolicy.apply` (replays the coasted assistant turn then appends the
re-prompt, without mutating the input). Integration tests drive the real
`Proxy._forward_streaming` against a mock SSE upstream that returns a coasted
turn on the first request and a real tool call on the second — proving the
rung re-submits and surfaces the tool call — plus cases proving the budget is
honoured when the model keeps coasting, and that the disabled switch is a
no-op.

    docker compose exec -T gateway python - < tests/test_coast.py
"""

import asyncio
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.coast import CoastPolicy

_COAST_TEXT = "PLEASE_MAKE_THE_TOOL_CALL"
_COASTED_CONTENT = "Chunk 5 done: 12 products, 0 picks. Pulling next chunk."


def _chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-coast",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# A coasted turn: stop, non-empty content, no tool calls, reasoning == content.
_COAST_TURN = [
    _chunk(
        {
            "role": "assistant",
            "content": _COASTED_CONTENT,
            "reasoning_content": _COASTED_CONTENT,
        }
    ),
    _chunk({}, finish_reason="stop"),
]

# A healthy turn: the model actually emits the tool call.
_TOOL_CALL_TURN = [
    _chunk(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_next",
                    "type": "function",
                    "function": {"name": "next_chunk", "arguments": "{}"},
                }
            ],
        }
    ),
    _chunk({}, finish_reason="tool_calls"),
]


def _loop_request():
    """A request driving a tool loop: tools present + prior assistant tool call."""
    return {
        "model": "test",
        "messages": [
            {"role": "user", "content": "vet the products"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "next_chunk", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "next_chunk", "description": "next chunk"},
            }
        ],
    }


# --- unit tests -------------------------------------------------------


def test_should_nudge_coasted_turn():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop",
            _COASTED_CONTENT,
            None,
            _COASTED_CONTENT,
            req["tools"],
            req["messages"],
        )
        is True
    )


def test_should_nudge_ignores_real_tool_call():
    req = _loop_request()
    tool_calls = [{"id": "c", "function": {"name": "next_chunk", "arguments": "{}"}}]
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop", "", tool_calls, "", req["tools"], req["messages"]
        )
        is False
    )


def test_should_nudge_requires_non_empty_content():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop", "   ", None, "   ", req["tools"], req["messages"]
        )
        is False
    )


def test_should_nudge_requires_reasoning_match():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop",
            _COASTED_CONTENT,
            None,
            "some different reasoning",
            req["tools"],
            req["messages"],
        )
        is False
    )


def test_should_nudge_requires_tools():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop",
            _COASTED_CONTENT,
            None,
            _COASTED_CONTENT,
            None,
            req["messages"],
        )
        is False
    )


def test_should_nudge_requires_prior_tool_call():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "stop",
            _COASTED_CONTENT,
            None,
            _COASTED_CONTENT,
            req["tools"],
            [{"role": "user", "content": "hi"}],
        )
        is False
    )


def test_should_nudge_requires_stop_finish_reason():
    req = _loop_request()
    assert (
        CoastPolicy(text=_COAST_TEXT).should_nudge(
            "length",
            _COASTED_CONTENT,
            None,
            _COASTED_CONTENT,
            req["tools"],
            req["messages"],
        )
        is False
    )


def test_apply_replays_coast_turn_and_appends_nudge():
    req = _loop_request()
    out = CoastPolicy(text=_COAST_TEXT).apply(req, _COASTED_CONTENT)
    assert len(out["messages"]) == len(req["messages"]) + 2, out
    assert out["messages"][-2] == {
        "role": "assistant",
        "content": _COASTED_CONTENT,
    }, out
    assert out["messages"][-1] == {"role": "user", "content": _COAST_TEXT}, out
    # Input is left unmutated.
    assert len(req["messages"]) == 3, req


# --- integration harness ---------------------------------------------


class _MockHandler(BaseHTTPRequestHandler):
    chunks_by_request: list = []
    requests = 0

    def do_POST(self):
        idx = min(type(self).requests, len(type(self).chunks_by_request) - 1)
        type(self).requests += 1
        chunks = type(self).chunks_by_request[idx]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            try:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                break

    def log_message(self, *args):  # silence request logging
        pass


class _MockServer:
    def __init__(self, chunks_by_request):
        self.server = HTTPServer(("127.0.0.1", 0), _MockHandler)
        _MockHandler.chunks_by_request = chunks_by_request
        _MockHandler.requests = 0
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()


def _run_forward(chunks_by_request, coast_max=2, enabled=True):
    mock = _MockServer(chunks_by_request)
    try:

        async def run():
            settings.llm_base_url = f"http://127.0.0.1:{mock.port}"
            settings.loop_detection_enabled = True
            settings.think_cleanup_enabled = True
            settings.think_nudge_enabled = False
            settings.coast_detection_enabled = enabled
            settings.coast_nudge_text = _COAST_TEXT
            settings.coast_max_attempts = coast_max
            recorder = Recorder("/tmp/coast_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(_loop_request()).encode()
            resp = await proxy._forward_streaming(
                request_id="coast-itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "coast-itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )
            records = [
                json.loads(line) for line in open("/tmp/coast_integration.jsonl") if line.strip()
            ]
            return resp, records

        return asyncio.run(run())
    finally:
        mock.stop()


def test_coasted_turn_is_reprompted_and_returns_tool_call():
    resp, records = _run_forward([_COAST_TURN, _TOOL_CALL_TURN], coast_max=2)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert message.get("tool_calls"), message
    assert message["tool_calls"][0]["function"]["name"] == "next_chunk", message

    # Two upstream requests: the coasted turn and the re-prompted retry.
    assert _MockHandler.requests == 2, _MockHandler.requests
    final = records[-1]
    assert final["coast"] == {"attempts": 1, "outcome": "succeeded"}, final
    triggered = [r for r in records if r.get("coast", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records


def test_coast_budget_exhausted_returns_coasted_turn():
    resp, records = _run_forward([_COAST_TURN, _COAST_TURN, _COAST_TURN], coast_max=2)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    # No tool call — the rung gave up and returned the coasted turn as-is.
    assert not message.get("tool_calls"), message
    assert _MockHandler.requests == 3, _MockHandler.requests
    final = records[-1]
    assert final["coast"] == {"attempts": 2, "outcome": "exhausted"}, final


def test_coast_disabled_is_noop():
    resp, records = _run_forward([_COAST_TURN], coast_max=2, enabled=False)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert not message.get("tool_calls"), message
    assert _MockHandler.requests == 1, _MockHandler.requests
    assert not records[-1].get("coast"), records[-1]


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
