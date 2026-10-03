"""End-to-end test of the stalled-stream detection path.

Spins up a mock SSE upstream that returns headers and then either stays silent
(only keepalive/comment frames) or produces tokens. Verifies the silent case is
aborted with a ``stalled_detected`` verdict (instead of hanging) and the
flowing case is not aborted.
"""

import asyncio
import json

from app.config import settings
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import VerdictKind
from mock_upstream import MockUpstream, chunk, settings_override, sse

_chunk = chunk
_sse = sse


def _run_forward(steps):
    with MockUpstream([{"steps": steps}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            stall_detection_enabled=True,
            stall_ttft_seconds=0.2,
            stall_gap_seconds=0.3,
        ):
            recorder = Recorder("/tmp/stall_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            ).encode()

            async def run():
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


def test_tool_call_deltas_reset_stall_timer():
    """Tool-call fragments are content-bearing: a slow argument stream must
    not be misread as a stall even when no reasoning/content token appears."""
    # Each tool-call fragment arrives within the 0.3s gap budget, but there is
    # never a reasoning/content token — pre-fix this tripped the watchdog.
    steps = [
        (0.05, _sse(_chunk({"role": "assistant", "content": ""}))),
        (
            0.05,
            _sse(
                _chunk(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ]
                    }
                )
            ),
        ),
        (0.05, _sse(_chunk({}, finish_reason="tool_calls"))),
    ]
    resp = _run_forward(steps)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    call = data["choices"][0]["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather", call


def test_stall_abort_records_partial_output():
    """The abort record preserves partial reasoning/content/tool-calls."""
    steps = [
        (0.0, _sse(_chunk({"reasoning_content": "about to call a tool"}))),
        # Then nothing for far beyond the gap budget -> stall.
        (0.5, _sse(_chunk({}, finish_reason="stop"))),
    ]
    resp = _run_forward(steps)
    assert resp.status_code == settings.stall_abort_status, (
        resp.status_code,
        resp.body,
    )
    records = [json.loads(line) for line in open("/tmp/stall_integration.jsonl") if line.strip()]
    abort = [r for r in records if r.get("abort_kind") == VerdictKind.STALLED.value][-1]
    assert abort["partial_reasoning"] == "about to call a tool", abort
    assert abort["partial_content"] is None, abort
