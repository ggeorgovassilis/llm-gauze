"""The streaming path must not be gated by loop detection alone (#89).

Historically ``Proxy.forward`` entered the streaming path only when
``loop_detection_enabled`` was true, so disabling loop detection to silence one
false positive silently disabled every other streaming feature — stall
detection, runaway-reasoning detection, nudge, coast, think-cleanup, and the
tool-call guard — regardless of their own switches.

These tests drive the real ``Proxy.forward`` (the gating decision, not
``_forward_streaming``) with ``loop_detection_enabled=False`` and a single
other streaming feature enabled, proving that feature still fires.
"""

import asyncio
import json

from app.config import settings
from app.proxy import Proxy, _streaming_feature_enabled
from app.recorder import Recorder
from mock_upstream import MockUpstream, chunk, make_request, settings_override, sse

_chunk = chunk
_sse = sse


def _body(payload=None):
    return json.dumps(
        payload or {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    ).encode()


def _run_forward(specs, *, body=None, **overrides):
    """Drive ``Proxy.forward`` with loop detection off and only ``overrides`` on."""
    base = {
        "loop_detection_enabled": False,
        "stall_detection_enabled": False,
        "runaway_reasoning_enabled": False,
        "think_cleanup_enabled": False,
        "think_nudge_enabled": False,
        "coast_detection_enabled": False,
        "loop_retry_enabled": False,
        "tool_call_guard_enabled": False,
    }
    with MockUpstream(specs) as mock:
        base["llm_base_url"] = mock.url
        base.update(overrides)
        with settings_override(**base):
            recorder = Recorder("/tmp/streaming_independence.jsonl")
            proxy = Proxy(recorder)

            async def run():
                request = await make_request(body or _body(), path="/v1/chat/completions")
                return await proxy.forward(request, "v1/chat/completions")

            resp = asyncio.run(run())
            return resp, mock.request_count


# --- the gating decision ---------------------------------------------


def test_streaming_feature_enabled_matches_each_switch():
    # Every streaming feature must independently force the streaming path on.
    switches = [
        "loop_detection_enabled",
        "stall_detection_enabled",
        "runaway_reasoning_enabled",
        "think_nudge_enabled",
        "coast_detection_enabled",
        "think_cleanup_enabled",
        "tool_call_guard_enabled",
    ]
    # All off -> no streaming path.
    with settings_override(**{s: False for s in switches}):
        assert _streaming_feature_enabled() is False
    # Each switch alone -> streaming path.
    for switch in switches:
        with settings_override(**{s: (s == switch) for s in switches}):
            assert _streaming_feature_enabled() is True, switch


# --- each feature fires with loop detection off ----------------------


def test_stall_detection_fires_without_loop_detection():
    steps = [(0.03, b": ping\n\n") for _ in range(30)]
    resp, _ = _run_forward(
        [{"steps": steps}],
        stall_detection_enabled=True,
        stall_ttft_seconds=0.2,
        stall_gap_seconds=0.3,
    )
    assert resp.status_code == settings.stall_abort_status, (resp.status_code, resp.body)
    assert json.loads(resp.body)["error"]["type"] == "stalled_detected"


def test_runaway_reasoning_fires_without_loop_detection():
    chunks = [
        _chunk({"reasoning_content": "thinking " * 20}),
        _chunk({}, finish_reason="stop"),
    ]
    resp, _ = _run_forward(
        [{"chunks": chunks}],
        runaway_reasoning_enabled=True,
        runaway_reasoning_token_threshold=10,
        runaway_reasoning_max_attempts=1,
    )
    assert resp.status_code == settings.runaway_reasoning_abort_status, (
        resp.status_code,
        resp.body,
    )
    assert json.loads(resp.body)["error"]["type"] == "runaway_reasoning_detected"


def test_nudge_fires_without_loop_detection():
    empty_turn = [
        _chunk({"role": "assistant", "reasoning_content": "thinking hard"}),
        _chunk({}, finish_reason="stop"),
    ]
    answer_turn = [
        _chunk({"role": "assistant", "content": "the real answer"}),
        _chunk({}, finish_reason="stop"),
    ]
    resp, count = _run_forward(
        [{"chunks": empty_turn}, {"chunks": answer_turn}],
        think_nudge_enabled=True,
        think_nudge_max_attempts=2,
    )
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["choices"][0]["message"]["content"] == "the real answer"
    # Two upstream requests: the empty turn and the nudged retry.
    assert count == 2, count


def test_coast_fires_without_loop_detection():
    loop_request = {
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
    coasted = "Chunk 5 done: 12 products, 0 picks. Pulling next chunk."
    coast_turn = [
        _chunk({"role": "assistant", "content": coasted, "reasoning_content": coasted}),
        _chunk({}, finish_reason="stop"),
    ]
    tool_call_turn = [
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
    resp, count = _run_forward(
        [{"chunks": coast_turn}, {"chunks": tool_call_turn}],
        body=_body(loop_request),
        coast_detection_enabled=True,
        coast_max_attempts=2,
    )
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert message.get("tool_calls"), message
    assert message["tool_calls"][0]["function"]["name"] == "next_chunk"
    assert count == 2, count


def test_think_cleanup_fires_without_loop_detection():
    leaky_turn = [
        _chunk({"role": "assistant", "content": "<think>scratch</think> the real answer"}),
        _chunk({}, finish_reason="stop"),
    ]
    resp, _ = _run_forward(
        [{"chunks": leaky_turn}],
        think_cleanup_enabled=True,
    )
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert "real answer" in message["content"], message
    assert "scratch" not in message["content"], message
    assert message["reasoning_content"] == "scratch", message


def test_tool_call_guard_fires_without_loop_detection():
    truncated_call = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "London'},
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp, _ = _run_forward(
        [{"chunks": truncated_call}],
        tool_call_guard_enabled=True,
    )
    assert resp.status_code == 200, (resp.status_code, resp.body)
    call = json.loads(resp.body)["choices"][0]["message"]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"city": "London"}, call


def test_loop_detection_still_fires_when_enabled():
    # Sanity: with loop detection itself on (and everything else off) a
    # repetitive stream is still caught — the gate change did not break it.
    sentence = "Let me carefully reconsider the whole approach before continuing."
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        *[_chunk({"content": sentence + ". "}) for _ in range(40)],
        _chunk({}, finish_reason="stop"),
    ]
    resp, _ = _run_forward(
        [{"chunks": chunks}],
        loop_detection_enabled=True,
        loop_retry_enabled=False,
        loop_window_bytes=2000,
        loop_min_output_fraction=0.5,
    )
    assert resp.status_code == settings.loop_abort_status, (resp.status_code, resp.body)
    assert json.loads(resp.body)["error"]["type"] == "loop_detected"
