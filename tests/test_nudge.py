"""Nudge (re-prompt empty-text turns) — unit + integration.

Unit tests cover `NudgePolicy.should_nudge` (deterministic trigger) and
`NudgePolicy.apply` (appends the nudge user message without mutating input).
Integration tests drive the real `Proxy._forward_streaming` against a mock SSE
upstream that returns a reasoning-only turn on the first request and real
content on the second — proving the nudge re-submits and returns the real
answer — plus a second case proving the budget is honoured and the placeholder
floor is reached when nudging keeps failing.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import Turn
from app.remediation.nudge import NudgePolicy
from mock_upstream import MockUpstream, chunk, settings_override

_NUDGE = "PLEASE_REPLY_VISIBLY"
_PLACEHOLDER = "THE_PLACEHOLDER"


_chunk = chunk


# A reasoning-only turn: stop, no content, no tool calls, non-empty reasoning.
_EMPTY_TURN = [
    _chunk({"role": "assistant", "reasoning_content": "thinking hard"}),
    _chunk({}, finish_reason="stop"),
]

# A healthy turn: visible content.
_ANSWER_TURN = [
    _chunk({"role": "assistant", "content": "the real answer"}),
    _chunk({}, finish_reason="stop"),
]


# --- unit tests -------------------------------------------------------


def test_should_nudge_empty_turn():
    assert (
        NudgePolicy(text=_NUDGE, max_attempts=2).should_nudge("stop", "", None, "thinking") is True
    )


def test_should_nudge_ignores_tool_calls():
    tool_calls = [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]
    assert (
        NudgePolicy(text=_NUDGE, max_attempts=2).should_nudge("stop", "", tool_calls, "thinking")
        is False
    )


def test_should_nudge_ignores_visible_content():
    assert (
        NudgePolicy(text=_NUDGE, max_attempts=2).should_nudge("stop", "hello", None, "thinking")
        is False
    )


def test_should_nudge_requires_stop_finish_reason():
    assert (
        NudgePolicy(text=_NUDGE, max_attempts=2).should_nudge("length", "", None, "thinking")
        is False
    )


def test_should_nudge_requires_reasoning():
    assert NudgePolicy(text=_NUDGE, max_attempts=2).should_nudge("stop", "", None, "") is False


def test_apply_appends_nudge_message():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    out = NudgePolicy(text=_NUDGE, max_attempts=2).apply(Turn(), body)
    assert len(out["messages"]) == 2, out
    assert out["messages"][-1] == {"role": "user", "content": _NUDGE}, out
    assert out["messages"][0] == {"role": "user", "content": "hi"}, out


def test_apply_does_not_mutate_input():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    NudgePolicy(text=_NUDGE, max_attempts=2).apply(Turn(), body)
    assert len(body["messages"]) == 1, body


# --- integration harness ---------------------------------------------


def _run_forward(chunks_by_request, nudge_max=2, stream=False):
    specs = [{"chunks": turn} for turn in chunks_by_request]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=True,
            think_empty_response_placeholder=_PLACEHOLDER,
            think_nudge_enabled=True,
            think_nudge_text=_NUDGE,
            think_nudge_max_attempts=nudge_max,
        ):
            recorder = Recorder("/tmp/nudge_integration.jsonl")
            proxy = Proxy(recorder)
            req = {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            }
            if stream:
                req["stream"] = True
            body = json.dumps(req).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="nudge-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "nudge-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [
                    json.loads(line)
                    for line in open("/tmp/nudge_integration.jsonl")
                    if line.strip()
                ]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count


def test_empty_turn_is_nudged_and_returns_real_answer():
    resp, records, count = _run_forward([_EMPTY_TURN, _ANSWER_TURN], nudge_max=2)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert message["content"] == "the real answer", message

    # Two upstream requests: the empty turn and the nudged retry.
    assert count == 2, count
    # The final record reports a successful nudge.
    final = records[-1]
    assert final["nudge"] == {"attempts": 1, "outcome": "succeeded"}, final
    # The intermediate empty turn was recorded as a nudge trigger.
    triggered = [r for r in records if r.get("nudge", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    assert triggered[0]["nudge"]["attempt"] == 0, triggered[0]


def test_nudge_budget_exhausted_falls_back_to_placeholder():
    # Both requests return empty turns; with a budget of 1 the placeholder
    # floor is reached after one nudge.
    resp, records, count = _run_forward([_EMPTY_TURN, _EMPTY_TURN], nudge_max=1)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    assert message["content"] == _PLACEHOLDER, message
    assert message["reasoning_content"] == "thinking hard", message

    assert count == 2, count
    final = records[-1]
    assert final["nudge"] == {"attempts": 1, "outcome": "exhausted"}, final


def test_no_nudge_when_content_present():
    # A healthy first turn must not trigger a re-submission at all.
    resp, records, count = _run_forward([_ANSWER_TURN], nudge_max=2)
    data = json.loads(resp.body)
    assert data["choices"][0]["message"]["content"] == "the real answer", data
    assert count == 1, count
    assert records[-1].get("nudge") is None, records[-1]
