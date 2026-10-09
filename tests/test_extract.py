"""Extraction (recover answer from think-only turns) — unit + integration.

Unit tests cover ``ExtractionPolicy`` seed/read-exec dispatch, budget, and
exact ``gauze_read`` tool-name matching. Integration tests drive the real
``Proxy._forward_streaming`` against a mock SSE upstream: a think-only turn
seeds the reasoning resource, a ``gauze_read`` tool call pages through it, and
a final turn yields the visible answer — plus a budget-exhaustion case and a
foreign-tool-call pass-through case.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import Turn
from app.remediation.extract import RESOURCE_NAME, TOOL_NAME, ExtractionPolicy
from mock_upstream import MockUpstream, chunk, settings_override

_INSTRUCTION = "PLEASE_PAGE_AND_ANSWER"
_PLACEHOLDER = "THE_PLACEHOLDER"

_chunk = chunk


def _turn(**delta):
    return [
        _chunk({"role": "assistant", **delta}),
        _chunk({}, finish_reason="stop"),
    ]


_THINK_TURN = _turn(reasoning_content="line one\nline two\nline three")
_ANSWER_TURN = _turn(content="the real answer")


def _read_turn(line_from=2, line_count=1):
    return [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": TOOL_NAME,
                            "arguments": json.dumps(
                                {
                                    "resource": RESOURCE_NAME,
                                    "line_from": line_from,
                                    "line_count": line_count,
                                }
                            ),
                        },
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]


# --- unit tests -------------------------------------------------------


def _policy(max_attempts=3):
    return ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=max_attempts)


def test_think_only_turn_seeds():
    p = _policy()
    turn = Turn(finish_reason="stop", reasoning="line one\nline two")
    assert p.applies(turn, {}) is True


def test_seed_ignores_visible_content():
    p = _policy()
    turn = Turn(finish_reason="stop", content="hello", reasoning="thinking")
    assert p.applies(turn, {}) is False


def test_seed_ignores_tool_calls():
    p = _policy()
    turn = Turn(
        finish_reason="stop",
        tool_calls=[{"id": "c", "function": {"name": "f", "arguments": "{}"}}],
        reasoning="thinking",
    )
    assert p.applies(turn, {}) is False


def test_seed_requires_stop_finish_reason():
    p = _policy()
    assert p.applies(Turn(finish_reason="length", reasoning="thinking"), {}) is False


def test_seed_requires_reasoning():
    p = _policy()
    assert p.applies(Turn(finish_reason="stop", reasoning=""), {}) is False


def test_seed_captures_resource_and_appends_tool():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    p = _policy()
    turn = Turn(finish_reason="stop", reasoning="line one\nline two")
    out = p.apply(turn, body)
    assert p.resource_chars == len("line one\nline two")
    assert p.resource_lines == 2
    assert p.last_mode == "seed"
    assert p.last_read_args is None
    assert out["messages"][-1] == {"role": "user", "content": _INSTRUCTION}
    assert out["tools"][-1]["function"]["name"] == TOOL_NAME


def test_seed_does_not_mutate_input():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    _policy().apply(Turn(finish_reason="stop", reasoning="x"), body)
    assert body.get("tools") is None
    assert len(body["messages"]) == 1


def test_read_exec_fires_only_after_seed():
    p = _policy()
    turn = Turn(
        finish_reason="tool_calls",
        tool_calls=[{"function": {"name": TOOL_NAME, "arguments": "{}"}}],
    )
    assert p.applies(turn, {}) is False


def test_foreign_tool_call_is_not_intercepted():
    p = _policy()
    p.apply(Turn(finish_reason="stop", reasoning="x"), {})
    turn = Turn(
        finish_reason="tool_calls",
        tool_calls=[{"id": "c", "function": {"name": "client_tool", "arguments": "{}"}}],
    )
    assert p.applies(turn, {}) is False


def test_mixed_tool_calls_are_not_intercepted():
    p = _policy()
    p.apply(Turn(finish_reason="stop", reasoning="x"), {})
    turn = Turn(
        finish_reason="tool_calls",
        tool_calls=[
            {"id": "a", "function": {"name": TOOL_NAME, "arguments": "{}"}},
            {"id": "b", "function": {"name": "client_tool", "arguments": "{}"}},
        ],
    )
    assert p.applies(turn, {}) is False


def test_tool_name_matched_exactly():
    p = _policy()
    p.apply(Turn(finish_reason="stop", reasoning="x"), {})
    turn = Turn(
        finish_reason="tool_calls",
        tool_calls=[{"id": "c", "function": {"name": "gauze_read_extra", "arguments": "{}"}}],
    )
    assert p.applies(turn, {}) is False


def test_read_exec_slices_resource_and_replays_tool_result():
    p = _policy()
    p.apply(Turn(finish_reason="stop", reasoning="line one\nline two\nline three"), {})
    turn = Turn(
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": "call_read_1",
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "arguments": json.dumps(
                        {"resource": RESOURCE_NAME, "line_from": 2, "line_count": 1}
                    ),
                },
            }
        ],
    )
    out = p.apply(turn, {})
    assert p.last_mode == "read"
    assert p.last_read_args == [{"resource": RESOURCE_NAME, "line_from": 2, "line_count": 1}]
    assert out["messages"][-2]["role"] == "assistant"
    assert out["messages"][-2]["tool_calls"] == turn.tool_calls
    assert out["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_read_1",
        "content": "line two",
    }


def test_read_slice_defaults_line_count_to_remainder():
    p = _policy()
    p.apply(Turn(finish_reason="stop", reasoning="a\nb\nc"), {})
    assert p._read_slice({"resource": RESOURCE_NAME, "line_from": 2}) == "b\nc"


# --- integration harness ---------------------------------------------


def _run_forward(chunks_by_request, extract_max=3, stream=False, body=None):
    specs = [{"chunks": turn} for turn in chunks_by_request]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            stall_detection_enabled=False,
            runaway_reasoning_enabled=False,
            think_cleanup_enabled=True,
            think_empty_response_placeholder=_PLACEHOLDER,
            extract_enabled=True,
            extract_instruction=_INSTRUCTION,
            extract_max_attempts=extract_max,
            coast_detection_enabled=False,
            tool_call_guard_enabled=False,
        ):
            recorder = Recorder("/tmp/extract_integration.jsonl")
            proxy = Proxy(recorder)
            req = body or {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            if stream:
                req["stream"] = True
            req_bytes = json.dumps(req).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="extract-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=req_bytes,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "extract-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [
                    json.loads(line)
                    for line in open("/tmp/extract_integration.jsonl")
                    if line.strip()
                ]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count, mock.json_bodies()


def test_think_only_turn_is_extracted_and_returns_real_answer():
    resp, records, count, bodies = _run_forward([_THINK_TURN, _read_turn(), _ANSWER_TURN])
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == "the real answer", message

    # Three upstream requests: think-only (seed), read (read-exec), answer.
    assert count == 3, count
    final = records[-1]
    assert final["extract"] == {"attempts": 2, "outcome": "succeeded"}, final

    # Seed re-submission carried the instruction + the gauze_read tool.
    assert bodies[1]["messages"][-1] == {"role": "user", "content": _INSTRUCTION}
    assert bodies[1]["tools"][-1]["function"]["name"] == TOOL_NAME
    # Read-exec re-submission replayed the tool result as the bounded slice.
    assert bodies[2]["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_read_1",
        "content": "line two",
    }

    # Two triggered passes: seed then read.
    triggered = [r for r in records if r.get("extract", {}).get("outcome") == "triggered"]
    assert len(triggered) == 2, records
    assert triggered[0]["extract"]["mode"] == "seed", triggered[0]
    assert triggered[0]["extract"]["resource_lines"] == 3, triggered[0]
    assert triggered[0]["extract"]["read_args"] is None, triggered[0]
    assert triggered[1]["extract"]["mode"] == "read", triggered[1]
    assert triggered[1]["extract"]["read_args"] == [
        {"resource": RESOURCE_NAME, "line_from": 2, "line_count": 1}
    ], triggered[1]


def test_extract_budget_exhausted_falls_back_to_placeholder():
    # Keep looping on gauze_read: seed + one read consumes the budget of 2,
    # and the final read finds the rung spent — the placeholder floor applies.
    resp, records, count, _ = _run_forward([_THINK_TURN, _read_turn(), _read_turn()], extract_max=2)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == _PLACEHOLDER, message

    assert count == 3, count
    final = records[-1]
    assert final["extract"] == {"attempts": 2, "outcome": "exhausted"}, final


def test_foreign_tool_call_passes_through_untouched():
    req = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {"name": "client_tool", "description": "client tool"},
            }
        ],
    }
    foreign = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_client",
                        "type": "function",
                        "function": {"name": "client_tool", "arguments": "{}"},
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp, records, count, _ = _run_forward([foreign], body=req)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["tool_calls"][0]["function"]["name"] == "client_tool", message
    assert count == 1, count
    assert records[-1].get("extract") is None, records[-1]


def test_large_reasoning_is_never_resubmitted_in_full():
    big = "\n".join(f"reasoning line {i}" for i in range(1, 101))
    resp, records, count, bodies = _run_forward(
        [_turn(reasoning_content=big), _read_turn(line_from=50, line_count=1), _ANSWER_TURN]
    )
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert json.loads(resp.body)["choices"][0]["message"]["content"] == "the real answer"
    assert count == 3, count
    # The full stream must never appear in any single upstream request.
    for body in bodies:
        assert big not in json.dumps(body), body
