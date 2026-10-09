"""Empty-stop detection (thought then emitted nothing) — unit + integration.

Unit tests cover ``EmptyStopPolicy.should_nudge`` (the exact silent-empty-stop
fingerprint), ``EmptyStopPolicy.apply`` (appends the re-prompt without mutating
the input), and ``token_counts`` (parses LiteLLM's ``completion_tokens_details``
defensively). Integration tests drive the real ``Proxy._forward_streaming``
against a mock SSE upstream that returns a silent empty stop on the first
request and a real answer on the second — proving the rung re-submits with the
re-prompt — plus the budget-exhausted and disabled-switch cases.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import Turn
from app.remediation.empty_stop import EmptyStopPolicy, token_counts
from app.telemetry import telemetry
from mock_upstream import MockUpstream, chunk, settings_override

_NUDGE_TEXT = "PLEASE_PRODUCE_A_VISIBLE_REPLY"
_PLACEHOLDER = "THE_PLACEHOLDER"

_chunk = chunk

_USAGE = {
    "completion_tokens": 72,
    "prompt_tokens": 10,
    "total_tokens": 82,
    "completion_tokens_details": {"reasoning_tokens": 72, "text_tokens": 0},
}


def _request():
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "vet the products"}],
    }


# A silent empty stop: stop, no content, no tool calls, no reasoning, but the
# upstream usage proves the model consumed reasoning tokens with zero text.
_SILENT_EMPTY_TURN = [
    _chunk({}, usage=_USAGE),
    _chunk({}, finish_reason="stop"),
]

# A healthy turn: the model actually emits a visible answer.
_ANSWER_TURN = [
    _chunk({"role": "assistant", "content": "the real answer"}),
    _chunk({}, finish_reason="stop"),
]


# --- unit tests -------------------------------------------------------


def test_should_nudge_silent_empty_stop():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge("stop", "", None, "", 72, 0)
        is True
    )


def test_should_nudge_requires_empty_content():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge(
            "stop", "something", None, "", 72, 0
        )
        is False
    )


def test_should_nudge_ignores_tool_calls():
    tool_calls = [{"id": "c", "function": {"name": "f", "arguments": "{}"}}]
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge(
            "stop", "", tool_calls, "", 72, 0
        )
        is False
    )


def test_should_nudge_requires_empty_reasoning():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge(
            "stop", "", None, "internal thinking", 72, 0
        )
        is False
    )


def test_should_nudge_requires_stop_finish_reason():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge(
            "length", "", None, "", 72, 0
        )
        is False
    )


def test_should_nudge_requires_token_signal():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge(
            "stop", "", None, "", None, None
        )
        is False
    )


def test_should_nudge_requires_positive_reasoning_tokens():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge("stop", "", None, "", 0, 0)
        is False
    )


def test_should_nudge_requires_zero_text_tokens():
    assert (
        EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).should_nudge("stop", "", None, "", 72, 1)
        is False
    )


def test_applies_uses_turn_token_fields():
    p = EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2)
    turn = Turn(finish_reason="stop", reasoning_tokens=72, text_tokens=0)
    assert p.applies(turn, {}) is True


def test_apply_appends_nudge_and_does_not_mutate_input():
    req = _request()
    out = EmptyStopPolicy(text=_NUDGE_TEXT, max_attempts=2).apply(Turn(), req)
    assert len(out["messages"]) == len(req["messages"]) + 1, out
    assert out["messages"][-1] == {"role": "user", "content": _NUDGE_TEXT}, out
    assert len(req["messages"]) == 1, req


def test_token_counts_parses_completion_details():
    assert token_counts({"usage": _USAGE}) == (72, 0)


def test_token_counts_absent_usage_is_none():
    assert token_counts({}) == (None, None)
    assert token_counts({"usage": {}}) == (None, None)
    assert token_counts(None) == (None, None)


def test_token_counts_invalid_values_are_none():
    assert token_counts(
        {"usage": {"completion_tokens_details": {"reasoning_tokens": "nope", "text_tokens": 0}}}
    ) == (None, 0)


# --- integration harness ---------------------------------------------


def _run_forward(chunks_by_request, record_path, empty_stop_max=2, enabled=True):
    specs = [{"chunks": turn} for turn in chunks_by_request]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            stall_detection_enabled=False,
            runaway_reasoning_enabled=False,
            think_cleanup_enabled=True,
            think_empty_response_placeholder=_PLACEHOLDER,
            extract_enabled=False,
            coast_detection_enabled=False,
            empty_stop_detection_enabled=enabled,
            empty_stop_nudge_text=_NUDGE_TEXT,
            empty_stop_max_attempts=empty_stop_max,
            tool_call_guard_enabled=False,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(_request()).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="empty-stop-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "empty-stop-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [json.loads(line) for line in open(record_path) if line.strip()]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count, mock.json_bodies()


def test_empty_stop_is_reprompted_and_returns_answer(tmp_path):
    record_path = tmp_path / "empty_stop_integration.jsonl"
    resp, records, count, bodies = _run_forward([_SILENT_EMPTY_TURN, _ANSWER_TURN], record_path)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == "the real answer", message

    # Two upstream requests: the silent empty stop and the re-prompted retry.
    assert count == 2, count
    assert bodies[1]["messages"][-1] == {"role": "user", "content": _NUDGE_TEXT}, bodies[1]

    triggered = [r for r in records if r.get("empty_stop", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    # The discarded empty turn lives in `pre_remediation_body` (not `response_body`).
    assert triggered[0]["response_body"] is None, triggered[0]
    pre = triggered[0]["pre_remediation_body"]
    assert '"assistant"' in pre, pre
    assert '"finish_reason": "stop"' in pre, pre

    final = records[-1]
    assert final["empty_stop"] == {"attempts": 1, "outcome": "succeeded"}, final


def test_empty_stop_budget_exhausted_falls_back_to_placeholder(tmp_path):
    record_path = tmp_path / "empty_stop_exhausted_integration.jsonl"
    before = telemetry.snapshot()["counters"].get("empty_stop_aborts_total", 0)
    resp, records, count, _ = _run_forward(
        [_SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN],
        record_path,
        empty_stop_max=2,
    )
    after = telemetry.snapshot()["counters"].get("empty_stop_aborts_total", 0)
    assert after == before + 1, (before, after)

    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    # The rung gave up: no tool calls leak, and the placeholder floor applies.
    assert not message.get("tool_calls"), message
    assert message["content"] == _PLACEHOLDER, message

    assert count == 3, count
    final = records[-1]
    assert final["empty_stop"] == {"attempts": 2, "outcome": "exhausted"}, final


def test_empty_stop_disabled_is_noop(tmp_path):
    record_path = tmp_path / "empty_stop_disabled_integration.jsonl"
    before = telemetry.snapshot()["counters"].get("empty_stop_aborts_total", 0)
    resp, records, count, _ = _run_forward([_SILENT_EMPTY_TURN], record_path, enabled=False)
    after = telemetry.snapshot()["counters"].get("empty_stop_aborts_total", 0)
    assert after == before, (before, after)

    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert count == 1, count
    assert not records[-1].get("empty_stop"), records[-1]
