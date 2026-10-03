"""Runaway-reasoning detection (thinks endlessly, never answers) — unit + integration.

Unit tests cover `RunawayReasoningDetector` (token budget with no content trips;
content/tool-call disarms; below threshold does not) and
`RunawayReasoningPolicy.apply` (appends the nudge, no input mutation).
Integration tests drive the real `Proxy._forward_streaming` against a mock SSE
upstream: a reasoning-only turn that exhausts the window is re-submitted with a
stop-thinking nudge and, on a healthy retry, surfaces the answer; an
unrepentant model is aborted after the budget; a content-producing turn is never
flagged; and the disabled switch leaves the terminal turn as a plain loop.
"""

import asyncio
import json
import random

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.base import Turn, VerdictKind
from app.remediation.runaway import (
    RunawayReasoningDetector,
    RunawayReasoningPolicy,
)
from mock_upstream import MockUpstream, chunk, settings_override

_NUDGE_TEXT = "STOP_THINKING_ANSWER_NOW"


_chunk = chunk


def _reasoning(text: str) -> dict:
    return _chunk({"reasoning_content": text})


def _content(text: str) -> dict:
    return _chunk({"content": text})


# Non-repeating (high-entropy) reasoning: the whole point of #17 is that the
# stream is NOT repetitive, so the loop detector must stay silent and only the
# runaway token budget trips. Seeded for determinism.
_rng = random.Random(1234)
_REASONING_FLOOD = " ".join(
    "".join(_rng.choices("abcdefghijklmnopqrstuvwxyz", k=6)) for _ in range(2500)
)

# A runaway turn: a flood of non-repeating reasoning, then the window is hit.
_RUNAWAY_TURN = [
    _reasoning(_REASONING_FLOOD),
    _chunk({}, finish_reason="length"),
]

# A healthy answer after the nudge.
_ANSWER_TURN = [
    _content("The answer is 42."),
    _chunk({}, finish_reason="stop"),
]


# --- unit tests -------------------------------------------------------


def test_detector_trips_when_budget_exceeded_without_content():
    detector = RunawayReasoningDetector(token_threshold=10)
    detector.note(reasoning="a" * 100)  # ~25 tokens
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind == VerdictKind.RUNAWAY_REASONING
    assert verdict.details["token_threshold"] == 10


def test_detector_below_threshold_does_not_trip():
    detector = RunawayReasoningDetector(token_threshold=1000)
    detector.note(reasoning="a" * 10)
    assert detector.check() is None


def test_detector_disarmed_by_content():
    detector = RunawayReasoningDetector(token_threshold=10)
    detector.note(reasoning="a" * 100)
    detector.note(content="x")
    assert detector.check() is None


def test_detector_tool_call_counts_as_content():
    detector = RunawayReasoningDetector(token_threshold=10)
    detector.note(reasoning="a" * 100)
    detector.note(tool_calls=[{"index": 0}])  # stands in for a tool-call fragment
    assert detector.check() is None


def test_policy_appends_nudge_without_mutation():
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    out = RunawayReasoningPolicy(text=_NUDGE_TEXT, max_attempts=2).apply(Turn(), body)
    assert len(out["messages"]) == 2, out
    assert out["messages"][-1] == {"role": "user", "content": _NUDGE_TEXT}, out
    assert len(body["messages"]) == 1, body


# --- integration harness ---------------------------------------------


def _run_forward(chunks_by_request, *, enabled=True, max_attempts=2, loop_retry=False):
    specs = [{"chunks": turn} for turn in chunks_by_request]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=False,
            think_nudge_enabled=False,
            coast_detection_enabled=False,
            loop_retry_enabled=loop_retry,
            runaway_reasoning_enabled=enabled,
            runaway_reasoning_nudge_text=_NUDGE_TEXT,
            runaway_reasoning_max_attempts=max_attempts,
            runaway_reasoning_token_threshold=2000,
        ):
            recorder = Recorder("/tmp/runaway_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "q"}]}
            ).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="runaway-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "runaway-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [
                    json.loads(line)
                    for line in open("/tmp/runaway_integration.jsonl")
                    if line.strip()
                ]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count


def test_runaway_turn_is_nudged_and_returns_answer():
    resp, records, count = _run_forward([_RUNAWAY_TURN, _ANSWER_TURN])
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert "The answer is 42." in data["choices"][0]["message"]["content"]
    assert count == 2, count
    triggered = [r for r in records if r.get("runaway", {}).get("outcome") == "triggered"]
    assert len(triggered) == 1, records
    # The re-submission carried the nudge instruction.
    resubmitted = triggered[0]["runaway"]["resubmitted_body"]
    assert resubmitted["messages"][-1]["content"] == _NUDGE_TEXT


def test_runaway_budget_exhausted_aborts():
    resp, records, count = _run_forward(
        [_RUNAWAY_TURN, _RUNAWAY_TURN, _RUNAWAY_TURN], max_attempts=2
    )
    assert resp.status_code == 502, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["error"]["type"] == "runaway_reasoning_detected", data
    assert count == 3, count


def test_content_producing_turn_never_flagged():
    resp, records, count = _run_forward([_ANSWER_TURN])
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert count == 1, count
    assert not any(r.get("runaway") for r in records), records


def test_disabled_terminal_turn_is_plain_loop():
    # With the runaway feature off, a reasoning-only length turn falls back to
    # the generic loop verdict (and, with loop-retry also off, aborts as loop).
    resp, records, count = _run_forward([_RUNAWAY_TURN], enabled=False, loop_retry=False)
    assert resp.status_code == 502, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    assert data["error"]["type"] == "loop_detected", data
    assert count == 1, count
    assert not any(r.get("runaway") for r in records), records
