"""Whole-ladder remediation caps (rounds + wall-clock) — unit + integration.

The remediation ladder already bounds each rung with its own ``max_attempts``;
these caps bound the *entire* ladder per request. Unit tests pin the pure
``_remediation_exhausted`` decision; integration tests drive the real
``Proxy._forward_streaming`` against a mock SSE upstream that re-fires the
empty-stop rung on every turn, proving each cap stops the ladder before the
per-rung budget would.
"""

import asyncio
import json

from app.proxy import Proxy, _remediation_exhausted
from app.recorder import Recorder
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

# A silent empty stop: stop, no content, no tool calls, no reasoning, but the
# upstream usage proves the model consumed reasoning tokens with zero text.
_SILENT_EMPTY_TURN = [
    _chunk({}, usage=_USAGE),
    _chunk({}, finish_reason="stop"),
]


def _request():
    return {
        "model": "test",
        "messages": [{"role": "user", "content": "vet the products"}],
    }


class _FakeClock:
    """A wall-clock stand-in the tests advance by hand."""

    def __init__(self, start: float) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --- unit tests: the cap decision ------------------------------------


def test_remediation_exhausted_rounds_cap():
    assert _remediation_exhausted(rounds=2, max_rounds=2, deadline=1000.0, now=0.0) == "rounds"


def test_remediation_exhausted_time_cap():
    assert _remediation_exhausted(rounds=0, max_rounds=10, deadline=10.0, now=10.0) == "time"


def test_remediation_exhausted_none():
    assert _remediation_exhausted(rounds=1, max_rounds=2, deadline=10.0, now=9.0) is None


def test_remediation_exhausted_rounds_wins_when_both_trip():
    assert _remediation_exhausted(rounds=2, max_rounds=2, deadline=10.0, now=10.0) == "rounds"


# --- integration harness ---------------------------------------------


def _run_forward(
    chunks_by_request,
    record_path,
    *,
    max_rounds=16,
    max_seconds=300.0,
    started=None,
    clock=None,
):
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
            empty_stop_detection_enabled=True,
            empty_stop_nudge_text=_NUDGE_TEXT,
            empty_stop_max_attempts=5,
            tool_call_guard_enabled=False,
            max_remediation_rounds=max_rounds,
            max_remediation_seconds=max_seconds,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(_request()).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="remediation-caps-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "remediation-caps-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                    started=started,
                    clock=clock,
                )
                with open(record_path) as f:
                    records = [json.loads(line) for line in f if line.strip()]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count


def test_rounds_cap_stops_repeated_rung_before_per_rung_budget(tmp_path):
    record_path = tmp_path / "remediation_rounds_cap.jsonl"
    before = telemetry.snapshot()["counters"].get("remediation_rounds_exhausted_total", 0)
    resp, records, count = _run_forward(
        # The final spec repeats, so the rung would keep firing indefinitely
        # without the whole-ladder cap (per-rung budget is 5).
        [_SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN],
        record_path,
        max_rounds=2,
    )
    after = telemetry.snapshot()["counters"].get("remediation_rounds_exhausted_total", 0)
    assert after == before + 1, (before, after)

    # One initial turn plus two re-submissions, then the rounds cap tripped.
    assert count == 3, count

    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == _PLACEHOLDER, message
    assert not message.get("tool_calls"), message

    final = records[-1]
    assert final["rounds"] == 2, final
    assert final["remediation_exhausted"] == "rounds", final
    assert final["remediation_seconds"] >= 0, final
    assert final["empty_stop"] == {"attempts": 2, "outcome": "exhausted"}, final


def test_time_cap_stops_repeated_rung_via_short_budget(tmp_path):
    record_path = tmp_path / "remediation_time_cap.jsonl"
    clock = _FakeClock(0.0)
    clock.advance(1.0)  # trip the 1.0s budget before the first re-submission check
    before = telemetry.snapshot()["counters"].get("remediation_time_exhausted_total", 0)
    resp, records, count = _run_forward(
        [_SILENT_EMPTY_TURN, _SILENT_EMPTY_TURN],
        record_path,
        max_rounds=100,
        max_seconds=1.0,
        started=0.0,
        clock=clock,
    )
    after = telemetry.snapshot()["counters"].get("remediation_time_exhausted_total", 0)
    assert after == before + 1, (before, after)

    # The first turn would re-fire the rung, but the wall-clock cap trips first.
    assert count == 1, count

    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == _PLACEHOLDER, message

    final = records[-1]
    assert final["rounds"] == 0, final
    assert final["remediation_exhausted"] == "time", final
    assert final["remediation_seconds"] == 1.0, final
