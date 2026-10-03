"""Unit tests for the typed verdict routing registry (#91).

The verdict kinds used to be raw strings ("loop", "stalled",
"runaway_reasoning") switched on in several places in the proxy. These tests
cover the typed :class:`VerdictKind` enum, the single :class:`VerdictRoute`
registry that maps each kind to its abort response (message, HTTP status) and
telemetry (outcome label, abort counter), and the construction of typed
verdicts by the detectors.
"""

import json

import pytest
from app.config import settings
from app.proxy import _abort_error_response
from app.remediation.base import DiagnosisCode, StreamVerdict, VerdictKind
from app.remediation.context import CONTEXT_WINDOW_CODE
from app.remediation.loop import ThinkingLoopDetector
from app.remediation.runaway import RunawayReasoningDetector
from app.remediation.stall import StallDetector
from app.remediation.verdicts import VERDICT_ROUTES, route_for
from mock_upstream import settings_override


class _FakeClock:
    """A monotonic clock frozen at a fixed instant (enough for construction)."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


# --- the registry ----------------------------------------------------


def test_registry_covers_every_verdict_kind():
    assert set(VERDICT_ROUTES) == set(VerdictKind)


def test_route_for_accepts_member_or_string():
    assert route_for(VerdictKind.LOOP) is VERDICT_ROUTES[VerdictKind.LOOP]
    assert route_for("loop") is VERDICT_ROUTES[VerdictKind.LOOP]


def test_route_for_unknown_kind_raises():
    with pytest.raises(ValueError):
        route_for("mystery")


def test_telemetry_mapping():
    loop = VERDICT_ROUTES[VerdictKind.LOOP]
    assert (loop.outcome, loop.abort_metric) == ("loop_aborted", "loop_aborts_total")
    assert loop.stream_labeled is True

    stall = VERDICT_ROUTES[VerdictKind.STALLED]
    assert (stall.outcome, stall.abort_metric) == ("stalled", "stall_aborts_total")
    assert stall.stream_labeled is False

    runaway = VERDICT_ROUTES[VerdictKind.RUNAWAY_REASONING]
    assert (runaway.outcome, runaway.abort_metric) == (
        "runaway_reasoning_aborted",
        "runaway_reasoning_aborts_total",
    )
    assert runaway.stream_labeled is False


def test_abort_status_reads_live_settings():
    with settings_override(loop_abort_status=418):
        assert VERDICT_ROUTES[VerdictKind.LOOP].abort_status == 418


# --- abort response routing ------------------------------------------


def test_abort_response_uses_registry():
    cases = [
        (VerdictKind.LOOP, settings.loop_abort_status),
        (VerdictKind.STALLED, settings.stall_abort_status),
        (VerdictKind.RUNAWAY_REASONING, settings.runaway_reasoning_abort_status),
    ]
    for kind, status in cases:
        route = VERDICT_ROUTES[kind]
        resp = _abort_error_response(StreamVerdict(kind=kind, reason="r"))
        data = json.loads(resp.body)
        assert resp.status_code == status, kind
        assert data["error"]["type"] == f"{kind.value}_detected", kind
        assert data["error"]["message"] == route.message, kind
        assert data["error"]["reason"] == "r"


# --- verdict construction --------------------------------------------


def test_loop_detector_constructs_typed_kind():
    detector = ThinkingLoopDetector(
        window_bytes=2000,
        min_output_fraction=0.5,
        compression_ratio=0.15,
    )
    verdict = None
    for _ in range(80):
        detector.note(content="the cat sat on the mat. ")
        verdict = detector.check() or verdict
    assert verdict is not None
    assert verdict.kind is VerdictKind.LOOP


def test_stall_detector_constructs_typed_kind():
    detector = StallDetector(ttft_seconds=0.0, gap_seconds=0.0, clock=_FakeClock(0.0))
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind is VerdictKind.STALLED


def test_runaway_detector_constructs_typed_kind():
    detector = RunawayReasoningDetector(token_threshold=1)
    detector.note(reasoning="a" * 100)
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind is VerdictKind.RUNAWAY_REASONING


def test_context_window_code_is_typed():
    assert CONTEXT_WINDOW_CODE is DiagnosisCode.CONTEXT_WINDOW_EXCEEDED
    assert DiagnosisCode.CONTEXT_WINDOW_EXCEEDED.value == "context_window_exceeded"
