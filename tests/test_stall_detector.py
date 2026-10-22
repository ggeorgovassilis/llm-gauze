"""Unit tests for the stalled-stream detector."""

try:
    from app.remediation.base import VerdictKind
    from app.remediation.stall import StallDetector
except ImportError:  # pragma: no cover - run on host without the container
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from app.remediation.base import VerdictKind
    from app.remediation.stall import StallDetector


class _FakeClock:
    """A monotonic clock the tests can advance by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make(ttft_seconds=120.0, gap_seconds=60.0, start=0.0):
    clock = _FakeClock(start)
    detector = StallDetector(
        ttft_seconds=ttft_seconds,
        gap_seconds=gap_seconds,
        clock=clock,
    )
    return detector, clock


def _make_speed(
    *,
    min_speed_tps=1.0,
    speed_window_seconds=2.0,
    gap_seconds=60.0,
    ttft_seconds=10.0,
    speed_floor_enabled=True,
    start=0.0,
):
    clock = _FakeClock(start)
    detector = StallDetector(
        ttft_seconds=ttft_seconds,
        gap_seconds=gap_seconds,
        min_speed_tps=min_speed_tps,
        speed_window_seconds=speed_window_seconds,
        speed_floor_enabled=speed_floor_enabled,
        clock=clock,
    )
    return detector, clock


def test_stall_before_first_token():
    detector, clock = _make(ttft_seconds=10.0, gap_seconds=5.0)
    clock.advance(9.9)
    assert detector.check() is None
    clock.advance(0.2)  # 10.1s — past the TTFT budget
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind == VerdictKind.STALLED, verdict
    assert "first token" in verdict.reason, verdict
    assert verdict.details["saw_first_token"] is False


def test_no_stall_when_first_token_arrives_in_time():
    detector, clock = _make(ttft_seconds=10.0, gap_seconds=5.0)
    clock.advance(8.0)
    detector.note(content="x")
    # Once a token lands, the (tighter) gap timer governs, not TTFT.
    clock.advance(4.9)
    assert detector.check() is None
    clock.advance(0.2)  # 5.1s after the last token
    assert detector.check() is not None


def test_gap_timer_resets_on_each_token():
    detector, clock = _make(ttft_seconds=10.0, gap_seconds=5.0)
    detector.note(content="x")
    clock.advance(4.0)
    detector.note(content="x")  # resets the gap timer
    clock.advance(4.0)
    assert detector.check() is None
    clock.advance(1.1)  # 5.1s since the last token
    assert detector.check() is not None


def test_remaining_decreases_then_goes_non_positive():
    detector, clock = _make(ttft_seconds=10.0, gap_seconds=5.0)
    assert detector.remaining() == 10.0
    clock.advance(5.0)
    assert detector.remaining() == 5.0
    clock.advance(5.0)
    assert detector.remaining() <= 0


def test_verdict_after_first_token_mentions_gap():
    detector, clock = _make(ttft_seconds=10.0, gap_seconds=5.0)
    detector.note(content="x")
    clock.advance(5.0)
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind == VerdictKind.STALLED, verdict
    assert "content-bearing token" in verdict.reason, verdict
    assert verdict.details["saw_first_token"] is True
    assert verdict.details["elapsed_seconds"] == 5.0


def test_speed_floor_trips_slow_trickle_within_window():
    """A ~0.25 token/s trickle (1 char/2s) must trip the floor promptly."""
    detector, clock = _make_speed(min_speed_tps=1.0, speed_window_seconds=2.0)
    detector.note(content="a")  # 1 char = 0.25 tokens; floor needs 8 chars/2s
    clock.advance(1.9)
    assert detector.check() is None
    clock.advance(0.1)  # 2.0s after the first token -> below the floor
    verdict = detector.check()
    assert verdict is not None
    assert verdict.kind == VerdictKind.STALLED, verdict
    assert "minimum speed" in verdict.reason, verdict
    assert verdict.details["min_speed_tps"] == 1.0


def test_speed_floor_passes_steady_two_tps_stream():
    """A steady 2 tokens/s stream (4 chars/token) must never trip the floor."""
    detector, clock = _make_speed(min_speed_tps=1.0, speed_window_seconds=2.0)
    for _ in range(20):
        detector.note(content="abcd")  # 4 chars = 1 token
        assert detector.check() is None
        clock.advance(0.5)  # 2 tokens/s
    assert detector.check() is None
    assert detector.remaining() > 0


def test_speed_floor_passes_bursty_ten_tps_stream():
    """A 10 tokens/s stream must stay well clear of a 1 t/s floor."""
    detector, clock = _make_speed(min_speed_tps=1.0, speed_window_seconds=2.0)
    for _ in range(40):
        detector.note(content="abcd")
        assert detector.check() is None
        clock.advance(0.1)  # 10 tokens/s
    assert detector.check() is None


def test_speed_floor_disabled_restores_gap_only_behaviour():
    """With the floor off, the gap timer alone governs a trickle."""
    detector, clock = _make_speed(
        min_speed_tps=1.0,
        speed_window_seconds=2.0,
        gap_seconds=60.0,
        speed_floor_enabled=False,
    )
    detector.note(content="a")
    clock.advance(2.5)  # would have tripped the (disabled) floor
    assert detector.check() is None
    clock.advance(60.0)  # past the gap timer
    verdict = detector.check()
    assert verdict is not None
    assert "content-bearing token" in verdict.reason, verdict


def test_speed_floor_counts_tool_call_characters():
    """Tool-call name/arguments count toward the speed floor."""
    detector, clock = _make_speed(min_speed_tps=1.0, speed_window_seconds=2.0)
    # 6 + 12 = 18 chars of function name/arguments -> 4.5 tokens, above floor.
    detector.note(
        tool_calls=[
            {"index": 0, "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
        ]
    )
    clock.advance(1.9)
    assert detector.check() is None
