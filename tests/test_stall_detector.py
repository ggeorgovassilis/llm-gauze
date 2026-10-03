"""Unit tests for the stalled-stream detector.

Runs with plain Python (stdlib only) inside the container:

    docker compose exec -T gateway python - < tests/test_stall_detector.py
"""

import traceback

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
