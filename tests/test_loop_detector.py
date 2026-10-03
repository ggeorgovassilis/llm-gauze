"""Unit tests for the thinking-loop detector.

Runs with plain Python (stdlib only) inside the container:

    docker compose exec -T gateway python - < tests/test_loop_detector.py
"""

import random
import traceback

try:
    from app.remediation.base import VerdictKind
    from app.remediation.loop import ThinkingLoopDetector  # noqa: F401
except ImportError:  # pragma: no cover - run on host without the container
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from app.remediation.base import VerdictKind
    from app.remediation.loop import ThinkingLoopDetector  # noqa: F401


def _make(**kwargs) -> ThinkingLoopDetector:
    # Small window so tests run on short payloads; fraction 0.5 arms at half
    # the window. The ratio threshold matches production (0.15): a verbatim
    # loop compresses to ~0.004 while varied text stays well above it.
    defaults = dict(
        window_bytes=2000,
        min_output_fraction=0.5,
        compression_ratio=0.15,
    )
    defaults.update(kwargs)
    return ThinkingLoopDetector(**defaults)


_VOCAB = [
    "quick",
    "brown",
    "fox",
    "lazy",
    "dog",
    "river",
    "mountain",
    "silver",
    "bright",
    "quiet",
    "ancient",
    "harbour",
    "meadow",
    "galaxy",
    "telescope",
    "manuscript",
    "symphony",
    "geologist",
    "conductor",
    "pruned",
    "roses",
    "saplings",
    "fireflies",
    "astronomer",
    "catalogue",
    "distant",
    "breeze",
    "scent",
    "rain",
    "valley",
    "train",
    "sunrise",
    "chef",
    "stew",
    "herbs",
    "salt",
    "wooden",
    "boat",
    "drifted",
    "inscription",
    "debated",
    "historian",
    "limestone",
    "gorge",
    "faint",
    "signal",
    "detected",
    "sauce",
    "delicate",
    "orchestra",
    "rehearsed",
    "evening",
    "stratified",
    "language",
    "nature",
    "mathematics",
    "written",
    "weather",
    "afternoon",
    "suddenly",
    "catalogued",
]


def _varied_text(chunks: int, words: int = 12) -> str:
    """Genuinely non-repetitive prose (no fixed template, no cycling)."""
    rng = random.Random(1234)
    parts = []
    for _ in range(chunks):
        parts.append(" ".join(rng.choice(_VOCAB) for _ in range(words)) + ". ")
    return "".join(parts)


def test_repetitive_loop():
    detector = _make()
    verdict = None
    for _ in range(80):
        detector.note(content="the cat sat on the mat. ")
        verdict = detector.check() or verdict
    if verdict is None:
        verdict = detector.check()
    assert verdict is not None, "expected a loop, none detected"
    assert verdict.kind == VerdictKind.LOOP, verdict
    assert "entropy" in verdict.reason, verdict


def test_diverse_output_no_loop():
    detector = _make()
    text = _varied_text(300)
    # Feed in arbitrary-sized chunks to exercise partial-token accumulation.
    i = 0
    while i < len(text):
        step = (i * 7 + 11) % 40 + 1
        detector.note(content=text[i : i + step])
        assert detector.check() is None
        i += step
    assert detector.check() is None


def test_short_output_below_gate_is_never_a_loop():
    # Below min_output_bytes the detector is not armed: even blatantly
    # repetitive text must not be flagged while the stream is still warming up.
    detector = _make(window_bytes=10000, min_output_fraction=0.5)
    # ~4000 bytes, under the 5000-byte gate.
    for _ in range(100):
        detector.note(content="done — nothing left to vet in this chunk\n")
        assert detector.check() is None
    assert detector.check() is None


def test_arming_gate():
    detector = _make(window_bytes=2000, min_output_fraction=0.5)
    # 400 chars < 1000-byte gate -> not armed yet, repetitive or not.
    for _ in range(10):
        detector.note(content="the cat sat on the mat. ")
        assert detector.check() is None
    # Cross the gate with more repetition -> now armed and detected.
    verdict = None
    for _ in range(60):
        detector.note(content="the cat sat on the mat. ")
        verdict = detector.check() or verdict
    if verdict is None:
        verdict = detector.check()
    assert verdict is not None and verdict.kind == VerdictKind.LOOP, verdict


def test_enumeration_of_tool_results_does_not_loop():
    # Regression: the orchestrator enumerates the 12 vetter replies, many of
    # which are the identical "done — nothing left to vet in this chunk". That
    # enumeration is short relative to the window, so it must not trip the
    # detector (previously it did, via n-gram recurrence).
    detector = _make(window_bytes=32768, min_output_fraction=0.25)
    lines = [
        "1. done — nothing left to vet in this chunk",
        "2. kept NVIDIA GeForce GTX 1070",
        "3. done — nothing left to vet in this chunk",
        "4. done — nothing left to vet in this chunk",
        "5. kept NVIDIA Quadro RTX 6000",
        "6. done — nothing left to vet in this chunk",
        "7. done — nothing left to vet in this chunk",
        "8. kept NVIDIA GeForce RTX 5060",
        "9. done — nothing left to vet in this chunk",
        "10. done — nothing left to vet in this chunk",
        "11. kept NVIDIA Tesla V100 PCIe 16GB",
        "12. done — nothing left to vet in this chunk",
    ]
    for line in lines:
        detector.note(content=line + "\n")
        assert detector.check() is None
    assert detector.check() is None


def test_reset_clears_state():
    detector = _make(window_bytes=500, min_output_fraction=0.5)
    verdict = None
    for _ in range(30):
        detector.note(content="the cat sat on the mat. ")
        verdict = detector.check() or verdict
    assert verdict is not None and verdict.kind == VerdictKind.LOOP

    detector.reset()
    # After reset the buffer is empty, so the detector is disarmed again.
    detector.note(content="the cat sat on the mat. ")
    assert detector.check() is None


def test_compression_ratio_orders_repetition():
    detector = _make()
    repetitive = ("the cat sat on the mat. " * 30).strip()
    novel = (
        "the cat sat on the mat. a dog barked loudly in the night. "
        "the sun rose over the quiet hills. many birds flew across the sky."
    )
    assert detector._compression_ratio(repetitive) < detector._compression_ratio(novel)


def test_note_empty_string_is_noop():
    detector = _make()
    detector.note(content="")
    assert detector.check() is None


def test_compression_ratio_of_empty_text_is_one():
    assert _make()._compression_ratio("") == 1.0


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
