"""Unit tests for the thinking-loop detector.

Runs with plain Python (stdlib only) inside the container:

    docker compose exec -T gateway python - < tests/test_loop_detector.py
"""

import traceback

try:
    from app.remediation.loop import ThinkingLoopDetector  # noqa: F401
except ImportError:  # pragma: no cover - run on host without the container
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from app.remediation.loop import ThinkingLoopDetector  # noqa: F401


def _make(**kwargs) -> ThinkingLoopDetector:
    defaults = dict(
        window_sentences=10,
        jaccard_threshold=0.6,
        min_loop_count=2,
        ngram_size=3,
        compression_ratio=0.22,
        compression_min_chars=300,
    )
    defaults.update(kwargs)
    return ThinkingLoopDetector(**defaults)


def test_verbatim_loop():
    detector = _make()
    sentences = [
        "First, let us initialize variable total = 0.",
        "Next, loop through each element x in the list.",
        "First, let us initialize variable total = 0.",
        "Next, loop through each element x in the list.",
        "First, let us initialize variable total = 0.",
    ]
    for sentence in sentences:
        verdict = detector.feed(sentence)
        if verdict is not None:
            assert verdict.kind == "loop", verdict
            assert "recurrence" in verdict.reason, verdict
            return
    raise AssertionError("expected a loop, none detected")


def test_slight_drift_loop():
    detector = _make()
    sentences = [
        "Let me carefully reconsider whether the total should be "
        "initialised to zero before the loop begins.",
        "Let me carefully reconsider whether the count should be "
        "initialised to zero before the loop begins.",
        "Let me carefully reconsider whether the sum should be "
        "initialised to zero before the loop begins.",
    ]
    for sentence in sentences:
        verdict = detector.feed(sentence)
        if verdict is not None:
            assert verdict.kind == "loop", verdict
            return
    raise AssertionError("expected a drift loop, none detected")


def test_no_loop():
    detector = _make()
    text = [
        "The capital of France is Paris.",
        "Water boils at one hundred degrees Celsius.",
        "Shakespeare wrote many famous plays.",
        "Python is a popular programming language.",
    ]
    for sentence in text:
        assert detector.feed(sentence) is None, sentence
    assert detector.flush() is None


def test_partial_chunk_assembly():
    detector = _make(min_loop_count=1)
    verdict = None
    for _ in range(2):
        verdict = detector.feed("First, let us initialize vari") or verdict
        verdict = detector.feed("able total = 0.") or verdict
    assert verdict is not None, "expected loop after partial chunks"
    assert verdict.kind == "loop", verdict


def test_reset_clears_state():
    detector = _make(min_loop_count=1)
    assert detector.feed("First, let us initialize variable total = 0.") is None
    # Repeating with history present must loop.
    verdict = detector.feed("First, let us initialize variable total = 0.")
    assert verdict is not None and verdict.kind == "loop"

    detector.reset()
    # After reset, history is gone, so the same sentence must not loop.
    assert detector.feed("First, let us initialize variable total = 0.") is None


def test_compression_ratio_orders_repetition():
    detector = _make()
    repetitive = ("the cat sat on the mat. " * 30).strip()
    novel = (
        "the cat sat on the mat. a dog barked loudly in the night. "
        "the sun rose over the quiet hills. many birds flew across the sky."
    )
    assert detector._compression_ratio(repetitive) < detector._compression_ratio(
        novel
    )


def test_compression_triggers_loop():
    detector = _make(
        min_loop_count=1000,  # disable the recurrence signal entirely
        compression_min_chars=10,
        compression_ratio=0.5,
    )
    verdict = None
    for _ in range(20):
        verdict = detector.feed("the cat sat on the mat. ") or verdict
    if verdict is None:
        verdict = detector.flush()
    assert verdict is not None, "expected compression-based loop"
    assert verdict.kind == "loop", verdict
    assert "entropy" in verdict.reason, verdict


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
