"""Unit tests for the telemetry registry and its renderers.

Runs with plain Python (stdlib only) inside the container:

    docker compose exec -T gateway python - < tests/test_telemetry.py
"""

import json
import traceback

try:
    from app.telemetry import Telemetry
except ImportError:  # pragma: no cover - run on host without the container
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from app.telemetry import Telemetry


def test_counter_increment_and_labels():
    t = Telemetry()
    t.incr("requests_total", {"outcome": "success"})
    t.incr("requests_total", {"outcome": "success"})
    t.incr("requests_total", {"outcome": "loop_aborted"})
    snap = t.snapshot()["counters"]
    assert snap["requests_total"]["outcome=\"success\""] == 2
    assert snap["requests_total"]["outcome=\"loop_aborted\""] == 1


def test_unlabelled_counter_is_scalar():
    t = Telemetry()
    t.incr("retries_total")
    t.incr("retries_total")
    assert t.snapshot()["counters"]["retries_total"] == 2


def test_latency_histogram_and_mean():
    t = Telemetry()
    t.observe_latency(0.5)
    t.observe_latency(3.0)
    t.observe_latency(60.0)
    latency = t.snapshot()["latency"]
    assert latency["count"] == 3
    assert abs(latency["sum_seconds"] - 63.5) < 1e-6
    assert latency["mean_seconds"] == round(63.5 / 3, 3)


def test_upstream_down_gauge_transitions():
    t = Telemetry()
    assert t.snapshot()["upstream_down"] is False
    assert t.snapshot()["upstream_down_since"] is None

    t.set_upstream_down(True)
    snap = t.snapshot()
    assert snap["upstream_down"] is True
    assert snap["upstream_down_since"] is not None

    # Setting down again must not reset the window start.
    first_since = snap["upstream_down_since"]
    t.set_upstream_down(True)
    assert t.snapshot()["upstream_down_since"] == first_since

    t.set_upstream_down(False)
    snap = t.snapshot()
    assert snap["upstream_down"] is False
    assert snap["upstream_down_since"] is None


def test_prometheus_rendering():
    t = Telemetry()
    t.incr("requests_total", {"outcome": "success"})
    t.incr("loop_aborts_total", {"stream": "thinking"})
    t.observe_latency(2.5)
    t.set_upstream_down(True)

    text = t.render_prometheus()
    assert "# HELP requests_total" in text
    assert "# TYPE requests_total counter" in text
    assert 'requests_total{outcome="success"} 1' in text
    assert 'loop_aborts_total{stream="thinking"} 1' in text
    assert 'upstream_latency_seconds_bucket{le="5"} 1' in text
    assert 'upstream_latency_seconds_bucket{le="+Inf"} 1' in text
    assert "upstream_down 1" in text


def test_json_snapshot_is_serialisable():
    t = Telemetry()
    t.incr("requests_total", {"outcome": "success"})
    t.observe_latency(1.0)
    # Must not raise on JSON serialisation (floats, ints, None all fine).
    json.dumps(t.snapshot())


def _run_all():
    import inspect

    funcs = [
        (name, obj)
        for name, obj in inspect.getmembers(
            inspect.getmodule(inspect.currentframe()),
            inspect.isfunction,
        )
        if name.startswith("test_")
    ]
    failures = []
    for name, func in funcs:
        try:
            func()
            print(f"PASS {name}")
        except AssertionError:
            failures.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()
    if failures:
        print(f"\n{len(failures)} failed: {failures}")
        raise SystemExit(1)
    print(f"\n{len(funcs)} tests passed")


if __name__ == "__main__":
    _run_all()
