"""In-process metrics registry and exposition.

Bandaid's telemetry surface: a small set of counters, a latency histogram, and
an upstream-down gauge, held in memory and exposed over HTTP at ``/metrics``.

Deliberately dependency-free — a ``threading.Lock`` plus plain dicts, no
Prometheus client library, no third-party runtime deps. Two renderers turn the
same snapshot into either Prometheus text or minimal JSON (chosen by the
request's ``Accept`` header).

Counters are incremented by the proxy at the exact points it already decides an
outcome (retry, abort, success, upstream failure), so they are a live roll-up of
what ``records.jsonl`` persists — no new data is invented here.
"""

import threading
import time
from collections import Counter

# Latency histogram bucket upper bounds (seconds). Spans the realistic range of
# local-LLM generation times (a slow 12B model can legitimately take minutes).
_LATENCY_BUCKETS = (1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)

# Metric metadata for the Prometheus exposition: name -> (type, help).
_METRICS = {
    "requests_total": ("counter", "Requests handled, by final outcome."),
    "attempts_total": ("counter", "Upstream attempts made (one per retry)."),
    "retries_total": ("counter", "Retries scheduled."),
    "upstream_errors_total": ("counter", "Transport errors, by exception type."),
    "loop_aborts_total": ("counter", "Loops aborted, by stream."),
    "stall_aborts_total": ("counter", "Stalls aborted."),
    "upstream_latency_seconds": ("histogram", "Upstream response latency."),
    "upstream_down": ("gauge", "Whether the upstream is currently failing."),
}


def _label_key(labels):
    return tuple(sorted(labels.items())) if labels else ()


def _fmt_labels(labels):
    return ",".join(f'{k}="{v}"' for k, v in labels.items())


class Telemetry:
    """Thread-safe in-memory metrics registry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[tuple, int]] = {}
        self._latency_buckets: Counter = Counter()
        self._latency_sum = 0.0
        self._latency_count = 0
        self._upstream_down = False
        self._upstream_down_since: float | None = None
        self._started = time.time()

    # -- recording -----------------------------------------------------

    def incr(self, name: str, labels: dict | None = None, amount: int = 1) -> None:
        """Increment counter `name`, optionally scoped by `labels`."""
        key = _label_key(labels)
        with self._lock:
            bucket = self._counters.setdefault(name, {})
            bucket[key] = bucket.get(key, 0) + amount

    def observe_latency(self, seconds: float) -> None:
        """Record an upstream latency sample into the histogram."""
        with self._lock:
            self._latency_sum += seconds
            self._latency_count += 1
            for upper in _LATENCY_BUCKETS:
                if seconds <= upper:
                    self._latency_buckets[upper] += 1

    def set_upstream_down(self, down: bool) -> None:
        """Set the upstream-down gauge; tracks when the window began."""
        with self._lock:
            if down and not self._upstream_down:
                self._upstream_down_since = time.time()
            elif not down:
                self._upstream_down_since = None
            self._upstream_down = down

    # -- snapshot ------------------------------------------------------

    def _counters_snapshot(self) -> dict:
        out: dict = {}
        for name, buckets in sorted(self._counters.items()):
            if len(buckets) == 1 and () in buckets:
                out[name] = buckets[()]
            else:
                out[name] = {
                    _fmt_labels(dict(labels)): value
                    for labels, value in sorted(buckets.items())
                }
        return out

    def snapshot(self) -> dict:
        """Structured dict for the JSON exposition."""
        with self._lock:
            mean = (
                self._latency_sum / self._latency_count
                if self._latency_count
                else 0.0
            )
            return {
                "uptime_seconds": round(time.time() - self._started, 3),
                "counters": self._counters_snapshot(),
                "latency": {
                    "count": self._latency_count,
                    "sum_seconds": round(self._latency_sum, 3),
                    "mean_seconds": round(mean, 3),
                },
                "upstream_down": self._upstream_down,
                "upstream_down_since": self._upstream_down_since,
            }

    # -- exposition ----------------------------------------------------

    def render_prometheus(self) -> str:
        """Render the snapshot as Prometheus text exposition format."""
        lines: list[str] = []
        for name in sorted(_METRICS):
            mtype, help_text = _METRICS[name]
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
            with self._lock:
                if name == "upstream_latency_seconds":
                    buckets = dict(self._latency_buckets)
                    total = self._latency_count
                    for upper in _LATENCY_BUCKETS:
                        lines.append(
                            f'{name}_bucket{{le="{upper:g}"}} {buckets.get(upper, 0)}'
                        )
                    lines.append(f'{name}_bucket{{le="+Inf"}} {total}')
                    lines.append(f"{name}_sum {self._latency_sum}")
                    lines.append(f"{name}_count {total}")
                elif name == "upstream_down":
                    lines.append(
                        f"upstream_down {1 if self._upstream_down else 0}"
                    )
                else:
                    buckets = self._counters.get(name, {})
                    if not buckets:
                        lines.append(f"{name} 0")
                    else:
                        for labels, value in sorted(buckets.items()):
                            if labels:
                                lines.append(
                                    f'{name}{{{_fmt_labels(dict(labels))}}} {value}'
                                )
                            else:
                                lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


# Process-wide singleton, mirroring `settings` in app.config.
telemetry = Telemetry()
