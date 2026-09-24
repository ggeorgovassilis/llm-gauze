"""Remediation pipeline.

Separates three concerns so new capabilities can be added without touching the
proxy:

1. **Data collection** — the `Recorder` persists every exchange/attempt.
2. **Understanding** — `Detector`s classify a failure into a `Diagnosis`.
3. **Action** — `Backoff`/`RetryPolicy` turn a `Diagnosis` into behaviour.

Later phases (sloppy-response cleanup, loop detection, context-window
detection, ...) implement these interfaces rather than editing the proxy.
"""

from app.remediation.base import Backoff, Detector, Diagnosis, RetryPolicy
from app.remediation.retry import ExponentialBackoff, RetryableDetector

__all__ = [
    "Backoff",
    "Detector",
    "Diagnosis",
    "RetryPolicy",
    "ExponentialBackoff",
    "RetryableDetector",
]
