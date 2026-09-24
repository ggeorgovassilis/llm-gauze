"""Concrete retry capability: transient-failure detection + exponential backoff.

Understanding layer: `RetryableDetector` classifies failures.
Action layer: `ExponentialBackoff` computes delays.
"""

import logging
import random

import httpx

from app.remediation.base import Backoff, Detector, Diagnosis

logger = logging.getLogger("bandaid.remediation.retry")

# Transport-level failures mean "no valid response was received", which is
# almost always safe to retry against a flaky local LLM. We therefore classify
# by *category* rather than enumerating every possible exception type:
#
#   * `httpx.RequestError` — base class for every network/timeout error httpx
#     raises (ConnectError, ReadError/ReadTimeout, WriteError/WriteTimeout,
#     RemoteProtocolError, PoolTimeout, ...). Covers connection refused,
#     connection reset mid-response, and timeouts alike.
#   * `OSError` — raw socket errors that can leak through unwrapped
#     (ConnectionResetError, BrokenPipeError, ConnectionRefusedError, ...).
#   * `TimeoutError` — asyncio-level timeouts not wrapped by httpx.
#
# Exceptions that look like bugs in *our* code (TypeError, ValueError, ...) are
# deliberately left out so they surface instead of being masked by retries.
_RETRYABLE_EXCEPTIONS = (httpx.RequestError, OSError, TimeoutError)


class RetryableDetector(Detector):
    """Classifies transient upstream failures as retryable."""

    def __init__(self, retryable_statuses: set[int]) -> None:
        self.retryable_statuses = set(retryable_statuses)

    def diagnose_exception(self, exc: Exception) -> Diagnosis:
        if isinstance(exc, _RETRYABLE_EXCEPTIONS):
            return Diagnosis(
                retryable=True,
                reason=f"transient upstream error {type(exc).__name__}: {exc}",
            )
        return Diagnosis(
            retryable=False,
            reason=f"non-retryable exception {type(exc).__name__}: {exc}",
        )

    def diagnose_status(self, status: int) -> Diagnosis:
        if status in self.retryable_statuses:
            return Diagnosis(
                retryable=True, reason=f"retryable HTTP status {status}"
            )
        return Diagnosis(
            retryable=False, reason=f"non-retryable HTTP status {status}"
        )


class ExponentialBackoff(Backoff):
    """Exponential backoff with optional full jitter and an upper cap."""

    def __init__(
        self,
        initial: float,
        base: float,
        maximum: float,
        jitter: bool,
    ) -> None:
        self.initial = initial
        self.base = base
        self.maximum = maximum
        self.jitter = jitter

    def delay(self, attempt: int) -> float:
        # attempt is 1-based; the first retry waits `initial`.
        delay = self.initial * (self.base ** (attempt - 1))
        delay = min(delay, self.maximum)
        if self.jitter:
            delay = random.uniform(0, delay)  # full jitter
        return delay
