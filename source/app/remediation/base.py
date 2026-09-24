"""Shared abstractions for the remediation pipeline.

The project architecture separates three concerns:

1. **Data collection** — the Recorder persists every exchange/attempt.
2. **Understanding** — Detectors classify a failure into a Diagnosis.
3. **Action** — Backoff/Retry policies turn a Diagnosis into behaviour.

New remediation capabilities (sloppy-response cleanup, loop detection,
context-window detection, ...) implement these interfaces rather than touching
the proxy directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Diagnosis:
    """The understanding layer's verdict about a single failure."""

    retryable: bool
    reason: str

    def __str__(self) -> str:
        return self.reason


class Detector(ABC):
    """Understanding layer: classify a failure as retryable or not."""

    @abstractmethod
    def diagnose_exception(self, exc: Exception) -> Diagnosis:
        """Classify an exception raised while contacting the upstream."""

    @abstractmethod
    def diagnose_status(self, status: int) -> Diagnosis:
        """Classify an HTTP status returned by the upstream."""


class Backoff(ABC):
    """Action layer: compute the delay before the next attempt."""

    @abstractmethod
    def delay(self, attempt: int) -> float:
        """Return the delay in seconds to wait before retrying `attempt`.

        `attempt` is 1-based; the first retry is attempt 1.
        """


@dataclass
class RetryPolicy:
    """Binds understanding (detector) and action (backoff) into one policy."""

    max_attempts: int
    detector: Detector
    backoff: Backoff

    def should_retry(self, diagnosis: Diagnosis, attempt: int) -> bool:
        return diagnosis.retryable and attempt < self.max_attempts
