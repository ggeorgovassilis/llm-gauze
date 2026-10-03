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
from dataclasses import dataclass, field


@dataclass
class Diagnosis:
    """The understanding layer's verdict about a single failure."""

    retryable: bool
    reason: str
    # Optional machine-readable code (e.g. ``"context_window_exceeded"``) that
    # lets the proxy route a verdict to a bespoke response instead of the
    # generic pass-through/502 path. ``None`` for plain retry/status verdicts.
    code: str | None = None

    def __str__(self) -> str:
        return self.reason


class Detector(ABC):
    """Understanding layer: classify a failure as retryable or not."""

    @abstractmethod
    def diagnose_exception(self, exc: Exception) -> Diagnosis:
        """Classify an exception raised while contacting the upstream."""

    @abstractmethod
    def diagnose_status(self, status: int, body: bytes | None = None) -> Diagnosis:
        """Classify an HTTP status returned by the upstream.

        ``body`` carries the upstream response payload when available, so
        body-aware detectors (e.g. context-window overflow) can inspect it.
        """


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


@dataclass
class StreamVerdict:
    """The understanding layer's verdict about a content stream.

    ``kind`` describes what was detected (e.g. ``"loop"``); ``reason`` is a
    human-readable explanation and ``details`` carries machine-readable extras.
    """

    kind: str
    reason: str | None = None
    details: dict = field(default_factory=dict)


class ContentWatchdog(ABC):
    """Understanding layer for a content stream — the uniform watchdog contract.

    A *content watchdog* observes the generated text of a single streamed turn
    (reasoning, visible content, and tool-call fragments) and raises a
    :class:`StreamVerdict` the moment it detects a problem — a loop, a silent
    stall, or endless reasoning. Sibling of ``Detector``: where ``Detector``
    classifies transport/status failures, a ``ContentWatchdog`` observes the
    generated text itself.

    All three watchdogs (:class:`ThinkingLoopDetector`, :class:`StallDetector`,
    :class:`RunawayReasoningDetector`) implement this one interface, so the
    streaming pipeline drives them uniformly and a new watchdog is
    "implement and register", not "edit the loop".

    Stateful by design — one instance per (request, stream), never shared
    across requests.
    """

    @abstractmethod
    def note(
        self,
        *,
        reasoning: str | None = None,
        content: str | None = None,
        tool_calls: list | None = None,
    ) -> None:
        """Feed one content-bearing event (any combination of the three).

        Called by the pipeline once per SSE delta that carries model output. A
        text-based watchdog accumulates the text here; a time-based watchdog
        resets its timer on any content-bearing event; a token-count watchdog
        tallies its budget and notes whether visible content/tool-calls have
        appeared.
        """

    @abstractmethod
    def check(self) -> StreamVerdict | None:
        """Return a verdict iff the watchdog has tripped, else ``None``.

        Called by the pipeline after each ``note`` and again at end of stream.
        """

    @abstractmethod
    def remaining(self) -> float | None:
        """Seconds until this watchdog trips if no token arrives, else ``None``.

        A time-based watchdog (stall) returns its remaining budget so the
        pipeline can bound each read; text/count-based watchdogs return
        ``None`` (they trip inside ``note``/``check``, not on a timer).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear all accumulated state so the watchdog can be reused."""
