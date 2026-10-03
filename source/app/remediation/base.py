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
from enum import Enum


class VerdictKind(str, Enum):
    """Machine-readable kind of a content-stream verdict.

    Detectors *construct* verdicts with these constants and the proxy *routes*
    them through the registry in :mod:`app.remediation.verdicts`. Using an enum
    means a misspelled kind fails at import/attribute access instead of
    silently falling through to a default branch at runtime.
    """

    LOOP = "loop"
    STALLED = "stalled"
    RUNAWAY_REASONING = "runaway_reasoning"


class DiagnosisCode(str, Enum):
    """Machine-readable diagnosis codes the proxy routes on.

    The context-window overflow is the one diagnosis that short-circuits the
    retry loop; like :class:`VerdictKind` it is typed so a misspelled code
    fails at import rather than runtime.
    """

    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"


@dataclass
class Diagnosis:
    """The understanding layer's verdict about a single failure."""

    retryable: bool
    reason: str
    # Optional machine-readable code (e.g. ``DiagnosisCode.CONTEXT_WINDOW_EXCEEDED``)
    # that lets the proxy route a verdict to a bespoke response instead of the
    # generic pass-through/502 path. ``None`` for plain retry/status verdicts.
    code: DiagnosisCode | None = None

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

    ``kind`` is the :class:`VerdictKind` that was detected; ``reason`` is a
    human-readable explanation and ``details`` carries machine-readable extras.
    """

    kind: VerdictKind
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


@dataclass
class Turn:
    """The assembled result of one streamed turn, as the ladder sees it.

    A single, uniform view of a completed turn that a :class:`Remediation`
    step inspects: the finish reason, the visible content, the reasoning
    stream, any tool calls, and (for abort-verdict steps) the content verdict
    that aborted the stream.
    """

    finish_reason: str | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    verdict: StreamVerdict | None = None


class Remediation(ABC):
    """Action layer: one rung of the composable remediation ladder.

    A remediation step inspects a completed :class:`Turn` (plus the request it
    came from) and, when its trigger fires, produces the re-submission request
    body that gives the model another chance. Steps carry their own attempt
    budget so the ladder drives every rung uniformly — a new remediation is
    "implement and register", not "edit the loop".

    Implementers set :attr:`name` (a short id used for recording/telemetry)
    and :attr:`max_attempts`, and implement :meth:`applies` (the trigger) and
    :meth:`apply` (the re-submission body).
    """

    #: Short id used to key this step's recorder entries and attempt counter.
    name: str = "remediation"
    #: How many re-submissions this step may make before the ladder gives up.
    max_attempts: int = 1

    @abstractmethod
    def applies(self, turn: Turn, request_body: dict) -> bool:
        """Whether this step triggers for the given turn and request."""

    @abstractmethod
    def apply(self, turn: Turn, request_body: dict) -> dict:
        """Return a copy of ``request_body`` prepared for re-submission."""


class Transform(ABC):
    """A pure content/body transform that reports its mutations.

    Transforms rewrite a value the pipeline hands them — an assembled
    completion (think cleanup, tool-call repair) or a request body
    (message-overflow trimming) — and report a list of ``{kind, …}`` change
    records so no mutation is silent. Unlike :class:`Remediation`, a transform
    never re-submits: it edits in place and the pipeline moves on.

    Subclasses expose a canonical :meth:`apply` entry point (whose exact
    signature depends on the value transformed) and may keep convenience
    methods for finer steps (e.g. ``relocate`` / ``guard_empty``). This base
    exists so every transform shares one type and one documented contract.
    """
