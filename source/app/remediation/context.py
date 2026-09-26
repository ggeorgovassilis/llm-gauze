"""Context-window overflow detection.

The upstream (llama.cpp behind LiteLLM) rejects a request that no longer fits
its context window with a deterministic error *message* — carried in the HTTP
response body or, occasionally, an exception message — rather than a distinct
status code. Retrying cannot succeed: the same oversized input fails identically
every time. Recognising the message lets the gateway fail fast with an explicit
``context_window_exceeded`` response instead of burning ``retry_max_attempts``
attempts and then returning a misleading generic 502.
"""

from app.config import settings
from app.remediation.base import Detector, Diagnosis

# Default substrings (matched case-insensitively) that llama.cpp emits when the
# context window fills. Kept exact — never word-fuzzy — so there are no false
# positives. Overridable via ``CONTEXT_WINDOW_MARKERS`` for other versions.
_DEFAULT_MARKERS = (
    "exceeded the context window",
    "context window exceeded",
    "context length exceeded",
    "maximum context length",
    "prompt is too long",
    "prompt is longer than the context",
    "input is too long",
    "ran out of context",
    # LiteLLM's own phrasing when it wraps the underlying provider's error.
    "exceeds the available context size",
)

CONTEXT_WINDOW_CODE = "context_window_exceeded"


def _decode(body: bytes | None) -> str:
    if not body:
        return ""
    return body.decode("utf-8", errors="replace")


class ContextWindowDetector(Detector):
    """Recognises a context-window-fill error and defers everything else.

    Wraps a fallback :class:`Detector` (the transient-failure classifier) so it
    can slot into the existing ``RetryPolicy`` unchanged: it only *overrides*
    the verdict when the context-window signature is present.
    """

    def __init__(
        self,
        fallback: Detector,
        markers: tuple[str, ...] | None = None,
    ) -> None:
        self.fallback = fallback
        self.markers = tuple(
            m.lower() for m in (markers or _DEFAULT_MARKERS)
        )

    @classmethod
    def from_settings(cls, fallback: Detector) -> "ContextWindowDetector":
        markers = tuple(
            s.strip()
            for s in settings.context_window_markers.split(",")
            if s.strip()
        )
        return cls(fallback, markers or _DEFAULT_MARKERS)

    def matches(self, text: str) -> bool:
        haystack = text.lower()
        return any(marker in haystack for marker in self.markers)

    def _overflow(self) -> Diagnosis:
        return Diagnosis(
            retryable=False,
            reason="context window exceeded",
            code=CONTEXT_WINDOW_CODE,
        )

    def diagnose_exception(self, exc: Exception) -> Diagnosis:
        # The overflow normally arrives as an HTTP body; a few stacks surface
        # it as an exception message instead. Catch either.
        if self.matches(str(exc)):
            return self._overflow()
        return self.fallback.diagnose_exception(exc)

    def diagnose_status(
        self, status: int, body: bytes | None = None
    ) -> Diagnosis:
        if self.matches(_decode(body)):
            return self._overflow()
        return self.fallback.diagnose_status(status, body)
