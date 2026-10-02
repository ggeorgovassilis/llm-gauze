"""Stalled-stream detection — a time-based watchdog for hung generations.

A model that is *crunching* (prefill, or thinking between tokens) is
legitimately silent; a *stalled* model simply never produces another token.
The two are indistinguishable from content alone — a stream emitting nothing
carries no signal to analyse. The only reliable discriminator is wall-clock
time, anchored to content-bearing tokens.

Key design points:

* The timer resets **only** on content-bearing tokens (thinking or response
  deltas) — never on SSE keepalives, comment lines, or empty ``data:`` frames,
  which prove the connection is alive but not the model.
* Two deadlines: time-to-first-token (generous, for prefill) and inter-token
  gap (tighter, once generation has begun).
* Pure and unit-testable via an injectable monotonic clock.
"""

import time

from app.config import settings
from app.remediation.base import StreamVerdict


class StallDetector:
    """Watchdog that flags a stream that has stopped producing content tokens.

    Unlike :class:`ThinkingLoopDetector`, this observes *time* rather than
    *text*, so it must be driven by the streaming loop (call :meth:`note_token`
    on each content-bearing token and poll :meth:`remaining` between reads).
    """

    def __init__(
        self,
        *,
        ttft_seconds: float = 120.0,
        gap_seconds: float = 60.0,
        clock=time.monotonic,
    ) -> None:
        self.ttft_seconds = ttft_seconds
        self.gap_seconds = gap_seconds
        self._clock = clock
        self._started = self._clock()
        self._last_token = self._started
        self.saw_first_token = False

    @classmethod
    def from_settings(cls) -> "StallDetector":
        return cls(
            ttft_seconds=settings.stall_ttft_seconds,
            gap_seconds=settings.stall_gap_seconds,
        )

    def note_token(self) -> None:
        """Record arrival of a content-bearing token (resets the gap timer)."""
        self._last_token = self._clock()
        self.saw_first_token = True

    def deadline(self) -> float:
        """Absolute monotonic time after which this stream is declared stalled."""
        if not self.saw_first_token:
            return self._started + self.ttft_seconds
        return self._last_token + self.gap_seconds

    def remaining(self) -> float:
        """Seconds left before this stream is declared stalled (may be <= 0)."""
        return self.deadline() - self._clock()

    @property
    def stalled(self) -> bool:
        return self._clock() >= self.deadline()

    def verdict(self) -> StreamVerdict:
        """Build the abort verdict for a stalled stream."""
        now = self._clock()
        anchor = self._last_token if self.saw_first_token else self._started
        elapsed = now - anchor
        if self.saw_first_token:
            reason = (
                f"no content-bearing token for {self.gap_seconds:g}s (last token {elapsed:g}s ago)"
            )
        else:
            reason = f"no first token within {self.ttft_seconds:g}s ({elapsed:g}s elapsed)"
        return StreamVerdict(
            kind="stalled",
            reason=reason,
            details={
                "ttft_seconds": self.ttft_seconds,
                "gap_seconds": self.gap_seconds,
                "saw_first_token": self.saw_first_token,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
