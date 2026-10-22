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
* A **minimum-speed floor** augments the gap timer: once the first token has
  arrived the stream must sustain a configurable tokens-per-second rate over a
  sliding window, so a slow *trickle* (tokens arriving just under the gap
  timer) is caught too. Tokens are counted from content-bearing text via the
  ~4 chars/token heuristic.
* Pure and unit-testable via an injectable monotonic clock.
"""

import math
import time
from collections import deque

from app.config import settings
from app.remediation.base import ContentWatchdog, StreamVerdict, VerdictKind


def _token_chars(
    reasoning: str | None,
    content: str | None,
    tool_calls: list | None,
) -> float:
    """Estimate content-bearing characters for one note event.

    Tool-call fragments count as content (a model slowly streaming tool
    arguments is working, not stalled); only their name/arguments strings are
    measured, mirroring the text streams.
    """
    chars = len(reasoning or "") + len(content or "")
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function")
        if isinstance(fn, dict):
            chars += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
    return float(chars)


class StallDetector(ContentWatchdog):
    """Watchdog that flags a stream that has stopped producing content tokens.

    Unlike :class:`ThinkingLoopDetector`, this observes *time* rather than
    *text*. It implements the uniform :class:`ContentWatchdog` contract: the
    pipeline calls :meth:`note` on each content-bearing token and polls
    :meth:`remaining` between reads to bound each read by the stall budget.
    """

    # Characters per token approximation (matches the runaway detector's rule
    # of thumb of ~4 bytes/token).
    CHARS_PER_TOKEN = 4.0

    def __init__(
        self,
        *,
        ttft_seconds: float = 120.0,
        gap_seconds: float = 60.0,
        min_speed_tps: float = 1.0,
        speed_window_seconds: float = 60.0,
        speed_floor_enabled: bool = True,
        clock=time.monotonic,
    ) -> None:
        self.ttft_seconds = ttft_seconds
        self.gap_seconds = gap_seconds
        self.min_speed_tps = min_speed_tps
        self.speed_window_seconds = speed_window_seconds
        self.speed_floor_enabled = speed_floor_enabled
        self._clock = clock
        self._started = self._clock()
        self._last_token = self._started
        self.saw_first_token = False
        self._first_token: float | None = None
        # (monotonic time, chars) per content-bearing event, ascending, pruned
        # to the trailing speed window.
        self._token_events: deque[tuple[float, float]] = deque()

    @classmethod
    def from_settings(cls) -> "StallDetector":
        return cls(
            ttft_seconds=settings.stall_ttft_seconds,
            gap_seconds=settings.stall_gap_seconds,
            min_speed_tps=settings.stall_min_speed_tps,
            speed_window_seconds=settings.stall_speed_window_seconds,
            speed_floor_enabled=settings.stall_speed_floor_enabled,
        )

    def note(
        self,
        *,
        reasoning: str | None = None,
        content: str | None = None,
        tool_calls: list | None = None,
    ) -> None:
        """Record arrival of a content-bearing event (resets the gap timer)."""
        if not (reasoning or content or tool_calls):
            return
        now = self._clock()
        self._last_token = now
        if not self.saw_first_token:
            self._first_token = now
            self.saw_first_token = True
        self._token_events.append((now, _token_chars(reasoning, content, tool_calls)))
        self._prune(now)

    def _prune(self, now: float) -> None:
        """Drop token events that have aged out of the trailing speed window."""
        cutoff = now - self.speed_window_seconds
        while self._token_events and self._token_events[0][0] <= cutoff:
            self._token_events.popleft()

    def _active_events(self, now: float) -> list[tuple[float, float]]:
        """Token events still inside the trailing speed window at ``now``."""
        return [(t, c) for t, c in self._token_events if t > now - self.speed_window_seconds]

    def _window_tokens(self, now: float) -> float:
        """Estimated tokens inside the trailing speed window at ``now``."""
        chars = sum(c for _, c in self._active_events(now))
        return chars / self.CHARS_PER_TOKEN

    def _speed_floor_deadline(self) -> float:
        """Absolute monotonic time the speed floor trips (no new tokens).

        The stream must sustain ``min_speed_tps`` over ``speed_window_seconds``
        once the first token has arrived. Assuming no new tokens, the trailing
        count only ever decreases as events age out, so the trip time is the
        first age-out boundary (or the window-fill moment) where the count
        falls below the floor.
        """
        target = self.min_speed_tps * self.speed_window_seconds * self.CHARS_PER_TOKEN
        if self._first_token is None:
            return math.inf
        events = self._active_events(self._clock())
        total = sum(c for _, c in events)
        if total < target:
            return self._first_token + self.speed_window_seconds
        removed = 0.0
        for t, c in events:
            removed += c
            if total - removed < target:
                return t + self.speed_window_seconds
        return math.inf

    def check(self) -> StreamVerdict | None:
        """Return the stall verdict iff the deadline has passed."""
        if self._clock() >= self.deadline():
            return self.verdict()
        return None

    def deadline(self) -> float:
        """Absolute monotonic time after which this stream is declared stalled."""
        if not self.saw_first_token:
            return self._started + self.ttft_seconds
        gap_deadline = self._last_token + self.gap_seconds
        if not self.speed_floor_enabled:
            return gap_deadline
        return min(gap_deadline, self._speed_floor_deadline())

    def remaining(self) -> float:
        """Seconds left before this stream is declared stalled (may be <= 0)."""
        return self.deadline() - self._clock()

    def reset(self) -> None:
        """Clear state so the watchdog can be reused for a new stream."""
        self._started = self._clock()
        self._last_token = self._started
        self.saw_first_token = False
        self._first_token = None
        self._token_events.clear()

    def verdict(self) -> StreamVerdict:
        """Build the abort verdict for a stalled stream."""
        now = self._clock()
        if not self.saw_first_token:
            elapsed = now - self._started
            reason = f"no first token within {self.ttft_seconds:g}s ({elapsed:g}s elapsed)"
            return StreamVerdict(
                kind=VerdictKind.STALLED,
                reason=reason,
                details={
                    "ttft_seconds": self.ttft_seconds,
                    "gap_seconds": self.gap_seconds,
                    "saw_first_token": False,
                    "elapsed_seconds": round(elapsed, 3),
                },
            )
        if self.speed_floor_enabled and self._speed_floor_deadline() <= (
            self._last_token + self.gap_seconds
        ):
            tokens = self._window_tokens(now)
            reason = (
                f"minimum speed floor {self.min_speed_tps:g} t/s not sustained "
                f"({tokens:g} tokens over {self.speed_window_seconds:g}s)"
            )
            return StreamVerdict(
                kind=VerdictKind.STALLED,
                reason=reason,
                details={
                    "ttft_seconds": self.ttft_seconds,
                    "gap_seconds": self.gap_seconds,
                    "min_speed_tps": self.min_speed_tps,
                    "speed_window_seconds": self.speed_window_seconds,
                    "saw_first_token": True,
                    "window_tokens": round(tokens, 3),
                    "elapsed_seconds": round(now - (self._first_token or now), 3),
                },
            )
        elapsed = now - self._last_token
        reason = f"no content-bearing token for {self.gap_seconds:g}s (last token {elapsed:g}s ago)"
        return StreamVerdict(
            kind=VerdictKind.STALLED,
            reason=reason,
            details={
                "ttft_seconds": self.ttft_seconds,
                "gap_seconds": self.gap_seconds,
                "saw_first_token": True,
                "elapsed_seconds": round(elapsed, 3),
            },
        )
