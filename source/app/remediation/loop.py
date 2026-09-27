"""Output loop detection — the content-stream understanding layer.

Reasoning models sometimes get stuck regenerating the same (or near-identical)
text instead of progressing. This detector observes a stream of text (thinking
or response) and raises a ``StreamVerdict`` the moment a loop is detected, so
the gateway can abort the request early.

Detection is **byte-level**, not sentence-level:

* the stream is accumulated as raw text and repetition is measured over a
  sliding **byte window** via zlib compression ratio (low entropy = loop);
* there are no sentence boundaries, n-gram sets or Jaccard similarity — the
  previous n-gram-recurrence signal was removed because it could not tell a
  genuine loop from a model legitimately enumerating near-identical tool
  results (e.g. twelve vetters each replying ``done — nothing left to vet``);

* detection is **gated**: it only arms once the stream has produced a fraction
  of the output window. Short, structured enumerations finish and move on
  before the detector arms, so they can never trip it; a genuine loop keeps
  emitting repetitive output and is caught as it approaches the window limit.

Only the standard library is used, so the detector is pure and unit-testable
without the gateway.
"""

import zlib

from app.config import settings
from app.remediation.base import StreamDetector, StreamVerdict


class ThinkingLoopDetector(StreamDetector):
    """Detect loops in a text stream via byte-window compression entropy."""

    def __init__(
        self,
        *,
        window_bytes: int = 32768,
        min_output_fraction: float = 0.25,
        compression_ratio: float = 0.22,
    ) -> None:
        self.window_bytes = window_bytes
        self.min_output_fraction = min_output_fraction
        self.compression_ratio = compression_ratio
        # Output that must be produced before detection arms.
        self.min_output_bytes = int(window_bytes * min_output_fraction)
        self._buffer = ""

    @classmethod
    def from_settings(cls) -> "ThinkingLoopDetector":
        """Build a detector from the gateway's environment settings."""
        return cls(
            window_bytes=settings.loop_window_bytes,
            min_output_fraction=settings.loop_min_output_fraction,
            compression_ratio=settings.loop_compression_ratio,
        )

    def feed(self, text: str) -> StreamVerdict | None:
        """Feed arbitrary text; returns a verdict iff a loop is detected.

        ``None`` means "no loop (yet)". Until ``min_output_bytes`` have been
        produced the detector is not armed and always returns ``None``.
        """
        if not text:
            return None
        self._buffer += text
        if len(self._buffer) < self.min_output_bytes:
            return None
        return self._check_window()

    def flush(self) -> StreamVerdict | None:
        """Run a final check over the trailing buffer, if any."""
        verdict = None
        if self._buffer:
            verdict = self._check_window()
        self._buffer = ""
        return verdict

    def reset(self) -> None:
        """Clear all state so the detector can be reused for a new stream."""
        self._buffer = ""

    def _check_window(self) -> StreamVerdict | None:
        """Compression-ratio check over the most recent ``window_bytes``."""
        window = self._buffer[-self.window_bytes :]
        # Skip a trivial window (short partial tail) — compression on a tiny
        # buffer is dominated by zlib framing overhead, not repetition.
        if len(window) < self.min_output_bytes:
            return None
        ratio = self._compression_ratio(window)
        if ratio < self.compression_ratio:
            return StreamVerdict(
                kind="loop",
                reason=f"low entropy (compression ratio {ratio:.2f})",
                details={"compression_ratio": ratio, "window_bytes": len(window)},
            )
        return None

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """Ratio of zlib-compressed to raw size; lower = more repetitive."""
        if not text:
            return 1.0
        raw = text.encode("utf-8")
        return len(zlib.compress(raw)) / len(raw)
