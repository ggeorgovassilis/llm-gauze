"""Thinking/output loop detection — the content-stream understanding layer.

Reasoning models sometimes get stuck regenerating the same (or near-identical)
sentences instead of progressing. This detector observes a stream of text
(thinking or response) and raises a ``StreamVerdict`` the moment a loop is
detected, so the gateway can abort the request early.

It fixes the issues identified in the original draft:

* stores the sliding window as **text** (not n-gram sets) so the compression
  signal can be computed over the real window content;
* is **stateful per stream** with an explicit ``reset()`` — one instance per
  (request, stream), never shared across requests;
* treats n-gram recurrence and low compression as **two independent** signals;
* returns a structured ``StreamVerdict`` instead of raw dicts;
* accumulates partial chunks and splits them into sentences internally;
* ignores fragments shorter than the n-gram size, so trivial repeats (list
  markers, region codes, lone digits) can't trip the recurrence signal on
  structured output such as lists, tables or prices.

Only the standard library is used, so the detector is pure and unit-testable
without the gateway.
"""

import re
import zlib

from app.config import settings
from app.remediation.base import StreamDetector, StreamVerdict

# A sentence boundary is any full stop, question mark, exclamation mark, or
# newline. This is deliberately simple for v1 (see ticket open question on
# richer boundary detection).
_SENTENCE_END = re.compile(r"[.!?\n]")
_WORD = re.compile(r"\w+")


class ThinkingLoopDetector(StreamDetector):
    """Detect loops in a single text stream via n-gram recurrence + entropy."""

    def __init__(
        self,
        *,
        window_sentences: int = 20,
        jaccard_threshold: float = 0.65,
        min_loop_count: int = 3,
        ngram_size: int = 3,
        compression_ratio: float = 0.22,
        compression_min_chars: int = 300,
    ) -> None:
        self.window_sentences = window_sentences
        self.jaccard_threshold = jaccard_threshold
        self.min_loop_count = min_loop_count
        self.ngram_size = ngram_size
        self.compression_ratio = compression_ratio
        self.compression_min_chars = compression_min_chars
        self._pending = ""
        self._window: list[str] = []

    @classmethod
    def from_settings(cls) -> "ThinkingLoopDetector":
        """Build a detector from the gateway's environment settings."""
        return cls(
            window_sentences=settings.loop_window_sentences,
            jaccard_threshold=settings.loop_jaccard_threshold,
            min_loop_count=settings.loop_min_loop_count,
            ngram_size=settings.loop_ngram_size,
            compression_ratio=settings.loop_compression_ratio,
            compression_min_chars=settings.loop_compression_min_chars,
        )

    def feed(self, text: str) -> StreamVerdict | None:
        """Feed arbitrary text; returns a verdict iff a loop is detected.

        Incoming text may be a partial token/chunk — it is accumulated and
        split into sentences internally. ``None`` means "no loop (yet)".
        """
        self._pending += text
        while True:
            sentence, remainder = self._split_sentence(self._pending)
            if sentence is None:
                self._pending = remainder
                return None
            self._pending = remainder
            verdict = self._observe_sentence(sentence)
            if verdict is not None:
                return verdict

    def flush(self) -> StreamVerdict | None:
        """Process any trailing partial text as a final sentence."""
        verdict = None
        if self._pending.strip():
            verdict = self._observe_sentence(self._pending)
        self._pending = ""
        return verdict

    def reset(self) -> None:
        """Clear all state so the detector can be reused for a new stream."""
        self._pending = ""
        self._window.clear()

    @staticmethod
    def _split_sentence(text: str) -> tuple[str | None, str]:
        match = _SENTENCE_END.search(text)
        if match is None:
            return None, text
        index = match.end()
        return text[:index], text[index:]

    def _observe_sentence(self, sentence: str) -> StreamVerdict | None:
        clean = sentence.strip()
        if not clean:
            return None

        # Fragments shorter than the n-gram size cannot form a real n-gram, so
        # they carry no repetition signal. Skipping them stops trivial repeats
        # — list markers ("1."), region codes ("eu."), lone digits — from
        # collapsing to an identical one-element set and over-matching, which
        # previously tripped the loop on legitimate structured output (lists,
        # tables, prices). A genuine loop repeats substantial sentences, never
        # single tokens.
        if len(_WORD.findall(clean.lower())) < self.ngram_size:
            return None

        ngrams = self._ngrams(clean)

        # Signal 1: high recurrence against the recent window.
        matches = 0
        for past in self._window:
            if self._jaccard(ngrams, self._ngrams(past)) >= self.jaccard_threshold:
                matches += 1

        self._window.append(clean)
        if len(self._window) > self.window_sentences:
            self._window.pop(0)

        if matches >= self.min_loop_count:
            return StreamVerdict(
                kind="loop",
                reason=(
                    f"high n-gram recurrence "
                    f"({matches} similar sentences in window)"
                ),
                details={"matches": matches, "window_size": len(self._window)},
            )

        # Signal 2: low entropy over the whole window (independent of signal 1).
        full_window = " ".join(self._window)
        if len(full_window) >= self.compression_min_chars:
            ratio = self._compression_ratio(full_window)
            if ratio < self.compression_ratio:
                return StreamVerdict(
                    kind="loop",
                    reason=f"low entropy (compression ratio {ratio:.2f})",
                    details={"compression_ratio": ratio},
                )

        return None

    def _ngrams(self, text: str) -> frozenset:
        """Word n-grams of ``text``.

        Callers must only pass text with at least ``ngram_size`` words (see
        ``_observe_sentence``); shorter input yields an empty set so it can
        never match anything.
        """
        words = _WORD.findall(text.lower())
        n = self.ngram_size
        if len(words) < n:
            return frozenset()
        return frozenset(tuple(words[i : i + n]) for i in range(len(words) - n + 1))

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _compression_ratio(text: str) -> float:
        """Ratio of zlib-compressed to raw size; lower = more repetitive."""
        if not text:
            return 1.0
        raw = text.encode("utf-8")
        return len(zlib.compress(raw)) / len(raw)
