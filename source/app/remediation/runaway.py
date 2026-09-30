"""Runaway-reasoning detection — the model thinks endlessly, never answers.

Some models, instead of producing a visible answer (or a tool call), emit a
continuous stream of *reasoning* tokens that never resolves into content. The
reasoning is non-repeating (high entropy), so the loop detector — which looks
for low entropy / verbatim repetition — cannot see it; and tokens are flowing,
so the stall watchdog (which looks for *silence*) cannot see it either. The
turn simply runs until it hits the output-window limit and terminates with
``finish_reason: "length"`` and an empty reply (see #17).

The fingerprint is the invariant **reasoning tokens keep flowing while content
tokens stay at zero**. Two windows observe it:

1. **Proactive (streaming):** a token-count watchdog over the reasoning stream.
   Once the reasoning token budget is exceeded with no content yet produced,
   the stream is aborted early so the remaining output budget can be spent on a
   retry that actually answers.
2. **Terminal:** the output window was exhausted (``finish_reason: "length"``)
   with reasoning present but no content and no tool calls — the model burned
   its entire budget thinking.

Remediation mirrors the nudge rung: re-submit with an explicit "stop thinking,
answer now" instruction, capped by a small attempt budget.

Only the standard library is used, so both classes are pure and unit-testable
without the gateway.
"""

from app.config import settings
from app.remediation.base import StreamVerdict


class RunawayReasoningDetector:
    """Flag a turn that keeps reasoning without ever producing an answer.

    Driven manually by the streaming loop (like :class:`StallDetector`): call
    :meth:`note_reasoning` on each reasoning delta and :meth:`note_content` when
    any visible content or tool-call fragment arrives. :attr:`triggered` becomes
    True once the reasoning token budget is exceeded while no content has
    appeared — a tool call counts as content (the model *is* acting).
    """

    # Characters per token approximation (matches the loop detector's
    # "~4 bytes/token" rule of thumb).
    CHARS_PER_TOKEN = 4.0

    def __init__(self, token_threshold: int = 2000) -> None:
        self.token_threshold = token_threshold
        self._reasoning_chars = 0
        self._saw_content = False

    @classmethod
    def from_settings(cls) -> "RunawayReasoningDetector":
        return cls(settings.runaway_reasoning_token_threshold)

    def note_reasoning(self, text: str) -> None:
        """Record reasoning text (accumulates the token budget)."""
        if text:
            self._reasoning_chars += len(text)

    def note_content(self) -> None:
        """Record that visible content (or a tool call) has appeared."""
        self._saw_content = True

    @property
    def reasoning_tokens(self) -> int:
        return int(self._reasoning_chars / self.CHARS_PER_TOKEN)

    @property
    def triggered(self) -> bool:
        """True once the reasoning budget is exceeded with no content yet."""
        if self._saw_content:
            return False
        return self._reasoning_chars >= self.token_threshold * self.CHARS_PER_TOKEN

    def verdict(self) -> StreamVerdict:
        """Build the abort verdict for a runaway-reasoning stream."""
        return StreamVerdict(
            kind="runaway_reasoning",
            reason=(
                f"reasoning token budget exceeded "
                f"({self.reasoning_tokens} tokens) with no content"
            ),
            details={
                "reasoning_tokens": self.reasoning_tokens,
                "token_threshold": self.token_threshold,
            },
        )


class RunawayReasoningPolicy:
    """Build the re-submission for a runaway turn: append a stop-thinking nudge.

    ``apply`` appends a ``user`` message telling the model to stop analysing
    and produce its answer (or make the tool call) now. Pure and unit-testable
    without the gateway.
    """

    def __init__(
        self,
        text: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.text = (
            text if text is not None else settings.runaway_reasoning_nudge_text
        )
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else settings.runaway_reasoning_max_attempts
        )

    @classmethod
    def from_settings(cls) -> "RunawayReasoningPolicy":
        return cls(
            settings.runaway_reasoning_nudge_text,
            settings.runaway_reasoning_max_attempts,
        )

    def apply(self, request_body: dict) -> dict:
        """Return a copy of the request body with the nudge message appended."""
        body = dict(request_body)
        messages = list(body.get("messages") or [])
        messages.append({"role": "user", "content": self.text})
        body["messages"] = messages
        return body
