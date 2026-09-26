"""Nudge — re-submit an empty-text turn with a short re-prompt.

Some reasoning models finish a turn with ``finish_reason: stop``, no visible
``content``, no ``tool_calls``, but non-empty ``reasoning_content``: they
thought their way to an answer and then delivered nothing. Retrying the
*identical* request reproduces the same empty turn, but re-submitting with one
appended user message ("you produced no visible output; emit content or a tool
call") gives the model a second chance to surface a real answer.

This is the first rung of the empty-response ladder — above the #11 placeholder
floor and below #13 extraction — so it is deliberately tiny and deterministic:

* ``should_nudge`` is a pure predicate over the assembled turn (no heuristics);
* ``apply`` appends the nudge message to a copy of the request body without
  mutating the input (pure, unit-testable without the gateway).
"""

from app.config import settings


class NudgePolicy:
    """Decide whether to nudge an empty turn and build the nudged request."""

    def __init__(
        self,
        text: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.text = text if text is not None else settings.think_nudge_text
        self.max_attempts = (
            max_attempts
            if max_attempts is not None
            else settings.think_nudge_max_attempts
        )

    @classmethod
    def from_settings(cls) -> "NudgePolicy":
        return cls(settings.think_nudge_text, settings.think_nudge_max_attempts)

    def should_nudge(
        self,
        finish_reason: str | None,
        content: str,
        tool_calls: list | None,
        reasoning: str,
    ) -> bool:
        """True for a deterministically-empty turn that is worth re-prompting.

        The trigger is exact: the model *stopped* (not truncated, not calling a
        tool), produced no visible text and no tool calls, but did produce
        reasoning. A tool-call turn legitimately has empty content, so it is
        exempt — as is any turn that produced visible text.
        """
        return (
            finish_reason == "stop"
            and not (content or "").strip()
            and not tool_calls
            and (reasoning or "").strip() != ""
        )

    def apply(self, request_body: dict) -> dict:
        """Return a copy of the request body with the nudge message appended."""
        body = dict(request_body)
        messages = list(body.get("messages") or [])
        messages.append({"role": "user", "content": self.text})
        body["messages"] = messages
        return body
