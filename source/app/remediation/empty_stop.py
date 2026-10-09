"""Empty-stop detection — re-prompt a turn that thought then emitted nothing.

Some reasoning models finish a turn with ``finish_reason: stop``, empty visible
``content``, no ``tool_calls``, and no ``reasoning_content`` — while still
consuming reasoning tokens (``completion_tokens_details.reasoning_tokens > 0``
and ``text_tokens == 0``). The model reasoned internally, delivered nothing,
and stopped, which silently ends an agentic tool loop (see #135).

The existing rungs cannot catch this: extraction (#13) requires non-empty
``reasoning_content`` (this model never exposes it), coast (#16) requires
non-empty visible content, and the loop/runaway watchdogs require a loop.
There is genuinely nothing to extract — the reasoning is internal and lost —
so the only recourse is to re-submit the turn with a re-prompt appended, up to
a configurable attempt budget.

The trigger is the exact silent-empty-stop fingerprint:

* ``finish_reason == "stop"`` and ``tool_calls`` empty/absent;
* visible ``content`` empty (stripped);
* ``reasoning_content`` empty (stripped);
* ``reasoning_tokens > 0`` and ``text_tokens == 0`` (where usage is reported).

Only the standard library is used, so the policy is pure and unit-testable
without the gateway.
"""

from app.config import settings
from app.remediation.base import Remediation, Turn


def _to_optional_int(value) -> int | None:
    """Coerce ``value`` to an int, returning ``None`` when absent/invalid."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def token_counts(meta: dict | None) -> tuple[int | None, int | None]:
    """Pull ``(reasoning_tokens, text_tokens)`` from a turn's usage metadata.

    LiteLLM reports ``completion_tokens_details`` on the final streamed chunk;
    the proxy folds it into the turn's ``meta["usage"]``. Returns ``(None,
    None)`` when the upstream did not report usage — the rung then cannot
    distinguish a silent empty stop from an ordinary empty stop and stays
    quiet.
    """
    usage = (meta or {}).get("usage")
    if not isinstance(usage, dict):
        return None, None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None, None
    return (
        _to_optional_int(details.get("reasoning_tokens")),
        _to_optional_int(details.get("text_tokens")),
    )


class EmptyStopPolicy(Remediation):
    """Decide whether a silent empty stop is worth re-prompting and build the
    re-submission.

    ``should_nudge`` is a pure predicate over the assembled turn; ``apply``
    appends the re-prompt as a ``user`` message to a copy of the request body
    without mutating the input (pure, unit-testable without the gateway).
    """

    name = "empty_stop"

    def __init__(self, text: str, max_attempts: int) -> None:
        self.text = text
        self.max_attempts = max_attempts

    @classmethod
    def from_settings(cls) -> "EmptyStopPolicy":
        return cls(settings.empty_stop_nudge_text, settings.empty_stop_max_attempts)

    def should_nudge(
        self,
        finish_reason: str | None,
        content: str,
        tool_calls: list | None,
        reasoning: str,
        reasoning_tokens: int | None,
        text_tokens: int | None,
    ) -> bool:
        """True for the exact silent-empty-stop fingerprint.

        The model *stopped* (not truncated, not calling a tool), produced no
        visible text, no tool call, and no reasoning content — yet the upstream
        usage proves it consumed reasoning tokens without producing any text
        tokens. A turn with visible content is coast (or a real answer); a turn
        with non-empty reasoning is extraction; a turn with neither token
        signal is indistinguishable from an ordinary empty stop, so it is
        exempt.
        """
        return (
            finish_reason == "stop"
            and not (content or "").strip()
            and not tool_calls
            and not (reasoning or "").strip()
            and reasoning_tokens is not None
            and text_tokens is not None
            and reasoning_tokens > 0
            and text_tokens == 0
        )

    def applies(self, turn: Turn, request_body: dict) -> bool:
        """The ladder's trigger: a silent empty stop (see :meth:`should_nudge`)."""
        return self.should_nudge(
            turn.finish_reason,
            turn.content,
            turn.tool_calls,
            turn.reasoning,
            turn.reasoning_tokens,
            turn.text_tokens,
        )

    def apply(self, turn: Turn, request_body: dict) -> dict:
        """Return a copy of the request body with the re-prompt appended.

        Unlike extraction (which replays the captured reasoning) or coast
        (which replays the coasted assistant message), the empty assistant turn
        left nothing in context, so only the re-prompt is appended.
        """
        body = dict(request_body)
        messages = list(body.get("messages") or [])
        messages.append({"role": "user", "content": self.text})
        body["messages"] = messages
        return body
