"""Coast detection — re-prompt a turn that announced work but did none.

Some models, mid-way through a multi-step tool loop, end a turn with
``finish_reason: stop``, non-empty visible ``content``, and no ``tool_calls``
even though the request's ``tools`` list was non-empty and the conversation
already contains an assistant tool-call turn. The model "coasted": it emitted
the memorised status line ("Chunk 5 done … Pulling next chunk.") without
actually generating the call. Copilot's agent loop only continues while the
assistant emits tool calls, so such a turn ends the workflow silently — no
error, no crash, no user indication (see #16).

This is the sibling of the nudge rung: nudge fires on *empty* turns (no
visible content), this fires on *non-empty* turns whose chain-of-thought
collapsed to be byte-identical with the visible content — the deterministic
"coasting" fingerprint observed in the incident. Re-submitting with the
coasted turn replayed plus a short re-prompt gives the model a second chance
to actually emit the tool call.

The trigger is a conjunction of fully-deterministic signals:

* ``finish_reason == "stop"`` and ``tool_calls`` empty/absent;
* ``content`` non-empty (stripped);
* the request's ``tools`` list non-empty (a tool call was possible);
* a prior assistant message in the conversation carries ``tool_calls``
  (the model was driving a loop);
* ``reasoning`` stripped equals ``content`` stripped (the CoT collapsed).

Only the standard library is used, so the policy is pure and unit-testable
without the gateway.
"""

from app.config import settings
from app.remediation.base import Remediation, Turn


def _has_prior_assistant_tool_call(messages: list | None) -> bool:
    """Whether any assistant message in the conversation already emitted a
    tool call (evidence the model was driving a loop, not answering a query)."""
    for message in messages or []:
        if (
            isinstance(message, dict)
            and message.get("role") == "assistant"
            and message.get("tool_calls")
        ):
            return True
    return False


class CoastPolicy(Remediation):
    """Decide whether a coasted turn is worth re-prompting and build the
    re-submission.

    ``should_nudge`` is a pure predicate over the assembled turn plus the
    request's ``tools``/``messages``; ``apply`` appends the coasted assistant
    message and the re-prompt to a copy of the request body without mutating
    the input (pure, unit-testable without the gateway).
    """

    name = "coast"

    def __init__(
        self,
        text: str | None = None,
        max_attempts: int | None = None,
    ) -> None:
        self.text = text if text is not None else settings.coast_nudge_text
        self.max_attempts = (
            max_attempts if max_attempts is not None else settings.coast_max_attempts
        )

    @classmethod
    def from_settings(cls) -> "CoastPolicy":
        return cls(settings.coast_nudge_text, settings.coast_max_attempts)

    def should_nudge(
        self,
        finish_reason: str | None,
        content: str,
        tool_calls: list | None,
        reasoning: str,
        tools: list | None,
        messages: list | None,
    ) -> bool:
        """True for a deterministically-coasted turn that is worth re-prompting.

        The trigger is exact: the model *stopped* (not truncated, not calling a
        tool), produced visible text but no tool calls, *could* have called a
        tool (``tools`` present and the conversation already contains a prior
        assistant tool-call turn), and its reasoning collapsed to be
        byte-identical with its visible text — the coasting fingerprint from
        #16. A legitimate final answer has no pending tool loop (fails the
        prior-tool-call check) and normally has reasoning distinct from its
        content (fails the equality check), so it is exempt.
        """
        visible = (content or "").strip()
        return (
            finish_reason == "stop"
            and visible != ""
            and not tool_calls
            and bool(tools)
            and _has_prior_assistant_tool_call(messages)
            and (reasoning or "").strip() == visible
        )

    def applies(self, turn: Turn, request_body: dict) -> bool:
        """The ladder's trigger: a coasted turn (see :meth:`should_nudge`)."""
        return self.should_nudge(
            turn.finish_reason,
            turn.content,
            turn.tool_calls,
            turn.reasoning,
            request_body.get("tools"),
            request_body.get("messages"),
        )

    def apply(self, turn: Turn, request_body: dict) -> dict:
        """Return a copy of the request body with the coasted turn replayed and
        the re-prompt appended.

        Unlike nudge (whose empty turn left nothing in context), the coasted
        assistant message is replayed so the re-prompt ("you said you would
        call a tool; do so now") refers to something the model actually said.
        """
        body = dict(request_body)
        messages = list(body.get("messages") or [])
        messages.append({"role": "assistant", "content": turn.content})
        messages.append({"role": "user", "content": self.text})
        body["messages"] = messages
        return body
