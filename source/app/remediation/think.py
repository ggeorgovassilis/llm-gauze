"""Think-tag cleanup — relocate leaked thinking tags out of visible content.

Reasoning models occasionally emit their chain-of-thought as literal
``<think>…</think>`` / ``<reasoning>…</reasoning>`` markup inside the *visible*
``content`` field instead of the dedicated ``reasoning_content`` stream. The
result is either a polluted visible message or — when the model emits only the
tag and nothing else — an empty visible turn, which aborts Copilot's agentic
flow and strips the agent of any ability to remediate.

This guard runs on the fully-assembled completion (after streaming
reconstruction) and:

* **relocates** the inner text of any leaked thinking tag out of ``content``
  and appends it to ``reasoning`` — nothing is discarded;
* when the visible ``content`` is empty/whitespace but reasoning exists, emits
  a short configurable placeholder so the client still receives a non-empty
  response;
* reports exactly what it changed so the recorder can log it (no silent
  mutation).

Only the standard library is used, so the guard is pure and unit-testable
without the gateway.
"""

import re

from app.config import settings
from app.remediation.base import Transform

_DEFAULT_TAGS = ("think", "thinking", "reasoning")


class ThinkContentGuard(Transform):
    """Relocate leaked thinking tags and guarantee a non-empty visible reply.

    ``clean(content, reasoning)`` returns ``(content, reasoning, changes)``:
    the possibly-rewritten visible content and reasoning streams, plus a list
    of change records for the recorder.
    """

    def __init__(self, tags: tuple[str, ...], placeholder: str) -> None:
        self.tags = tuple(t.lower() for t in tags)
        self.placeholder = placeholder

    @classmethod
    def from_settings(cls) -> "ThinkContentGuard":
        tags = tuple(s.strip() for s in settings.think_tags.split(",") if s.strip())
        return cls(tags or _DEFAULT_TAGS, settings.think_empty_response_placeholder)

    # --- public API --------------------------------------------------

    def apply(
        self,
        content: str,
        reasoning: str = "",
        tool_calls: list | None = None,
    ) -> tuple[str, str, list[dict]]:
        """Canonical :class:`Transform` entry point (alias for :meth:`clean`)."""
        return self.clean(content, reasoning, tool_calls)

    def clean(
        self,
        content: str,
        reasoning: str = "",
        tool_calls: list | None = None,
    ) -> tuple[str, str, list[dict]]:
        """Rewrite ``content``/``reasoning``; return them plus change records.

        ``tool_calls`` is the assembled tool-call list (or ``None``). A turn
        that carries tool calls has *legitimately* empty visible content — the
        model is invoking a tool, not answering in prose — so the placeholder
        is never emitted for it.
        """
        content = content or ""
        reasoning = reasoning or ""
        changes: list[dict] = []

        content, reasoning, changes = self.relocate(content, reasoning)
        content, guard_changes = self.guard_empty(content, reasoning, tool_calls)
        changes.extend(guard_changes)
        return content, reasoning, changes

    def relocate(self, content: str, reasoning: str = "") -> tuple[str, str, list[dict]]:
        """Relocate leaked thinking tags only (no placeholder applied).

        The nudge rung inspects the relocated turn to decide whether to
        re-submit; the placeholder floor is applied separately afterwards.
        """
        content = content or ""
        reasoning = reasoning or ""
        changes: list[dict] = []
        return self._relocate(content, reasoning, changes)

    def guard_empty(
        self,
        content: str,
        reasoning: str = "",
        tool_calls: list | None = None,
    ) -> tuple[str, list[dict]]:
        """Apply the placeholder floor to an empty visible turn."""
        content = content or ""
        reasoning = reasoning or ""
        changes: list[dict] = []
        return self._guard_empty(content, reasoning, tool_calls, changes)

    # --- internals ---------------------------------------------------

    def _complete_pattern(self) -> re.Pattern:
        alternatives = "|".join(re.escape(t) for t in self.tags)
        # <tag>inner</tag> — DOTALL lets inner text span newlines; IGNORECASE
        # catches case variants. Non-greedy so the first closing tag wins.
        return re.compile(rf"<({alternatives})>(.*?)</\1>", re.IGNORECASE | re.DOTALL)

    def _unmatched_pattern(self) -> re.Pattern:
        alternatives = "|".join(re.escape(t) for t in self.tags)
        # A trailing opening tag with no matching close: "<think>truncated…".
        return re.compile(rf"<({alternatives})>(.*)$", re.IGNORECASE | re.DOTALL)

    def _relocate(
        self, content: str, reasoning: str, changes: list[dict]
    ) -> tuple[str, str, list[dict]]:
        """Move the inner text of leaked thinking tags into ``reasoning``."""
        inner_parts: list[str] = []

        def _replace(match: re.Match) -> str:
            inner_parts.append(match.group(2))
            return ""

        content = self._complete_pattern().sub(_replace, content)

        # A trailing unmatched opening tag means the model started a thinking
        # block and never closed it — the rest of the text is reasoning.
        unmatched = self._unmatched_pattern().search(content)
        if unmatched is not None:
            inner_parts.append(unmatched.group(2))
            content = content[: unmatched.start()]

        if inner_parts:
            extra = "\n".join(part for part in inner_parts)
            reasoning = f"{reasoning}\n{extra}" if reasoning else extra
            changes.append(
                {
                    "kind": "relocated_think",
                    "chars": sum(len(p) for p in inner_parts),
                    "blocks": len(inner_parts),
                }
            )
        return content, reasoning, changes

    def _guard_empty(
        self,
        content: str,
        reasoning: str,
        tool_calls: list | None,
        changes: list[dict],
    ) -> tuple[str, list[dict]]:
        """Guarantee non-empty visible content when reasoning was produced.

        A tool-call turn has empty content by design, so it is exempt: the
        model answered with a tool invocation, not prose.
        """
        if content.strip() == "" and reasoning.strip() != "" and not tool_calls:
            changes.append({"kind": "empty_content_placeholder"})
            return self.placeholder, changes
        return content, changes
