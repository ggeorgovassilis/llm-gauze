"""Message-overflow protection — warn/truncate oversized tool results.

A tool result can be arbitrarily large, and a local model has no way to know
it should summarise or skip it — it just receives a huge ``role: "tool"``
message that silently eats its context window (see #15). This guard is the
*request-side* mirror of think-cleanup: before the request is forwarded
upstream, any ``role: "tool"`` message whose content exceeds a configurable
threshold is (a) warned — a note telling the model the content was too large
and to find a workaround — and (b) optionally truncated to a bounded prefix so
the context is reclaimed.

Only ``role: "tool"`` messages are touched: they are the one input the model
*requested and can re-request more cheaply*, so the warning is actionable.
``user``/``system``/``assistant`` content is never modified.

The trigger is a pure size comparison (character count vs a threshold) — no
fuzzy heuristics. The threshold unit is *characters* (not bytes) so the trigger
and the truncation prefix share one unit; for ASCII they coincide with bytes.
Only the standard library is used, so the guard is pure and unit-testable
without the gateway.
"""

from app.config import settings


class MessageOverflowGuard:
    """Warn and optionally truncate oversized ``role: "tool"`` messages.

    ``process(body)`` returns ``(body, changes)``: the request body with any
    oversized tool results rewritten, plus a list of change records for the
    recorder — ``{"kind": "tool_overflow", "index": …, "size": …,
    "truncated": …}``. The input body is never mutated; when nothing trips the
    threshold, the original dict is returned unchanged.
    """

    def __init__(
        self,
        threshold: int | None = None,
        truncate: bool | None = None,
        warning: str | None = None,
    ) -> None:
        self.threshold = (
            threshold if threshold is not None
            else settings.message_overflow_threshold
        )
        self.truncate = (
            truncate if truncate is not None
            else settings.message_overflow_truncate
        )
        self.warning = (
            warning if warning is not None
            else settings.message_overflow_warning
        )

    @classmethod
    def from_settings(cls) -> "MessageOverflowGuard":
        return cls(
            settings.message_overflow_threshold,
            settings.message_overflow_truncate,
            settings.message_overflow_warning,
        )

    def process(self, body: dict) -> tuple[dict, list[dict]]:
        """Rewrite oversized tool results; return the body and change records."""
        if not isinstance(body, dict):
            return body, []
        messages = body.get("messages")
        if not isinstance(messages, list):
            return body, []

        changes: list[dict] = []
        new_messages: list | None = None
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue
            content = message.get("content")
            if not isinstance(content, str) or len(content) <= self.threshold:
                continue

            if new_messages is None:
                new_messages = list(messages)
            new_message = dict(message)
            new_message["content"] = self._rewrite(content)
            new_messages[index] = new_message
            changes.append(
                {
                    "kind": "tool_overflow",
                    "index": index,
                    "size": len(content),
                    "truncated": self.truncate,
                }
            )

        if new_messages is None:
            return body, []
        new_body = dict(body)
        new_body["messages"] = new_messages
        return new_body, changes

    def _rewrite(self, content: str) -> str:
        """Prepend the warning; truncate to a bounded prefix when enabled."""
        if self.truncate:
            return f"{self.warning}\n{self._bounded_prefix(content)}"
        return f"{self.warning}\n{content}"

    def _bounded_prefix(self, content: str) -> str:
        """First line, capped at the threshold (guards a single-line blob)."""
        line = content.split("\n", 1)[0]
        if len(line) > self.threshold:
            return line[: self.threshold] + "..."
        return line
