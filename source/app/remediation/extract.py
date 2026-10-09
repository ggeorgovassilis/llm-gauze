"""Extraction — recover a visible answer from a think-only model's reasoning.

A reasoning model finishes a turn with ``finish_reason: stop``, empty visible
``content``, no ``tool_calls``, but non-empty ``reasoning_content``: it answered
*inside* its chain-of-thought and delivered nothing. Retrying the identical
request re-derives the same thought stream from scratch, and re-submitting the
full stream as context would re-spend the context window — leaving no room to
answer.

Extraction instead treats the model's own reasoning as a *resource* and lets the
model page through it in bounded windows via a namespaced ``gauze_read`` tool
call. The answer is already in the reasoning; the model reads it back rather
than re-thinking it. The full stream never appears in any single request.

This replaces the old nudge rung: where nudge blindly re-prompted, extraction
carries the reasoning forward. The state machine is:

    unseeded -> seeded -> (read)* -> content

* **Seed** — a think-only turn (stop, empty content, no tool call, non-empty
  reasoning) captures the reasoning as the resource, appends the extraction
  instruction as a ``user`` message, and appends the ``gauze_read`` tool
  definition to ``request_body["tools"]`` (added, never replacing the client's
  tools).
* **Read-exec** — a turn whose ``tool_calls`` are *exclusively* ``gauze_read``
  slices the resource by line per call and replays the assistant tool-call
  message plus the matching ``role: "tool"`` result(s).

The ladder is constructed fresh per request, so per-request state on the rung
is safe. Only the standard library is used, so the policy is pure and
unit-testable without the gateway.
"""

import json

from app.config import settings
from app.remediation.base import Remediation, Turn

#: The namespaced read tool; matched by exact name. A turn carrying the
#: *client's* tool calls is never intercepted.
TOOL_NAME = "gauze_read"

#: Canonical resource name under which the captured reasoning is stored. There
#: is only ever one resource per request, so this is a fixed key the model
#: references in its ``gauze_read`` calls.
RESOURCE_NAME = "reasoning"


def _to_int(value, default: int) -> int:
    """Coerce ``value`` to an int, falling back to ``default``."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class ExtractionPolicy(Remediation):
    """Recover a visible answer from the reasoning via ``gauze_read`` paging.

    ``applies``/``apply`` dispatch on the turn's shape: a think-only turn seeds
    the resource, and a ``gauze_read``-only tool-call turn executes a read.
    Both consume the rung's shared ``max_attempts`` budget, exactly like
    nudge/coast.
    """

    name = "extract"

    def __init__(self, instruction: str, max_attempts: int) -> None:
        self.instruction = instruction
        self.max_attempts = max_attempts
        # Per-request state (safe: the ladder is fresh per request).
        self._resource: str | None = None
        self._last_mode: str | None = None
        self._last_read_args: list | None = None

    @classmethod
    def from_settings(cls) -> "ExtractionPolicy":
        return cls(settings.extract_instruction, settings.extract_max_attempts)

    # --- triggers -----------------------------------------------------

    def applies(self, turn: Turn, request_body: dict) -> bool:
        """The ladder's trigger: seed a think-only turn or execute a read.

        Read-exec fires only once seeded (the resource exists), and only when
        the turn's tool calls are *exclusively* ``gauze_read`` — a foreign
        (client) tool call is passed through untouched.
        """
        if self._resource is not None and self._is_read_exec(turn):
            return True
        return self._resource is None and self._is_think_only(turn)

    @staticmethod
    def _is_think_only(turn: Turn) -> bool:
        """True for a deterministically-empty turn worth extracting from."""
        return (
            turn.finish_reason == "stop"
            and not (turn.content or "").strip()
            and not turn.tool_calls
            and (turn.reasoning or "").strip() != ""
        )

    @staticmethod
    def _is_read_exec(turn: Turn) -> bool:
        """True when the turn's tool calls are all ``gauze_read`` (and non-empty)."""
        calls = turn.tool_calls
        return bool(calls) and all(ExtractionPolicy._is_gauze_read(c) for c in calls)

    @staticmethod
    def _is_gauze_read(call) -> bool:
        return (
            isinstance(call, dict)
            and isinstance(call.get("function"), dict)
            and call["function"].get("name") == TOOL_NAME
        )

    # --- action -------------------------------------------------------

    def apply(self, turn: Turn, request_body: dict) -> dict:
        """Return a copy of the request body for re-submission.

        Dispatches on the turn's shape: a tool-call turn is a read-exec, any
        other (guaranteed think-only) turn is a seed.
        """
        if self._is_read_exec(turn):
            return self._read_exec(turn, request_body)
        return self._seed(turn, request_body)

    def _seed(self, turn: Turn, request_body: dict) -> dict:
        """Capture the reasoning as the resource and register ``gauze_read``."""
        self._resource = turn.reasoning or ""
        self._last_mode = "seed"
        self._last_read_args = None

        body = dict(request_body)
        messages = list(body.get("messages") or [])
        messages.append({"role": "user", "content": self.instruction})
        body["messages"] = messages

        tools = list(body.get("tools") or [])
        tools.append(self._tool_definition())
        body["tools"] = tools
        return body

    def _read_exec(self, turn: Turn, request_body: dict) -> dict:
        """Slice the resource per ``gauze_read`` call and replay tool messages."""
        body = dict(request_body)
        messages = list(body.get("messages") or [])
        tool_calls = list(turn.tool_calls or [])

        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        read_args: list = []
        for call in tool_calls:
            args = self._parse_args(call)
            read_args.append(args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": self._read_slice(args),
                }
            )

        self._last_mode = "read"
        self._last_read_args = read_args
        body["messages"] = messages
        return body

    # --- resource reading ---------------------------------------------

    def _parse_args(self, call: dict) -> dict:
        """Parse a ``gauze_read`` call's ``function.arguments`` into a dict.

        ``ToolCallGuard`` already validated/repaired the JSON, but a model can
        still supply non-object arguments or wrong-typed values, so this is
        defensive and never raises.
        """
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                args = {}
        else:
            args = raw if isinstance(raw, dict) else {}
        return args if isinstance(args, dict) else {}

    def _read_slice(self, args: dict) -> str:
        """Return the requested line window of the captured resource."""
        lines = (self._resource or "").split("\n")
        line_from = _to_int(args.get("line_from"), 1)
        if line_from < 1:
            line_from = 1
        if args.get("line_count") is None:
            line_count = max(1, len(lines) - line_from + 1)
        else:
            line_count = _to_int(args.get("line_count"), 1)
            if line_count < 1:
                line_count = 1
        start = line_from - 1
        return "\n".join(lines[start : start + line_count])

    def _tool_definition(self) -> dict:
        """The ``gauze_read`` function tool definition appended on seed."""
        return {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "description": (
                    "Read a bounded window of your captured chain-of-thought "
                    "reasoning by line. Use this to page through your previous "
                    "reasoning and then produce the final visible answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "resource": {
                            "type": "string",
                            "description": (
                                f"Name of the reasoning resource to read ('{RESOURCE_NAME}')."
                            ),
                        },
                        "line_from": {
                            "type": "integer",
                            "description": "1-based line number of the first line to read.",
                        },
                        "line_count": {
                            "type": "integer",
                            "description": "Number of lines to read.",
                        },
                    },
                    "required": ["resource", "line_from", "line_count"],
                },
            },
        }

    # --- recorder/telemetry accessors ---------------------------------

    @property
    def resource_chars(self) -> int:
        """Length of the captured resource (0 before seed)."""
        return len(self._resource or "")

    @property
    def resource_lines(self) -> int:
        """Line count of the captured resource (0 before seed)."""
        return len((self._resource or "").split("\n"))

    @property
    def last_mode(self) -> str | None:
        """Mode of the most recent action: ``seed``, ``read``, or ``None``."""
        return self._last_mode

    @property
    def last_read_args(self) -> list | None:
        """Parsed ``gauze_read`` arguments of the most recent read (else None)."""
        return self._last_read_args
