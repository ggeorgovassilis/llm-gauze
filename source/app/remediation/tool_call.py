"""Tool-call syntax enforcement — validate/repair/flag tool-call JSON.

Local models sometimes emit malformed or truncated tool calls: unbalanced
braces, cut-off ``function.arguments``, or a stray trailing comma. The gateway
currently forwards these verbatim, so the client's own JSON parse fails (or,
worse, a half-formed tool call is executed against an agent runtime).

This guard runs on the fully-assembled tool-call list (after streaming
reconstruction) and:

* validates each ``function.arguments`` as a JSON string;
* repairs where the breakage is deterministic — a truncated document whose
  braces/brackets/strings can be closed unambiguously;
* flags the rest (mangled structure, missing function/name) without crashing
  the request;
* reports every mutation so the recorder can log it (no silent mutation).

Only the standard library is used, so the guard is pure and unit-testable
without the gateway.

Scope note: this validates that ``arguments`` is *well-formed JSON*, not that
it matches the target function's schema — the gateway does not know the tool
schema (see #13 for AI-assisted extraction).
"""

import json

from app.config import settings


def _drop_trailing_comma(out: list[str]) -> None:
    """Drop a trailing comma (and surrounding whitespace) before a closer.

    ``{"a": 1,`` and ``[1, 2,`` are common truncation artefacts: the value was
    emitted and a comma for the *next* element is dangling when the stream was
    cut. A trailing comma inside a container is invalid JSON, so remove it
    before appending the matching closer.
    """
    i = len(out) - 1
    while i >= 0 and out[i] in " \t\n\r":
        i -= 1
    if i >= 0 and out[i] == ",":
        del out[i]


def _try_repair_arguments(text: str) -> str | None:
    """Repair a truncated JSON ``arguments`` string, or return ``None``.

    The only breakage we can fix deterministically is *truncation*: the input
    ends mid-string, or one or more containers are left open. Anything else
    (mismatched brackets, a dangling escape, garbage mid-document) is left to
    the caller to flag.
    """
    text = (text or "").strip()
    if not text:
        return "{}"

    stack: list[str] = []
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        out.append(ch)
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("{")
        elif ch == "[":
            stack.append("[")
        elif ch == "}":
            if not stack or stack[-1] != "{":
                return None
            stack.pop()
        elif ch == "]":
            if not stack or stack[-1] != "[":
                return None
            stack.pop()

    # A dangling escape (ends with a lone ``\``) is ambiguous — was it an
    # escaped quote or a half-written ``\n``? Leave it to the flagger.
    if escape:
        return None

    if in_string:
        out.append('"')

    # Close open containers innermost-first, trimming a trailing comma before
    # each closer (``{"a": 1,`` -> ``{"a": 1}``).
    for opener in reversed(stack):
        _drop_trailing_comma(out)
        out.append("}" if opener == "{" else "]")

    repaired = "".join(out)
    try:
        json.loads(repaired)
    except json.JSONDecodeError:
        return None
    return repaired


class ToolCallGuard:
    """Validate assembled tool calls; repair deterministic breakage; flag rest.

    ``validate(tool_calls)`` mutates the call list in place (repairing
    ``function.arguments`` where possible) and returns a list of change records
    for the recorder: ``{"kind": "repaired_arguments", "index": …}`` for a
    fix, ``{"kind": "flagged_malformed", "index": …, "reason": …}`` for
    something left as-is.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_settings(cls) -> "ToolCallGuard":
        return cls()

    def validate(self, tool_calls: list | None) -> list[dict]:
        changes: list[dict] = []
        for index, call in enumerate(tool_calls or []):
            changes.extend(self._validate_call(index, call))
        return changes

    def _validate_call(self, index: int, call: dict) -> list[dict]:
        if not isinstance(call, dict):
            return [
                {
                    "kind": "flagged_malformed",
                    "index": index,
                    "reason": "not_an_object",
                }
            ]

        fn = call.get("function")
        if not isinstance(fn, dict):
            return [
                {
                    "kind": "flagged_malformed",
                    "index": index,
                    "reason": "missing_function",
                }
            ]

        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            return [
                {
                    "kind": "flagged_malformed",
                    "index": index,
                    "reason": "missing_name",
                }
            ]

        arguments = fn.get("arguments")
        if arguments is None:
            arguments = ""
        if not isinstance(arguments, str):
            return [
                {
                    "kind": "flagged_malformed",
                    "index": index,
                    "reason": "arguments_not_string",
                }
            ]

        # Well-formed already: leave untouched.
        try:
            json.loads(arguments)
            return []
        except json.JSONDecodeError:
            pass

        repaired = _try_repair_arguments(arguments)
        if repaired is not None:
            fn["arguments"] = repaired
            return [{"kind": "repaired_arguments", "index": index}]

        return [
            {
                "kind": "flagged_malformed",
                "index": index,
                "reason": "unfixable_arguments",
            }
        ]
