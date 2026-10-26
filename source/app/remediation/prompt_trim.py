"""Rule-driven request-side prompt trimming (opt-in).

The client preamble is re-sent on every turn and competes with the user's task
for a small local model's context window. This transform rewrites
``messages[*].content`` before the request is forwarded, using a small,
operator-supplied rule language so the effect is shareable, observable and
reversible without touching the client.

Rules are plain stdlib JSON (``version: 1`` plus an ordered ``rules`` list)
loaded once from :data:`app.config.settings.prompt_trim_rules_file`. Evaluation
is deterministic: no timestamps, ids, sizes or random values are inserted into
the output, so an identical ``(body, rules)`` pair yields byte-identical bytes
across turns and process restarts. That keeps a prompt-prefix cache valid.

Safety is deliberately conservative:

* only string ``messages[*].content`` is ever rewritten;
* the ``messages`` list is never emptied and the last ``user`` message is
  never dropped;
* a ``drop`` is skipped when it would orphan a tool call or a tool result;
* the scan is bounded per message because ``re`` has no timeout;
* every non-``messages`` request key is left untouched.

The transform reports one ``prompt_trim`` change record per rule application
and never logs prompt text.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+; used only to build a witness for load-time idempotence.
    from re import _parser as _re_parser  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - the parser ships with CPython
    _re_parser = None  # type: ignore[assignment]

from app.config import settings
from app.remediation.base import Transform

logger = logging.getLogger("llm_gauze.prompt_trim")

_ACTIONS = {"drop", "replace", "sub", "truncate"}
_DEFAULT_MARKER = "... [trimmed]"
_MAX_WITNESS_REPEAT = 1000


@dataclass(frozen=True)
class PromptRule:
    """A validated, compiled prompt-trim rule."""

    id: str
    action: str
    pattern: re.Pattern[str]
    roles: tuple[str, ...] | None
    min_chars: int = 0
    content: str | None = None
    replacement: str | None = None
    chars: int | None = None


@dataclass(frozen=True)
class PromptTrimResult:
    """Internal result of one :meth:`PromptTrimTransform.process` call."""

    body: dict
    changes: list[dict]
    scan_skips: int
    changed: bool


def _is_int(value: Any) -> bool:
    """True for real integers, excluding ``bool`` (a subclass of ``int``)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _role_tuple(role: Any) -> tuple[str, ...] | None:
    """Normalise a rule's ``when.role`` to a tuple; ``None`` matches any role."""
    if isinstance(role, str):
        return (role,)
    if isinstance(role, list):
        return tuple(role)
    return None


def _category_chars(category: Any) -> list[str]:
    """Pick a representative character for a regex ``CATEGORY``."""
    name = str(category)
    if "NOT_DIGIT" in name:
        return ["a"]
    if "DIGIT" in name:
        return ["0"]
    if "NOT_SPACE" in name:
        return ["a"]
    if "SPACE" in name:
        return [" "]
    if "NOT_WORD" in name:
        return ["!"]
    if "WORD" in name:
        return ["a"]
    return ["a"]


def _class_char(items: list[tuple[Any, Any]]) -> str | None:
    """Pick a character accepted by an ``IN`` (character class) node."""
    negate = False
    candidates: list[str] = []
    for op, value in items:
        name = str(op)
        if name == "NEGATE":
            negate = True
        elif name == "LITERAL":
            candidates.append(chr(value))
        elif name == "RANGE":
            candidates.append(chr(value[0]))
        elif name == "CATEGORY":
            candidates.extend(_category_chars(value))
    if not negate:
        return candidates[0] if candidates else "a"
    for char in "abcdefghijklmnopqrstuvwxyz0123456789 _-":
        if char not in candidates:
            return char
    return None


def _generate_witness(tokens: list[tuple[Any, Any]]) -> str | None:
    """Build one string matching a parsed regex, or ``None`` when unsupported.

    Only used for load-time idempotence validation; an unsupported construct
    merely falls back to the replacement-string check.
    """
    parts: list[str] = []
    for op, value in tokens:
        name = str(op)
        if name == "LITERAL":
            parts.append(chr(value))
        elif name == "NOT_LITERAL":
            parts.append("a" if value != ord("a") else "b")
        elif name == "ANY":
            parts.append("a")
        elif name == "IN":
            char = _class_char(value)
            if char is None:
                return None
            parts.append(char)
        elif name == "CATEGORY":
            parts.append(_category_chars(value)[0])
        elif name == "BRANCH":
            generated = _generate_witness(value[1][0])
            if generated is None:
                return None
            parts.append(generated)
        elif name == "SUBPATTERN":
            generated = _generate_witness(value[-1])
            if generated is None:
                return None
            parts.append(generated)
        elif name in {"MAX_REPEAT", "MIN_REPEAT"}:
            minimum, _, subpattern = value
            if minimum > _MAX_WITNESS_REPEAT:
                return None
            generated = _generate_witness(subpattern)
            if generated is None:
                return None
            parts.append(generated * minimum)
        elif name in {"AT", "ASSERT_NOT"}:
            continue
        elif name == "ASSERT":
            # A positive assertion's text must exist for the whole pattern to
            # match. Emitting it here places it correctly for a lookahead and
            # yields a match at a later position for a lookbehind.
            generated = _generate_witness(value[1])
            if generated is None:
                return None
            parts.append(generated)
        else:  # GROUPREF and anything newer/rarer: cannot build a witness.
            return None
    return "".join(parts)


def _witness_for(pattern: re.Pattern[str]) -> str | None:
    """Return one string that ``pattern.search`` matches, when available."""
    if _re_parser is None:  # pragma: no cover - CPython always has it
        return None
    try:
        parsed = _re_parser.parse(pattern.pattern, 0)
        witness = _generate_witness(list(parsed))
    except (re.error, ValueError, TypeError):  # pragma: no cover - defensive
        return None
    if witness is None:
        return None
    if pattern.search(witness) is not None:
        return witness
    # Lookbehind/assertion generation can need a little surrounding context.
    for prefix in ("", "a", "</example>"):
        for suffix in ("", "a"):
            candidate = prefix + witness + suffix
            if pattern.search(candidate) is not None:
                return candidate
    return None


def _content_matches(rule: PromptRule, role: str, content: str) -> bool:
    """Whether a candidate content still trips the rule's own ``when``."""
    if rule.roles is not None and role not in rule.roles:
        return False
    if len(content) < rule.min_chars:
        return False
    return rule.pattern.search(content) is not None


def _apply_action(rule: PromptRule, content: str, marker: str) -> str | None:
    """Apply a rule's action to ``content`` (``None`` means drop)."""
    if rule.action == "drop":
        return None
    if rule.action == "replace":
        return rule.content or ""
    if rule.action == "sub":
        return rule.pattern.sub(rule.replacement or "", content)
    assert rule.chars is not None  # validated
    chars = rule.chars
    if chars <= len(marker):
        # No room for content; a bounded slice of the marker still honours the
        # "marker counts within chars" bound.
        return marker[:chars]
    return content[: chars - len(marker)] + marker


def _is_non_idempotent(rule: PromptRule, marker: str) -> bool:
    """Whether the rule's own output would match the rule again."""
    if rule.action == "drop":
        return False
    role = rule.roles[0] if rule.roles else "system"
    witness = _witness_for(rule.pattern)
    if witness is not None:
        if len(witness) < rule.min_chars:
            padded = witness + "a" * (rule.min_chars - len(witness))
            if rule.pattern.search(padded) is not None:
                witness = padded
        if len(witness) >= rule.min_chars and rule.pattern.search(witness) is not None:
            output = _apply_action(rule, witness, marker)
            if output is not None and _content_matches(rule, role, output):
                return True
    # Fallback for patterns with no witness: the action's own text is the
    # output fragment that could re-match.
    candidates = [candidate for candidate in (rule.content, rule.replacement) if candidate]
    if rule.action == "truncate" and rule.chars is not None:
        candidates.append(marker[: rule.chars])
    for candidate in candidates:
        if candidate and _content_matches(rule, role, candidate):
            return True
    return False


def validate_rules(
    document: Any, marker: str = _DEFAULT_MARKER
) -> tuple[list[PromptRule], list[str]]:
    """Validate a parsed rule document, returning compiled rules and errors."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return [], ["rule document must be an object"]
    if not _is_int(document.get("version")) or document.get("version") != 1:
        errors.append("version must be 1")
    raw_rules = document.get("rules")
    if not isinstance(raw_rules, list):
        return [], [*errors, "rules must be a list"]

    rules: list[PromptRule] = []
    seen: set[str] = set()
    for raw in raw_rules:
        if not isinstance(raw, dict):
            errors.append("rule must be an object")
            continue
        rule_id = raw.get("id")
        if not isinstance(rule_id, str) or not rule_id:
            errors.append("rule id must be a non-empty string")
            continue
        if rule_id in seen:
            errors.append(f"duplicate rule id: {rule_id}")
            continue
        seen.add(rule_id)

        when = raw.get("when")
        then = raw.get("then")
        if not isinstance(when, dict):
            errors.append(f"{rule_id}: when must be an object")
            continue
        if not isinstance(then, dict):
            errors.append(f"{rule_id}: then must be an object")
            continue

        action = then.get("action")
        if action not in _ACTIONS:
            errors.append(f"{rule_id}: unknown action {action!r}")
            continue

        pattern_text = when.get("content_regex")
        if not isinstance(pattern_text, str):
            errors.append(f"{rule_id}: when.content_regex must be a string")
            continue
        try:
            pattern = re.compile(pattern_text)
        except re.error as exc:
            errors.append(f"{rule_id}: invalid regex: {exc}")
            continue

        role = when.get("role")
        if role is not None and not isinstance(role, (str, list)):
            errors.append(f"{rule_id}: when.role must be a string or list")
            continue
        roles = _role_tuple(role)
        if roles is not None and any(not isinstance(item, str) for item in roles):
            errors.append(f"{rule_id}: when.role entries must be strings")
            continue

        minimum = when.get("min_chars", 0)
        if not _is_int(minimum) or minimum < 0:
            errors.append(f"{rule_id}: when.min_chars must be a non-negative integer")
            continue

        content = replacement = None
        chars = None
        if action == "replace":
            content = then.get("content")
            if not isinstance(content, str):
                errors.append(f"{rule_id}: replace needs then.content")
                continue
        elif action == "sub":
            replacement = then.get("replacement")
            if not isinstance(replacement, str):
                errors.append(f"{rule_id}: sub needs then.replacement")
                continue
        elif action == "truncate":
            chars = then.get("chars")
            if not isinstance(chars, int) or isinstance(chars, bool) or chars < 0:
                errors.append(f"{rule_id}: truncate needs a non-negative integer then.chars")
                continue

        rule = PromptRule(
            id=rule_id,
            action=action,
            pattern=pattern,
            roles=roles,
            min_chars=minimum,
            content=content,
            replacement=replacement,
            chars=chars,
        )
        if _is_non_idempotent(rule, marker):
            errors.append(f"{rule_id}: rule is not idempotent; its output still matches")
            continue
        rules.append(rule)
    return rules, errors


class PromptTrimTransform(Transform):
    """A deterministic, rule-driven rewrite of ``messages[*].content``.

    Build one per process via :meth:`from_settings`: rule files are loaded and
    validated once, so an invalid file logs a single ``ERROR`` and disables
    trimming for the lifetime of the gateway. When disabled (or in dry-run,
    or when nothing matches) :meth:`process` returns the original body object,
    so a caller can forward the original bytes with their original
    ``Content-Length``.
    """

    def __init__(
        self,
        rules: list[PromptRule] | tuple[PromptRule, ...] = (),
        *,
        enabled: bool = True,
        disabled_reason: str | None = None,
        dry_run: bool = False,
        max_scan_chars: int = 200_000,
        marker: str = _DEFAULT_MARKER,
    ) -> None:
        self.rules = list(rules)
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.dry_run = dry_run
        self.max_scan_chars = max_scan_chars
        self.marker = marker

    @classmethod
    def from_settings(cls) -> "PromptTrimTransform":
        """Load and validate the configured rule file (once, at startup)."""
        marker = settings.prompt_trim_truncate_marker
        common: dict[str, Any] = {
            "dry_run": settings.prompt_trim_dry_run,
            "max_scan_chars": settings.prompt_trim_max_scan_chars,
            "marker": marker,
        }
        if not settings.prompt_trim_enabled:
            return cls(enabled=False, disabled_reason="disabled", **common)

        path = settings.prompt_trim_rules_file.strip()
        if not path:
            logger.error(
                "prompt trimming is enabled but PROMPT_TRIM_RULES_FILE is empty; "
                "disabling prompt trimming"
            )
            return cls(enabled=False, disabled_reason="invalid_rules", **common)
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error(
                "cannot load prompt-trim rule file %s: %s; disabling prompt trimming",
                path,
                exc,
            )
            return cls(enabled=False, disabled_reason="invalid_rules", **common)

        rules, errors = validate_rules(document, marker)
        if errors:
            logger.error(
                "invalid prompt-trim rule file %s: %s; disabling prompt trimming",
                path,
                "; ".join(errors),
            )
            return cls(enabled=False, disabled_reason="invalid_rules", **common)
        return cls(rules, **common)

    def apply(self, body: dict) -> tuple[dict, list[dict]]:
        """Canonical :class:`Transform` entry point: ``(body, changes)``."""
        result = self.process(body)
        return result.body, list(result.changes)

    def process(self, body: dict) -> PromptTrimResult:
        """Evaluate every rule and return the rewritten body plus changes.

        Dry-run and no-change paths return ``body`` itself (not a copy), which
        lets the caller detect that the original bytes should be forwarded.
        """
        if not self.enabled or not isinstance(body, dict):
            return PromptTrimResult(body, [], 0, False)
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return PromptTrimResult(body, [], 0, False)

        last_user = None
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user = index

        kept: list = []
        changes: list[dict] = []
        scan_skips = 0
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                kept.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, str):
                # ``null`` and list-of-parts content is skipped untouched.
                kept.append(message)
                continue
            if len(content) > self.max_scan_chars:
                scan_skips += 1
                logger.debug(
                    "prompt trim skipped index=%d chars=%d (scan bound)",
                    index,
                    len(content),
                )
                kept.append(message)
                continue

            current = content
            changed = False
            dropped = False
            for rule in self.rules:
                if not self._matches(rule, message.get("role"), current):
                    continue
                if rule.action == "drop":
                    if not self._drop_allowed(
                        index, message, last_user, len(kept), len(messages) - index - 1
                    ):
                        continue
                    change = self._change(rule, index, message.get("role"), len(current), 0)
                    changes.append(change)
                    self._log_change(change)
                    dropped = True
                    changed = True
                    break

                new_content = _apply_action(rule, current, self.marker)
                assert new_content is not None  # only ``drop`` returns None
                change = self._change(
                    rule, index, message.get("role"), len(current), len(new_content)
                )
                changes.append(change)
                self._log_change(change)
                current = new_content
                changed = True
                if rule.action == "replace":
                    break

            if dropped:
                continue
            if changed:
                new_message = dict(message)
                new_message["content"] = current
                kept.append(new_message)
            else:
                kept.append(message)

        if not changes:
            return PromptTrimResult(body, [], scan_skips, False)
        if self.dry_run:
            return PromptTrimResult(body, changes, scan_skips, False)
        new_body = dict(body)
        new_body["messages"] = kept
        return PromptTrimResult(new_body, changes, scan_skips, True)

    def _matches(self, rule: PromptRule, role: Any, content: str) -> bool:
        """Whether ``rule`` matches the current message state."""
        if rule.roles is not None and role not in rule.roles:
            return False
        if len(content) < rule.min_chars:
            return False
        return rule.pattern.search(content) is not None

    @staticmethod
    def _drop_allowed(
        index: int,
        message: dict,
        last_user: int | None,
        kept_count: int,
        remaining: int,
    ) -> bool:
        """Safety gate for ``drop``: never break pairing or empty the list."""
        if message.get("role") == "tool":
            return False
        if message.get("tool_calls"):
            return False
        if index == last_user:
            return False
        if kept_count == 0 and remaining == 0:
            return False
        return True

    @staticmethod
    def _change(
        rule: PromptRule, index: int, role: Any, chars_before: int, chars_after: int
    ) -> dict:
        return {
            "kind": "prompt_trim",
            "rule": rule.id,
            "index": index,
            "role": role,
            "action": rule.action,
            "chars_before": chars_before,
            "chars_after": chars_after,
        }

    @staticmethod
    def _log_change(change: dict) -> None:
        """One DEBUG line per application — sizes and ids, never prompt text."""
        logger.debug(
            "prompt trim rule=%s index=%s action=%s chars_before=%s chars_after=%s",
            change["rule"],
            change["index"],
            change["action"],
            change["chars_before"],
            change["chars_after"],
        )
