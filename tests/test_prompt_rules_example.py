"""Guard the starter prompt-trim rule set against its synthetic fixture.

This is the ``#152`` deliverable that ``#153`` consumes: the example rule file
must stay valid, system-only, match every evidence class, and be idempotent on
a Copilot-shaped request body.  The fixture is synthetic; no recorded prompt
text belongs here.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "prompt_rules.example.json"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "copilot_request_body.json"


def _rules() -> dict[str, Any]:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _role_matches(role: Any, message: dict[str, Any]) -> bool:
    if isinstance(role, str):
        return message.get("role") == role
    if isinstance(role, list):
        return message.get("role") in role
    return False


def _matches(rule: dict[str, Any], message: dict[str, Any]) -> bool:
    when = rule["when"]
    if not _role_matches(when.get("role"), message):
        return False
    content = message.get("content")
    if not isinstance(content, str) or len(content) < when.get("min_chars", 0):
        return False
    return re.search(when["content_regex"], content) is not None


def _apply(body: dict[str, Any], rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the documented ``drop``/``replace``/``sub``/``truncate`` actions."""
    result = copy.deepcopy(body)
    messages = result["messages"]
    for rule in rules:
        then = rule["then"]
        action = then["action"]
        pattern = rule["when"]["content_regex"]
        kept: list[dict[str, Any]] = []
        for message in messages:
            if not _matches(rule, message):
                kept.append(message)
                continue
            content = message["content"]
            if action == "drop":
                continue
            if action == "replace":
                message["content"] = then["content"]
            elif action == "sub":
                message["content"] = re.sub(pattern, then["replacement"], content)
            elif action == "truncate":
                marker = then.get("marker", "")
                keep = max(0, int(then["chars"]) - len(marker))
                message["content"] = content[:keep] + marker
            else:  # pragma: no cover - guarded by the validation test
                raise AssertionError(f"unknown action {action!r}")
            kept.append(message)
        result["messages"] = kept
    return result


def test_prompt_rules_example_is_valid() -> None:
    document = _rules()
    assert document["version"] == 1
    rules = document["rules"]
    assert rules
    ids = [rule["id"] for rule in rules]
    assert len(ids) == len(set(ids))
    for rule in rules:
        assert isinstance(rule["evidence_class"], str) and rule["evidence_class"]
        assert rule["decision"] in {"Replace", "Discard"}
        assert 0.0 <= float(rule["match_rate"]) <= 1.0
        assert rule["then"]["action"] in {"drop", "replace", "sub", "truncate"}
        re.compile(rule["when"]["content_regex"])


def test_prompt_rules_example_targets_system_messages_only() -> None:
    for rule in _rules()["rules"]:
        role = rule["when"]["role"]
        assert role == "system" or role == ["system"]


def test_fixture_is_copilot_shaped() -> None:
    body = _fixture()
    assert {"messages", "model", "stream", "tools"} <= set(body)
    assert body["messages"][0]["role"] == "system"
    system = body["messages"][0]["content"]
    for tag in (
        "<instructions>",
        "<toolUseInstructions>",
        "<outputFormatting>",
        "<example>",
        "<modeInstructions>",
    ):
        assert tag in system
    assert "VSCODE_USER_PROMPTS_FOLDER" in system


def test_prompt_rules_example_matches_every_rule_on_the_fixture() -> None:
    body = _fixture()
    for rule in _rules()["rules"]:
        assert any(
            _matches(rule, message) for message in body["messages"] if isinstance(message, dict)
        ), rule["id"]


def test_prompt_rules_example_is_idempotent() -> None:
    rules = _rules()["rules"]
    once = _apply(_fixture(), rules)
    twice = _apply(once, rules)
    assert json.dumps(once, sort_keys=True) == json.dumps(twice, sort_keys=True)


def test_prompt_rules_example_reduces_the_fixture_and_keeps_non_messages() -> None:
    body = _fixture()
    rules = _rules()["rules"]
    transformed = _apply(body, rules)

    assert len(transformed["messages"][0]["content"]) < len(body["messages"][0]["content"])
    assert transformed["messages"][0]["content"].count("<example>") == 0
    assert transformed["messages"][1:] == body["messages"][1:]
    assert {key: value for key, value in transformed.items() if key != "messages"} == {
        key: value for key, value in body.items() if key != "messages"
    }
