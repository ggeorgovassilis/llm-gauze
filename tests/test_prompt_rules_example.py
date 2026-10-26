"""Guard the starter prompt-trim rule set against its synthetic fixture.

This is the ``#152`` deliverable that ``#153`` consumes: the example rule file
must stay valid, system-only, match every evidence class, and be idempotent on
a Copilot-shaped request body.  The fixture is synthetic; no recorded prompt
text belongs here.

The rule engine and corpus reader come from
``scripts/analyse-prompt-preambles`` -- the tool that produced the analysis --
so these tests exercise the delivered implementation instead of a copy of it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "prompt_rules.example.json"

_KEPT_SYSTEM_BLOCKS = (
    "<instructions>",
    "</instructions>",
    "<toolUseInstructions>",
    "</toolUseInstructions>",
    "<outputFormatting>",
    "</outputFormatting>",
    "<agents>",
    "<agent>",
    "<modeInstructions>",
    "task_complete",
    "VSCODE_USER_PROMPTS_FOLDER",
    "{{VARIABLE}}",
)


@pytest.fixture(scope="module")
def rules() -> list[dict[str, Any]]:
    document = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return document["rules"]


def test_prompt_rules_example_passes_the_analysis_validator(
    analysis: Any,
    rules: list[dict[str, Any]],
) -> None:
    document = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    validated, errors = analysis.validate_rules(document)
    assert errors == []
    assert validated == document["rules"]
    assert [rule["id"] for rule in validated] == [rule["id"] for rule in rules]


def test_prompt_rules_example_keeps_its_analysis_metadata(rules: list[dict[str, Any]]) -> None:
    for rule in rules:
        assert rule["evidence_class"], rule["id"]
        assert rule["decision"] in {"Replace", "Discard"}, rule["id"]
        assert 0.0 <= float(rule["match_rate"]) <= 1.0, rule["id"]


def test_prompt_rules_example_targets_system_messages_only(
    rules: list[dict[str, Any]],
) -> None:
    for rule in rules:
        role = rule["when"]["role"]
        assert role == "system" or role == ["system"], rule["id"]


def test_rules_ignore_matching_text_in_non_system_messages(
    analysis: Any,
    copilot_body: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    hostile = (
        "one\ntwo\nthree\nfour\nfive\nsix\nseven\n"
        "<instructions>not system</instructions>\n"
        "<example>\nexample line\n</example>\n"
        + "Use fenced code blocks\n" * 8
        + "</outputFormatting>"
    )
    copilot_body["messages"].append({"role": "user", "content": hostile})

    transformed, changes = analysis.apply_rules(copilot_body, rules)

    assert transformed["messages"][-1]["content"] == hostile
    assert {change["role"] for change in changes} == {"system"}


def test_fixture_is_copilot_shaped(copilot_body: dict[str, Any]) -> None:
    assert {"messages", "model", "stream", "tools"} <= set(copilot_body)
    assert copilot_body["messages"][0]["role"] == "system"
    system = copilot_body["messages"][0]["content"]
    for tag in (
        "<instructions>",
        "<toolUseInstructions>",
        "<outputFormatting>",
        "<example>",
        "<modeInstructions>",
    ):
        assert tag in system
    assert "VSCODE_USER_PROMPTS_FOLDER" in system


def test_every_rule_matches_exactly_the_system_message(
    analysis: Any,
    copilot_body: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    for rule in rules:
        matched = [
            message for message in copilot_body["messages"] if analysis.rule_matches(rule, message)
        ]
        assert [message["role"] for message in matched] == ["system"], rule["id"]


def test_rules_trim_only_identity_and_output_formatting_example(
    analysis: Any,
    copilot_body: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    body = copilot_body
    original = body["messages"][0]["content"]

    transformed, changes = analysis.apply_rules(body, rules)
    system = transformed["messages"][0]["content"]

    assert original.splitlines()[0] not in system
    assert "<example>" not in system
    for block in _KEPT_SYSTEM_BLOCKS:
        assert block in system, block
    for removed in (
        "Use fenced code blocks with a language hint.",
        "Use small tables for comparisons.",
    ):
        assert removed not in system, removed

    saved = len(original) - len(system)
    assert 0 < saved < len(original) // 3
    assert [message["role"] for message in transformed["messages"]] == [
        message["role"] for message in body["messages"]
    ]
    assert transformed["messages"][1:] == body["messages"][1:]
    assert {key: value for key, value in transformed.items() if key != "messages"} == {
        key: value for key, value in body.items() if key != "messages"
    }
    assert {change["rule"] for change in changes} == {rule["id"] for rule in rules}


def test_each_rule_is_a_no_op_on_its_own_output(
    analysis: Any,
    copilot_body: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    for rule in rules:
        once, _ = analysis.apply_rules(copilot_body, [rule])
        twice, changes = analysis.apply_rules(once, [rule])
        assert twice == once, rule["id"]
        assert changes == [], rule["id"]


def test_the_full_rule_set_is_idempotent(
    analysis: Any,
    copilot_body: dict[str, Any],
    rules: list[dict[str, Any]],
) -> None:
    once, _ = analysis.apply_rules(copilot_body, rules)
    twice, changes = analysis.apply_rules(once, rules)
    assert twice == once
    assert changes == []


def test_load_corpus_excludes_the_live_records_file(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
) -> None:
    raw = (
        "\n".join(
            [
                json.dumps(make_record("r1", copilot_body)),
                "{not json",
                json.dumps(make_record("r2", copilot_body)),
            ]
        ).encode()
        + b"\n"
    )
    (tmp_path / "records-20260101-000000.jsonl").write_bytes(raw)
    (tmp_path / "records.jsonl").write_text(
        json.dumps(make_record("live", copilot_body)), encoding="utf-8"
    )

    records, manifest = analysis.load_corpus(tmp_path)

    assert [entry["file"] for entry in manifest] == ["records-20260101-000000.jsonl"]
    entry = manifest[0]
    assert entry["records"] == 2
    assert entry["corrupt_lines"] == 1
    assert entry["bytes"] == len(raw)
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert [record["request_id"] for record in records] == ["r1", "r2"]


def test_main_reports_evidence_without_prompt_text(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "records-20260101-000000.jsonl").write_text(
        json.dumps(make_record("r1", copilot_body)) + "\n", encoding="utf-8"
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    out = capsys.readouterr().out

    report = json.loads(out)
    assert report["corpus"]["totals"]["requests"] == 1
    assert report["corpus"]["sample_floor"]["met"] is False
    assert report["saving"]["requests_changed"] == 1
    assert report["saving"]["idempotent"] is True
    assert [rule["match_rate"] for rule in report["rules"]] == [1.0, 1.0, 1.0]
    assert "synthetic coding assistant" not in out
    assert "Add a unit test for the parser" not in out


def test_main_reports_an_empty_corpus_without_crashing(
    analysis: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["totals"]["requests"] == 0
    assert report["corpus"]["sample_floor"]["met"] is False
    assert [rule["match_rate"] for rule in report["rules"]] == [0.0, 0.0, 0.0]
    assert report["saving"]["idempotent"] is True
