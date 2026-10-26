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
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = ROOT / "config" / "prompt_rules.example.json"
SCRIPT_PATH = ROOT / "scripts" / "analyse-prompt-preambles"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "copilot_request_body.json"

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


def _load_analysis() -> Any:
    """Import the extensionless analysis script as a module."""
    loader = SourceFileLoader("analyse_prompt_preambles", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


analysis = _load_analysis()


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rules() -> list[dict[str, Any]]:
    document = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    return document["rules"]


def _record(request_id: str, body: dict[str, Any]) -> dict[str, Any]:
    response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
    return {
        "request_id": request_id,
        "timestamp": 1.0,
        "request_headers": {"x-interaction-id": "session-1"},
        "request_body": json.dumps(body),
        "response_body": json.dumps(response),
    }


def test_prompt_rules_example_passes_the_analysis_validator(
    rules: list[dict[str, Any]],
) -> None:
    document = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    validated, errors = analysis.validate_rules(document)
    assert errors == []
    assert validated == document["rules"]
    assert [rule["id"] for rule in validated] == [rule["id"] for rule in rules]


def test_prompt_rules_example_targets_system_messages_only(
    rules: list[dict[str, Any]],
) -> None:
    for rule in rules:
        role = rule["when"]["role"]
        assert role == "system" or role == ["system"], rule["id"]


def test_rules_ignore_matching_text_in_non_system_messages(
    rules: list[dict[str, Any]],
) -> None:
    body = _fixture()
    hostile = (
        "one\ntwo\nthree\nfour\nfive\nsix\nseven\n"
        "<instructions>not system</instructions>\n"
        "<example>\nexample line\n</example>\n"
        + "Use fenced code blocks\n" * 8
        + "</outputFormatting>"
    )
    body["messages"].append({"role": "user", "content": hostile})

    transformed, changes = analysis.apply_rules(body, rules)

    assert transformed["messages"][-1]["content"] == hostile
    assert {change["role"] for change in changes} == {"system"}


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


def test_every_rule_matches_exactly_the_system_message(
    rules: list[dict[str, Any]],
) -> None:
    body = _fixture()
    for rule in rules:
        matched = [message for message in body["messages"] if analysis.rule_matches(rule, message)]
        assert [message["role"] for message in matched] == ["system"], rule["id"]


def test_rules_trim_only_identity_and_output_formatting_example(
    rules: list[dict[str, Any]],
) -> None:
    body = _fixture()
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


def test_each_rule_is_a_no_op_on_its_own_output(rules: list[dict[str, Any]]) -> None:
    for rule in rules:
        once, _ = analysis.apply_rules(_fixture(), [rule])
        twice, changes = analysis.apply_rules(once, [rule])
        assert twice == once, rule["id"]
        assert changes == [], rule["id"]


def test_the_full_rule_set_is_idempotent(rules: list[dict[str, Any]]) -> None:
    once, _ = analysis.apply_rules(_fixture(), rules)
    twice, changes = analysis.apply_rules(once, rules)
    assert twice == once
    assert changes == []


def test_load_corpus_excludes_the_live_records_file(tmp_path: Path) -> None:
    raw = (
        "\n".join(
            [
                json.dumps(_record("r1", _fixture())),
                "{not json",
                json.dumps(_record("r2", _fixture())),
            ]
        ).encode()
        + b"\n"
    )
    (tmp_path / "records-20260101-000000.jsonl").write_bytes(raw)
    (tmp_path / "records.jsonl").write_text(
        json.dumps(_record("live", _fixture())), encoding="utf-8"
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "records-20260101-000000.jsonl").write_text(
        json.dumps(_record("r1", _fixture())) + "\n", encoding="utf-8"
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["totals"]["requests"] == 0
    assert report["corpus"]["sample_floor"]["met"] is False
    assert [rule["match_rate"] for rule in report["rules"]] == [0.0, 0.0, 0.0]
    assert report["saving"]["idempotent"] is True
