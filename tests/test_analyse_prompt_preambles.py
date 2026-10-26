"""Unit-test the aggregate maths behind the prompt-preamble analysis.

The published figures in ``docs/prompt-trimming-analysis.md`` come from
``scripts/analyse-prompt-preambles``.  The corpus is local and gitignored, so
these tests drive the same implementation with synthetic records and pin the
percentile, shared-prefix, taxonomy, sample-floor and saving calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "prompt_rules.example.json"


def _with_session_id(body: dict[str, Any], value: str) -> None:
    """Place ``session <value>`` inside the system identity block."""
    content = body["messages"][0]["content"]
    body["messages"][0]["content"] = content.replace(
        "<instructions>", f"session {value}\n<instructions>", 1
    )


def _write_corpus(directory: Path, files: dict[str, list[dict[str, Any]]]) -> None:
    for name, records in files.items():
        lines = "\n".join(json.dumps(record) for record in records)
        (directory / name).write_text(lines + "\n", encoding="utf-8")


def test_percentile_uses_nearest_rank_and_handles_empty_input(analysis: Any) -> None:
    assert analysis.percentile([], 95) == 0.0
    assert analysis.percentile([10, 20, 30, 40, 50], 20) == 10
    assert analysis.percentile([10, 20, 30, 40, 50], 50) == 30
    assert analysis.percentile([10, 20, 30, 40, 50], 95) == 50
    assert analysis.percentile([10, 20, 30, 40, 50], 100) == 50


def test_shared_prefix_length_counts_only_the_common_head(analysis: Any) -> None:
    assert analysis.shared_prefix_length("", "") == 0
    assert analysis.shared_prefix_length("abc", "") == 0
    assert analysis.shared_prefix_length("abc", "abd") == 2
    assert analysis.shared_prefix_length("abc", "abc") == 3
    assert analysis.shared_prefix_length("abc", "abcdef") == 3


def test_summary_reports_median_and_p95(analysis: Any) -> None:
    result = analysis.summary([1, 2, 3, 4, 5])

    assert result["count"] == 5
    assert result["median"] == 3
    assert result["p95"] == 5
    assert result["min"] == 1
    assert result["max"] == 5


def test_main_builds_taxonomy_and_stable_prefix_from_synthetic_records(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_corpus(
        tmp_path,
        {
            "records-20260101-000000.jsonl": [
                make_record("r1", copilot_body, session="session-a", timestamp=1.0),
                make_record("r2", copilot_body, session="session-a", timestamp=2.0),
            ]
        },
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    taxonomy = {row["id"]: row for row in report["taxonomy"]}
    for block in ("system.identity", "request.tools", "conversation.history"):
        assert block in taxonomy, block
    history = taxonomy["conversation.history"]
    assert history["variants"] == 1
    assert history["dominant_share"] == 1.0

    stable = report["stable_prefix_before"]
    assert stable["consecutive_pairs"] == 1
    assert stable["sessions"] == 1
    assert stable["global_common_prefix_chars"] == len(analysis.content_concat(copilot_body))


def test_taxonomy_variants_ignore_session_scoped_identifiers(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = copilot_body
    second = json.loads(json.dumps(first))
    _with_session_id(first, "0123456789abcdef")
    _with_session_id(second, "0123456789abcdee")
    _write_corpus(
        tmp_path,
        {
            "records-20260101-000000.jsonl": [
                make_record("r1", first),
                make_record("r2", second),
            ]
        },
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    identity = next(row for row in report["taxonomy"] if row["id"] == "system.identity")
    assert identity["variants"] == 1
    assert identity["dominant_share"] == 1.0


def test_taxonomy_counts_variants_that_differ_outside_session_ids(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = copilot_body
    second = json.loads(json.dumps(first))
    _with_session_id(first, "alpha")
    _with_session_id(second, "beta")
    _write_corpus(
        tmp_path,
        {
            "records-20260101-000000.jsonl": [
                make_record("r1", first),
                make_record("r2", second),
            ]
        },
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    identity = next(row for row in report["taxonomy"] if row["id"] == "system.identity")
    assert identity["variants"] == 2
    assert identity["dominant_share"] == 0.5


def test_sample_floor_requires_a_user_turn_in_every_file(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    system_only = {"messages": [{"role": "system", "content": "Synthetic system prompt."}]}
    files = {
        "records-20260101-000000.jsonl": [
            make_record(f"a{index}", copilot_body) for index in range(34)
        ],
        "records-20260102-000000.jsonl": [
            make_record(f"b{index}", copilot_body) for index in range(33)
        ],
        "records-20260103-000000.jsonl": [
            make_record(f"c{index}", system_only) for index in range(33)
        ],
    }
    _write_corpus(tmp_path, files)

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["totals"]["requests"] == 100
    assert report["corpus"]["sample_floor"] == {
        "minimum_requests": 100,
        "minimum_files": 3,
        "minimum_turns_per_file": 1,
        "met": False,
    }


def test_sample_floor_is_met_when_every_file_has_a_user_turn(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    files = {
        "records-20260101-000000.jsonl": [
            make_record(f"a{index}", copilot_body) for index in range(34)
        ],
        "records-20260102-000000.jsonl": [
            make_record(f"b{index}", copilot_body) for index in range(33)
        ],
        "records-20260103-000000.jsonl": [
            make_record(f"c{index}", copilot_body) for index in range(33)
        ],
    }
    _write_corpus(tmp_path, files)

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["sample_floor"]["met"] is True


def test_preamble_share_after_uses_the_trimmed_body(
    analysis: Any,
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    system = "KEEP<example>REMOVE</example>"
    body = {"model": "synthetic", "messages": [{"role": "system", "content": system}]}
    rules = {
        "version": 1,
        "rules": [
            {
                "id": "drop-example",
                "when": {"role": "system", "content_regex": "<example>.*?</example>"},
                "then": {"action": "sub", "replacement": ""},
            }
        ],
    }
    rules_path = tmp_path / "rules.json"
    rules_path.write_text(json.dumps(rules), encoding="utf-8")
    _write_corpus(tmp_path, {"records-20260101-000000.jsonl": [make_record("r1", body)]})

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(rules_path)]) == 0
    report = json.loads(capsys.readouterr().out)

    trimmed_system = "KEEP"
    trimmed_body = {
        "model": "synthetic",
        "messages": [{"role": "system", "content": trimmed_system}],
    }
    trimmed_length = len(json.dumps(trimmed_body, separators=(",", ":")))
    expected = len(trimmed_system) / trimmed_length

    saving = report["saving"]
    assert "preamble_share_of_body_after" not in saving
    assert saving["preamble_share_of_trimmed_body_after"]["median"] == pytest.approx(expected)
    assert saving["preamble_chars_after"]["median"] == len(trimmed_system)
