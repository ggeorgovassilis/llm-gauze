"""Unit-test the aggregate maths behind the prompt-preamble analysis.

The published figures in ``docs/prompt-trimming-analysis.md`` come from
``scripts/analyse-prompt-preambles``.  The corpus is local and gitignored, so
these tests drive the same implementation with synthetic records and pin the
percentile, shared-prefix, taxonomy, sample-floor, turn-semantics, request-id
dedup and saving calculations.
"""

from __future__ import annotations

import json
import shlex
import sys
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
    # Same history size, different final character: two conversations exercise
    # per-session pairing while keeping the history-size variants at one.
    altered = json.loads(json.dumps(copilot_body))
    altered["messages"][-1]["content"] = altered["messages"][-1]["content"][:-1] + "]"
    _write_corpus(
        tmp_path,
        {
            "records-20260101-000000.jsonl": [
                make_record("r1", copilot_body, session="session-a", timestamp=1.0),
                make_record("r2", copilot_body, session="session-a", timestamp=2.0),
                make_record("r3", altered, session="session-b", timestamp=1.0),
                make_record("r4", altered, session="session-b", timestamp=2.0),
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

    concat = analysis.content_concat(copilot_body)
    altered_concat = analysis.content_concat(altered)
    stable = report["stable_prefix_before"]
    # One consecutive pair per session; pairing across sessions would report 3.
    assert stable["consecutive_pairs"] == 2
    assert stable["sessions"] == 2
    assert stable["median"] == len(concat)
    assert stable["global_common_prefix_chars"] == analysis.shared_prefix_length(
        concat, altered_concat
    )
    assert stable["global_common_prefix_chars"] == len(concat) - 1


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


def test_turns_count_requests_not_user_messages(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The synthetic body carries at least two user-role messages
    # (environment/workspace and context/reminder wrappers) in one request.
    # ``turns`` is a request count, so it must stay at one and not follow the
    # user-message count.
    user_messages = [
        message for message in copilot_body["messages"] if message.get("role") == "user"
    ]
    assert len(user_messages) >= 2
    _write_corpus(
        tmp_path,
        {"records-20260101-000000.jsonl": [make_record("r1", copilot_body)]},
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["files"][0]["requests"] == 1
    assert report["corpus"]["files"][0]["turns"] == 1
    assert report["corpus"]["totals"]["turns"] == 1
    assert report["definitions"]["turn"] == (
        "a request whose message list contains at least one user message "
        "(a request-level unit, not a user-message count)"
    )


def test_duplicate_request_ids_count_once(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Retry/pass records repeat a request_id; the request count must dedupe
    # them even though every record is still reported in the raw record count.
    _write_corpus(
        tmp_path,
        {
            "records-20260101-000000.jsonl": [
                make_record("r1", copilot_body, timestamp=1.0),
                make_record("r1", copilot_body, timestamp=2.0),
            ]
        },
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["files"][0]["records"] == 2
    assert report["corpus"]["files"][0]["requests"] == 1
    assert report["corpus"]["files"][0]["turns"] == 1
    assert report["corpus"]["totals"]["records"] == 2
    assert report["corpus"]["totals"]["requests"] == 1
    assert report["corpus"]["totals"]["turns"] == 1


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

    # Pin the field the re-validation thresholds compare. One match removed
    # 25 characters: len("KEEP<example>REMOVE</example>") - len("KEEP").
    rule = report["rules"][0]
    assert rule["id"] == "drop-example"
    assert rule["median_chars_saved_per_matched_request"] == 25


def test_report_command_echoes_the_actual_arguments(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_corpus(
        tmp_path,
        {"records-20260101-000000.jsonl": [make_record("r1", copilot_body)]},
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["command"] == shlex.join(
        [
            "./scripts/analyse-prompt-preambles",
            "--data-dir",
            str(tmp_path),
            "--rules",
            str(RULES_PATH),
        ]
    )


def test_report_command_uses_the_invoked_program_name(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_corpus(
        tmp_path,
        {"records-20260101-000000.jsonl": [make_record("r1", copilot_body)]},
    )
    # Deliberately different from the script's repository path: the report must
    # echo the invoked program name rather than a hardcoded fallback.
    argv = [
        "/opt/llm-gauze/bin/analyse-prompt-preambles",
        "--data-dir",
        str(tmp_path),
        "--rules",
        str(RULES_PATH),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert analysis.main() == 0
    report = json.loads(capsys.readouterr().out)

    assert report["command"] == shlex.join(argv)


def test_corpus_excluded_path_follows_the_data_dir(
    analysis: Any,
    copilot_body: dict[str, Any],
    make_record: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_corpus(
        tmp_path,
        {"records-20260101-000000.jsonl": [make_record("r1", copilot_body)]},
    )

    assert analysis.main(["--data-dir", str(tmp_path), "--rules", str(RULES_PATH)]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["corpus"]["excluded"] == [
        f"{tmp_path / 'records.jsonl'} (live, still-growing file)"
    ]
