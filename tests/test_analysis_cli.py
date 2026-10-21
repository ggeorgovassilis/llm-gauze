"""Tests for the ``analyse-records`` CLI and pipeline grammar."""

import json

import pytest
from analysis_samples import sample_records, write_recording
from app.analysis import Digest, Filter, Project, Read
from app.analysis.cli import build_pipeline, main, parse_stage, render


def _run(tmp_path, pipeline_spec, *extra):
    write_recording(tmp_path, sample_records())
    return main(["read", str(tmp_path), *pipeline_spec.split(), *extra])


def test_build_pipeline_parses_filter_project_and_digest(tmp_path):
    pipeline = build_pipeline(
        "read | filter outcome=http_error,status=503,has_tool_calls=false | "
        "project id,outcome | digest endpoint",
        records=tmp_path,
    )
    assert isinstance(pipeline.stages[0], Read)
    assert isinstance(pipeline.stages[1], Filter)
    assert pipeline.stages[1].criteria == {
        "outcome": "http_error",
        "status": 503,
        "has_tool_calls": False,
    }
    assert isinstance(pipeline.stages[2], Project)
    assert pipeline.stages[2].fields == ["id", "outcome"]
    assert isinstance(pipeline.stages[3], Digest)
    assert pipeline.stages[3].name == "endpoint"


def test_parse_stage_projects_aliases_as_a_mapping():
    stage = parse_stage("project id,why=diagnosis.reason")
    assert isinstance(stage, Project)
    assert stage.fields == {"id": "id", "why": "diagnosis.reason"}

    alias_only = parse_stage("project why=diagnosis.reason")
    assert isinstance(alias_only, Project)
    assert alias_only.fields == {"why": "diagnosis.reason"}


def test_parse_stage_tolerates_empty_project_tokens():
    stage = parse_stage("project id,,why=diagnosis.reason,")
    assert isinstance(stage, Project)
    assert stage.fields == {"id": "id", "why": "diagnosis.reason"}


def test_parse_stage_rejects_malformed_project_aliases():
    with pytest.raises(ValueError, match="alias=path"):
        parse_stage("project =diagnosis.reason")
    with pytest.raises(ValueError, match="alias=path"):
        parse_stage("project why=")
    with pytest.raises(ValueError, match="duplicate project field"):
        parse_stage("project id,id=request_id")


def test_build_pipeline_adds_an_implicit_read(tmp_path):
    pipeline = build_pipeline("filter kind=abort | digest stuck", records=tmp_path)
    assert isinstance(pipeline.stages[0], Read)
    assert pipeline.stages[0].target == tmp_path


def test_build_pipeline_honours_read_path_and_options(tmp_path):
    pipeline = build_pipeline(f"read {tmp_path / 'records.jsonl'} | digest stuck")
    read = pipeline.stages[0]
    assert isinstance(read, Read)
    assert read.target == str(tmp_path / "records.jsonl")

    pipeline = build_pipeline(f"read {tmp_path} include_rotated=false | digest stuck")
    assert pipeline.stages[0].target == str(tmp_path)
    assert pipeline.stages[0].include_rotated is False


def test_parse_stage_rejects_bad_input():
    with pytest.raises(ValueError, match="unknown pipeline stage"):
        parse_stage("explode")
    with pytest.raises(ValueError, match="unknown filter"):
        parse_stage("filter nope=1")
    with pytest.raises(ValueError, match="expected key=value"):
        parse_stage("filter broken")
    with pytest.raises(ValueError, match="project requires"):
        parse_stage("project")
    with pytest.raises(ValueError, match="digest requires"):
        parse_stage("digest")
    with pytest.raises(ValueError, match="unknown read option"):
        parse_stage("read nope=1")


def test_build_pipeline_requires_digest_last(tmp_path):
    with pytest.raises(ValueError, match="digest must be the last"):
        build_pipeline("read | digest stuck | filter kind=abort", records=tmp_path)
    with pytest.raises(ValueError, match="empty pipeline"):
        build_pipeline("   ", records=tmp_path)


def test_cli_filter_and_project_outputs_jsonl(tmp_path, capsys):
    code = _run(
        tmp_path,
        "| filter outcome=upstream_failure | project id,outcome,diagnosis.reason",
        "--format",
        "jsonl",
    )
    assert code == 0
    output = capsys.readouterr().out.strip()
    record = json.loads(output)
    assert record["outcome"] == "upstream_failure"
    assert record["diagnosis.reason"].startswith("transient upstream error")


def test_cli_project_alias_renames_dotted_paths(tmp_path, capsys):
    code = _run(
        tmp_path,
        "| filter outcome=upstream_failure | project why=diagnosis.reason,id",
        "--format",
        "jsonl",
    )
    assert code == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert set(record) == {"why", "id"}
    assert record["why"].startswith("transient upstream error")


def test_cli_digest_outputs_json(tmp_path, capsys):
    code = _run(tmp_path, "| filter kind=abort | digest stuck", "--compact")
    assert code == 0
    output = capsys.readouterr().out
    assert "\n" not in output.strip()
    summary = json.loads(output)
    assert summary["digest"] == "stuck"
    assert summary["totals"]["aborts"] == 2


def test_cli_accepts_pipeline_option_and_data_dir(tmp_path, capsys):
    write_recording(tmp_path, sample_records())
    code = main(
        [
            "--data-dir",
            str(tmp_path),
            "--pipeline-spec",
            "read | digest remediation",
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["digest"] == "remediation"


def test_cli_records_option_selects_the_live_file(tmp_path, capsys):
    write_recording(tmp_path, sample_records())
    code = main(["--records", str(tmp_path / "records.jsonl"), "read | digest stuck"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["digest"] == "stuck"
    assert summary["totals"]["requests"] == 10


def test_cli_no_rotated_reads_only_live(tmp_path, capsys):
    write_recording(
        tmp_path,
        sample_records()[1:2],
        rotated=[("records-20260101-120000.jsonl", sample_records()[:1])],
    )
    code = main(
        [
            "read",
            str(tmp_path),
            "|",
            "project",
            "id",
            "--no-rotated",
            "--format",
            "jsonl",
        ]
    )
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1


def test_cli_requires_a_pipeline():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_cli_reports_unknown_digest(tmp_path, capsys):
    write_recording(tmp_path, sample_records())
    with pytest.raises(SystemExit) as exc:
        main(["read", str(tmp_path), "|", "digest", "nope"])
    assert exc.value.code == 2
    assert "unknown digest" in capsys.readouterr().err


def test_render_formats_dicts_and_streams():
    records = [{"id": 1}, {"id": 2}]
    assert json.loads(render(records)) == records
    assert render(records, output_format="jsonl").splitlines() == [
        '{"id": 1}',
        '{"id": 2}',
    ]
    assert render({"ok": True}, compact=True) == '{"ok": true}'
    assert render({"ok": True}, output_format="jsonl") == '{"ok": true}'


def test_render_serialises_non_json_values():
    assert render({"value": object()}).startswith('{\n  "value":')
