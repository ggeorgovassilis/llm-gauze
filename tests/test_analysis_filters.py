"""Tests for classification, filters, projection and pipeline composition."""

import json

import pytest
from analysis_samples import sample_records, tool_conversation_request
from app.analysis import (
    Digest,
    Filter,
    Project,
    Read,
    build_predicate,
    content_text,
    decode_body,
    diagnosis_reason,
    error_text,
    filter_records,
    has_tool_calls,
    is_abort,
    is_attempt,
    is_final,
    is_pass,
    normalise_tool_call,
    pass_rung,
    payload_objects,
    project,
    project_record,
    record_kind,
    record_rungs,
    request_messages,
    response_messages,
    tool_calls,
)
from app.analysis.classify import iter_pairs
from app.analysis.pipeline import default_records_path


def _records():
    return sample_records()


def _ids(records):
    return [record["id"] for record in records]


def test_record_kind_disambiguates_overloaded_outcomes():
    records = _records()
    kinds = {(record["request_id"], record["id"]): record_kind(record) for record in records}

    extract_pass = next(
        record
        for record in records
        if record["request_id"] == "req-extract" and record.get("extract")
    )
    final = next(
        record
        for record in records
        if record["request_id"] == "req-extract" and record.get("rounds") is not None
    )
    loop_abort = next(record for record in records if record.get("abort_kind") == "loop")
    loop_pass = next(record for record in records if record.get("loop_retry"))
    plain_success = next(record for record in records if record["request_id"] == "req-plain")

    assert record_kind(extract_pass) == "pass"
    assert record_kind(final) == "final"
    assert record_kind(loop_abort) == "abort"
    assert record_kind(loop_pass) == "pass"
    assert record_kind(plain_success) == "attempt"
    # Every record in the sample classifies into a known kind.
    assert set(kinds.values()) == {"attempt", "abort", "pass", "final"}


def test_is_helpers_and_pass_rung():
    records = _records()
    extract_pass = next(
        record for record in records if record.get("extract") and not record.get("rounds")
    )
    final = next(record for record in records if record.get("rounds") is not None)
    loop_abort = next(record for record in records if record.get("abort_kind") == "loop")

    assert is_pass(extract_pass) and not is_final(extract_pass) and not is_attempt(extract_pass)
    assert is_final(final) and not is_pass(final) and not is_attempt(final)
    assert is_abort(loop_abort) and not is_attempt(loop_abort)
    assert pass_rung(extract_pass) == "extract"
    assert pass_rung(final) is None
    assert record_rungs(final) == ["extract"]
    assert record_rungs(extract_pass) == ["extract"]


def test_tool_calls_are_normalised_and_results_preserved():
    record = tool_conversation_request("req-tools", 1.0)
    calls = tool_calls(record)
    assert [call["name"] for call in calls] == ["search", "buy"]
    assert calls[0]["arguments"] == {"q": "gpu"}
    assert has_tool_calls(record) is True
    assert tool_calls({"id": "x"}) == []


def test_filter_by_outcome_status_and_reason():
    records = _records()
    failures = list(filter_records(records, outcome="upstream_failure"))
    assert [record["outcome"] for record in failures] == ["upstream_failure"]

    http = list(filter_records(records, outcome=["http_error", "context_window_exceeded"]))
    assert {record["outcome"] for record in http} == {
        "http_error",
        "context_window_exceeded",
    }

    assert _ids(filter_records(records, status=503)) == ["req-http-1700000002.0-1-http_error"]
    assert len(list(filter_records(records, reason="retryable HTTP"))) == 1
    assert len(list(filter_records(records, error="ConnectError"))) == 1
    assert len(list(filter_records(records, retryable=True))) == 2
    assert len(list(filter_records(records, retryable=False))) == 1


def test_filter_disambiguates_aborts_passes_and_errors():
    records = _records()
    aborts = list(filter_records(records, kind="abort"))
    assert {record.get("abort_kind") for record in aborts} == {"stalled", "loop"}

    passes = list(filter_records(records, kind="pass"))
    assert {pass_rung(record) for record in passes} == {
        "extract",
        "coast",
        "empty_stop",
        "runaway",
        "loop_retry",
    }

    errors = list(filter_records(records, kind="error"))
    assert {record["outcome"] for record in errors} == {
        "upstream_failure",
        "http_error",
        "context_window_exceeded",
    }

    # kind=success selects every success outcome, including passes and finals.
    successes = list(filter_records(records, kind="success"))
    assert {record["outcome"] for record in successes} == {"success"}
    assert {record_kind(record) for record in successes} == {
        "attempt",
        "pass",
        "final",
    }

    # The loop outcome appears twice, but only one of them is an abort.
    loop_outcome = list(filter_records(records, outcome="loop_aborted"))
    assert len(loop_outcome) == 2
    assert len(list(filter_records(records, outcome="loop_aborted", kind="abort"))) == 1
    assert len(list(filter_records(records, outcome="loop_aborted", kind="pass"))) == 1


def test_filter_by_abort_kind_and_rung():
    records = _records()
    assert len(list(filter_records(records, abort_kind="stalled"))) == 1
    assert len(list(filter_records(records, rung="extract"))) == 3  # pass + two finals
    assert len(list(filter_records(records, rung="extract", kind="pass"))) == 1


def test_filter_by_path_query_time_window_and_attempt():
    records = _records()
    records.append(
        {
            "id": "other",
            "timestamp": 1_700_000_100.0,
            "request_id": "req-other",
            "method": "GET",
            "path": "/health",
            "query": "verbose=1",
            "outcome": "success",
            "attempt": 1,
            "status": 200,
            "response_body": "{}",
        }
    )
    assert len(list(filter_records(records, path="/v1/chat/completions"))) == len(records) - 1
    assert _ids(filter_records(records, path="/health")) == ["other"]
    assert len(list(filter_records(records, path="/v1/*"))) == len(records) - 1
    assert _ids(filter_records(records, query="verbose=1")) == ["other"]
    assert _ids(filter_records(records, since=1_700_000_100.0)) == ["other"]
    assert _ids(filter_records(records, until=1_700_000_001.0)) == [
        "req-retry-1700000000.0-1-upstream_failure",
        "req-retry-1700000001.0-2-success",
    ]
    assert len(list(filter_records(records, attempt=2))) == 2


def test_time_window_excludes_records_with_a_non_numeric_timestamp():
    records = [
        {"id": "good", "timestamp": 10.0},
        {"id": "bad", "timestamp": "yesterday"},
        {"id": "missing"},
    ]
    assert _ids(filter_records(records, since=0)) == ["good"]
    assert _ids(filter_records(records, until=10.0)) == ["good"]


def test_filter_by_request_id_tool_calls_and_predicate():
    records = _records()
    assert len(list(filter_records(records, request_id="req-tools"))) == 2
    tool_records = list(filter_records(records, has_tool_calls=True))
    assert {record["request_id"] for record in tool_records} == {"req-tools"}
    assert len(list(filter_records(records, has_tool_calls=False))) == len(records) - 2
    assert (
        len(list(filter_records(records, predicate=lambda record: record.get("status") == 503)))
        == 1
    )


def test_build_predicate_omitted_criteria_match_everything():
    predicate = build_predicate()
    assert all(predicate(record) for record in _records())


def test_projection_keeps_dotted_paths_aliases_and_nulls():
    record = {
        "id": "x",
        "outcome": "upstream_failure",
        "diagnosis": {"reason": "boom", "retryable": True},
        "response_body": None,
    }
    assert project_record(record, ["id", "diagnosis.reason"]) == {
        "id": "x",
        "diagnosis.reason": "boom",
    }
    assert project_record(record, {"why": "diagnosis.reason", "missing": "nope"}) == {
        "why": "boom",
        "missing": None,
    }
    projected = list(project([record], ["id", "outcome"]))
    assert projected == [{"id": "x", "outcome": "upstream_failure"}]


def test_pipeline_composes_read_filter_project_and_emits_json(tmp_path):
    from analysis_samples import write_recording

    write_recording(tmp_path, _records())
    pipeline = (
        Read(tmp_path)
        | Filter(kind="error")
        | Project(["request_id", "outcome", "diagnosis.reason"])
    )
    result = pipeline.run()
    records = list(result)
    assert {record["outcome"] for record in records} == {
        "upstream_failure",
        "http_error",
        "context_window_exceeded",
    }
    assert records[0]["request_id"] == "req-retry"
    assert pipeline.records() == records


def test_pipeline_ending_in_digest_returns_a_dict(tmp_path):
    from analysis_samples import write_recording

    write_recording(tmp_path, _records())
    result = (Read(tmp_path) | Digest("stuck")).run()
    assert isinstance(result, dict)
    assert result["digest"] == "stuck"
    with pytest.raises(TypeError):
        (Read(tmp_path) | Digest("stuck")).records()


def test_default_records_path_honours_environment(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp/somewhere")
    monkeypatch.setenv("RECORD_FILE", "custom.jsonl")
    assert default_records_path().as_posix() == "/tmp/somewhere/custom.jsonl"
    monkeypatch.delenv("DATA_DIR")
    monkeypatch.delenv("RECORD_FILE")
    assert default_records_path().as_posix() == "data/records.jsonl"


def test_unknown_records_and_empty_tails_are_handled():
    assert record_kind({"outcome": "something-else"}) == "unknown"
    # A pass with a discarded body but no nested rung block has no known rung.
    assert pass_rung({"response_body": None, "pre_remediation_body": "x"}) is None
    assert diagnosis_reason({"diagnosis": {"reason": 5}}) is None
    assert diagnosis_reason({"diagnosis": "not-a-dict"}) is None
    assert error_text({}) == ""
    assert decode_body("not-json") == {}
    assert request_messages({"request_body": "{}"}) == []
    assert list(iter_pairs([{}, {"request_id": "req"}, {"request_id": ""}])) == [
        ("unknown", {}),
        ("req", {"request_id": "req"}),
        ("unknown", {"request_id": ""}),
    ]


def test_payload_objects_accepts_objects_lists_and_sse():
    assert list(payload_objects({"a": 1})) == [{"a": 1}]
    assert list(payload_objects([{"a": 1}, 2, "x"])) == [{"a": 1}]
    assert list(payload_objects('[{"a": 1}, 2]')) == [{"a": 1}]
    assert list(payload_objects("")) == []
    assert list(payload_objects(42)) == []
    sse = 'data: {"a": 1}\ndata: not-json\ndata: [DONE]\nnot-a-data-line\n'
    assert list(payload_objects(sse)) == [{"a": 1}]


def test_message_extraction_skips_malformed_entries():
    body = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                "not-a-message",
                4,
            ]
        }
    )
    assert request_messages({"request_body": body}) == [{"role": "user", "content": "hi"}]

    response = json.dumps(
        {
            "messages": [{"role": "user", "content": "ignored"}],
            "choices": [
                "not-a-choice",
                {"message": "not-a-message"},
                {"message": {"role": "assistant", "content": "kept"}},
                {"message": {"tool_calls": [{"function": {"name": "t"}}]}},
            ],
        }
    )
    assert response_messages({"response_body": response}) == [
        {"role": "assistant", "content": "kept"},
        {"tool_calls": [{"function": {"name": "t"}}]},
    ]
    # A bare message object (not wrapped in choices/messages) is accepted.
    assert response_messages(
        {"response_body": json.dumps({"role": "assistant", "content": "bare"})}
    ) == [{"role": "assistant", "content": "bare"}]


def test_content_text_and_tool_call_normalisation_edges():
    assert content_text(None) is None
    assert content_text([{"type": "text"}]) is None
    assert content_text([{"text": "a"}, {"content": "b"}, "c", 5]) == "abc"
    assert normalise_tool_call("not-a-call") is None
    assert normalise_tool_call({"name": "n", "arguments": "not-json", "tool_call_id": "call"}) == {
        "id": "call",
        "name": "n",
        "arguments": "not-json",
    }


def test_tool_calls_skip_malformed_entries_and_include_partial_calls():
    record = {
        "request_body": json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            "not-a-call",
                            {"function": {"name": "ok", "arguments": "{}"}},
                        ],
                    }
                ]
            }
        ),
        "partial_tool_calls": ["not-a-call", {"name": "partial", "arguments": {}}],
    }
    assert [call["name"] for call in tool_calls(record)] == ["ok", "partial"]
