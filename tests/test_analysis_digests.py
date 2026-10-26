"""Tests for the four ready-made digests."""

import json

import pytest
from analysis_samples import sample_records, tool_call, tool_conversation_request
from app.analysis import endpoint, narrative, remediation, run_digest, stuck


def test_stuck_builds_abort_timeline_and_flags_false_positives():
    summary = stuck(sample_records())

    assert summary["digest"] == "stuck"
    assert summary["totals"] == {
        "requests": 10,
        "aborts": 2,
        "passes": 5,
        "attempts": 7,
    }
    assert summary["abort_kinds"] == {"loop": 1, "stalled": 1}
    assert [entry["abort_kind"] for entry in summary["timeline"]] == ["stalled", "loop"]
    assert summary["timeline"][1]["abort_stream"] == "content"
    assert summary["timeline"][1]["partial_content_chars"] == len("same same same")

    false_positives = summary["stall_false_positives"]
    assert len(false_positives) == 1
    assert false_positives[0]["signals"] == ["partial_content"]

    storms = {request["request_id"]: request for request in summary["retry_storms"]}
    assert storms["req-retry"]["retries"] == 1
    assert storms["req-tools"]["retries"] == 1

    stalled_request = next(
        request for request in summary["requests"] if request["request_id"] == "req-stall"
    )
    assert stalled_request["aborts"] == 1
    assert stalled_request["ended_with_abort"] is True
    assert stalled_request["last_outcome"] == "stalled"


def test_stuck_retry_storm_threshold():
    summary = stuck(sample_records(), retry_storm_threshold=2)
    assert summary["retry_storms"] == []


def test_endpoint_breaks_down_failures_and_retries():
    summary = endpoint(sample_records())

    assert summary["digest"] == "endpoint"
    assert summary["totals"] == {
        "attempts": 7,
        "errors": 3,
        "successful_attempts": 4,
    }
    assert summary["error_outcomes"] == {
        "context_window_exceeded": 1,
        "http_error": 1,
        "upstream_failure": 1,
    }
    assert summary["upstream_failures"]["count"] == 1
    assert summary["upstream_failures"]["by_error_type"] == {"ConnectError": 1}
    assert summary["upstream_failures"]["by_reason"] == {
        "transient upstream error ConnectError: All connection attempts failed": 1
    }
    assert summary["http_errors"]["by_status"] == {"503": 1}
    assert summary["http_errors"]["by_reason"] == {"retryable HTTP status 503": 1}
    assert summary["context_window_exceeded"] == 1
    assert summary["context_window_reasons"] == {"context window exceeded": 1}
    assert summary["retryable"] == {"false": 1, "true": 2}
    assert summary["retry_distribution"] == {"0": 3, "1": 2}
    assert summary["requests"][0]["request_id"] in {"req-retry", "req-tools"}


def test_remediation_joins_errors_passes_and_final_summary():
    summary = remediation(sample_records())

    assert summary["digest"] == "remediation"
    assert summary["totals"] == {
        "requests": 10,
        "errors": 3,
        "passes": 5,
        "resolved": 7,
        "unresolved": 3,
    }
    assert summary["rung_counts"] == {
        "coast": 1,
        "empty_stop": 1,
        "extract": 1,
        "loop_retry": 1,
        "runaway": 1,
    }
    assert summary["cap_trips"] == {"rounds": 1, "time": 0}
    assert summary["exhausted_rungs"] == {"extract": 1}

    requests = {request["request_id"]: request for request in summary["requests"]}
    extract = requests["req-extract"]
    assert extract["status"] == "remediated"
    assert extract["chain"] == ["extract", "success"]
    assert extract["final"]["rounds"] == 1
    assert extract["final"]["rungs"] == {"extract": {"attempts": 2, "outcome": "triggered"}}
    assert extract["passes"][0]["rung"] == "extract"
    assert extract["passes"][0]["has_pre_remediation_body"] is True

    cap = requests["req-cap"]
    assert cap["final"]["remediation_exhausted"] == "rounds"
    assert cap["final"]["rungs"]["extract"]["outcome"] == "exhausted"

    retry = requests["req-retry"]
    assert retry["status"] == "unresolved"
    assert [error["outcome"] for error in retry["errors"]] == ["upstream_failure"]

    runaway = requests["req-runaway"]
    assert runaway["passes"][0]["rung"] == "runaway"
    assert runaway["passes"][0]["has_resubmitted_body"] is True


def test_narrative_reconstructs_turn_tool_calls_and_conversation():
    summary = narrative(sample_records())

    assert summary["digest"] == "narrative"
    assert summary["totals"]["requests"] == 10
    # The same tool call appears in both attempted records but is emitted once.
    assert summary["totals"]["tool_calls"] == 1
    assert summary["tool_calls"][0]["name"] == "buy"

    request = next(item for item in summary["requests"] if item["request_id"] == "req-tools")
    assert [call["name"] for call in request["tool_calls"]] == ["buy"]
    assert request["tool_calls"][0]["arguments"] == {"id": 1}
    assert request["tool_calls"][0]["result"] is None
    assert 'User asked "buy the first one"' in request["summary"]
    assert "buy" in request["summary"]
    assert request["tool_repairs"] == [{"kind": "repaired_arguments", "index": 0}]

    roles = [event["kind"] for event in request["conversation"]]
    assert roles == ["user", "assistant"]

    full = narrative(sample_records(), include_history=True)
    full_request = next(item for item in full["requests"] if item["request_id"] == "req-tools")
    assert [event["kind"] for event in full_request["conversation"]] == [
        "user",
        "assistant",
        "tool_result",
        "user",
        "assistant",
    ]


def test_narrative_follows_tool_calls_to_results_and_clips_text():
    record = {
        "id": "x",
        "timestamp": 1.0,
        "request_id": "req",
        "outcome": "success",
        "response_body": None,
        "request_body": json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call("call_1", "search", {"q": "gpu"})],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "name": "search",
                        "content": "a very long result that will be clipped",
                    },
                ]
            }
        ),
    }
    summary = narrative([record], max_text=10)
    call = summary["requests"][0]["tool_calls"][0]
    assert call["name"] == "search"
    assert call["result"] == "a very lon…"


def test_narrative_reads_sse_response_bodies():
    sse = (
        'data: {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    record = {
        "id": "x",
        "timestamp": 1.0,
        "request_id": "req",
        "outcome": "success",
        "request_body": json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        "response_body": sse,
    }
    summary = narrative([record])
    assistant = [
        event for event in summary["requests"][0]["conversation"] if event["kind"] == "assistant"
    ]
    assert assistant == [{"kind": "assistant", "content": "hello", "tool_calls": []}]


def test_narrative_includes_partial_tool_calls_from_aborts():
    records = sample_records()
    stalled = next(record for record in records if record.get("abort_kind") == "stalled")
    stalled["partial_tool_calls"] = [tool_call("partial-1", "half", {"x": 1})]

    summary = narrative(records)
    request = next(item for item in summary["requests"] if item["request_id"] == "req-stall")
    assert request["tool_calls"][0]["name"] == "half"
    assert request["tool_calls"][0]["partial"] is True


def test_run_digest_rejects_unknown_names():
    with pytest.raises(ValueError, match="unknown digest"):
        run_digest("nope", [])


def test_narrative_handles_multipart_content():
    record = {
        "id": "x",
        "timestamp": 1.0,
        "request_id": "req",
        "outcome": "success",
        "request_body": json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "part one"}, "part two"],
                    }
                ]
            }
        ),
        "response_body": None,
    }
    summary = narrative([record])
    assert summary["requests"][0]["conversation"][0]["content"] == "part onepart two"


def test_tool_conversation_without_response_body_is_still_reconstructed():
    record = tool_conversation_request("req-tools", 1.0)
    record["response_body"] = None
    record["request_body"] = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call_1", "search", {"q": "gpu"})],
                },
            ]
        }
    )
    summary = narrative([record])
    assert summary["requests"][0]["tool_calls"][0]["name"] == "search"
    assert summary["requests"][0]["tool_calls"][0]["result"] is None


def test_narrative_keeps_orphan_tool_results():
    record = {
        "id": "x",
        "timestamp": 1.0,
        "request_id": "req",
        "outcome": "success",
        "request_body": json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "go"},
                    # A result whose call was trimmed from the history.
                    {"role": "tool", "tool_call_id": "call_gone", "name": "lookup", "content": "r"},
                ]
            }
        ),
        "response_body": None,
    }
    summary = narrative([record])
    assert summary["requests"][0]["tool_calls"] == [
        {
            "id": "call_gone",
            "name": "lookup",
            "arguments": None,
            "result": "r",
        }
    ]


def test_narrative_summary_without_a_user_message():
    record = {
        "id": "x",
        "timestamp": 1.0,
        "request_id": "req",
        "outcome": "success",
        "request_body": json.dumps({"messages": [{"role": "assistant", "content": "unprompted"}]}),
        "response_body": None,
    }
    summary = narrative([record])
    assert "No user message recorded" in summary["requests"][0]["summary"]
