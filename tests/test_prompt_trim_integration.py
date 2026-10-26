"""Rule-driven prompt trimming (#153) — proxy integration.

These tests drive the real ``Proxy`` against a mock upstream to prove the
trimmed body is forwarded, ordering against the overflow guard holds, the
record and telemetry carry the change, an invalid rule file disables trimming
without breaking requests, and the client still receives a valid completion.

All prompt text here is synthetic — never recorded conversation content.
"""

import asyncio
import json
import logging

from app.proxy import Proxy
from app.recorder import Recorder
from app.telemetry import telemetry
from conftest import write_rules
from mock_upstream import MockUpstream, chunk, make_request, settings_override

_OK_BODY = json.dumps(
    {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
    }
).encode()

_QUIET = {
    "loop_detection_enabled": False,
    "stall_detection_enabled": False,
    "runaway_reasoning_enabled": False,
    "extract_enabled": False,
    "coast_detection_enabled": False,
    "think_cleanup_enabled": False,
    "tool_call_guard_enabled": False,
}


def _raw_body(payload):
    """Serialise with non-default pretty whitespace.

    A re-serialisation regression (``json.dumps`` on a reparsed body) produces
    different bytes, so byte-identity assertions built on this are meaningful.
    """
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _run_forward(tmp_path, body, *, read_records, rules_path=None, extra_settings=None):
    settings_values = {
        **_QUIET,
        "llm_base_url": "",
        "prompt_trim_enabled": rules_path is not None,
        "prompt_trim_rules_file": str(rules_path or ""),
        "message_overflow_enabled": False,
    }
    settings_values.update(extra_settings or {})
    with MockUpstream(
        [{"status": 200, "content_type": "application/json", "body": _OK_BODY}]
    ) as mock:
        settings_values["llm_base_url"] = mock.url
        with settings_override(**settings_values):
            recorder = Recorder(tmp_path / "prompt_trim.jsonl")
            proxy = Proxy(recorder)

            async def run():
                request = await make_request(body)
                return await proxy.forward(request, "v1/chat/completions")

            response = asyncio.run(run())
            return response, read_records(tmp_path / "prompt_trim.jsonl"), mock.requests[0]


def _counter_delta(before, after, name):
    before_value = before["counters"].get(name, {})
    after_value = after["counters"].get(name, {})
    if isinstance(before_value, int) or isinstance(after_value, int):
        return after_value - before_value
    return {
        key: after_value.get(key, 0) - before_value.get(key, 0)
        for key in set(before_value) | set(after_value)
    }


def test_integration_forwards_trimmed_body_records_it_and_counts_it(tmp_path, read_records):
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "trim-system",
                "when": {"role": "system", "content_regex": "SECRET"},
                "then": {"action": "sub", "replacement": "PUBLIC"},
            }
        ],
    )
    body = json.dumps(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "SECRET preamble"},
                {"role": "user", "content": "task"},
            ],
        }
    ).encode()
    before = telemetry.snapshot()

    response, records, captured = _run_forward(
        tmp_path, body, read_records=read_records, rules_path=rules_path
    )

    after = telemetry.snapshot()
    assert response.status_code == 200
    assert json.loads(response.body)["choices"][0]["message"]["content"] == "ok"
    assert json.loads(captured)["messages"][0]["content"] == "PUBLIC preamble"
    final = records[-1]
    assert final["prompt_trim"] == [
        {
            "kind": "prompt_trim",
            "rule": "trim-system",
            "index": 0,
            "role": "system",
            "action": "sub",
            "chars_before": len("SECRET preamble"),
            "chars_after": len("PUBLIC preamble"),
        }
    ]
    assert json.loads(final["request_body"]) == json.loads(captured)
    delta = _counter_delta(before, after, "prompt_trim_rules_total")
    assert delta['action="sub",mode="apply",rule="trim-system"'] == 1


def test_integration_dry_run_forwards_original_bytes_and_records_changes(tmp_path, read_records):
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "trim-system",
                "when": {"role": "system", "content_regex": "SECRET"},
                "then": {"action": "replace", "content": "PUBLIC"},
            }
        ],
    )
    body = _raw_body(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "SECRET preamble"},
                {"role": "user", "content": "task"},
            ],
        }
    )
    before = telemetry.snapshot()

    response, records, captured = _run_forward(
        tmp_path,
        body,
        read_records=read_records,
        rules_path=rules_path,
        extra_settings={"prompt_trim_dry_run": True},
    )

    after = telemetry.snapshot()
    assert response.status_code == 200
    assert captured == body  # byte-for-byte same request body
    assert records[-1]["prompt_trim"][0]["action"] == "replace"
    assert records[-1]["request_body"] == body.decode("utf-8")
    delta = _counter_delta(before, after, "prompt_trim_rules_total")
    assert delta['action="replace",mode="dry_run",rule="trim-system"'] == 1


def test_integration_disabled_and_no_match_forward_identical_bytes(tmp_path, read_records):
    body = _raw_body(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "preamble"},
                {"role": "user", "content": "task"},
            ],
        }
    )

    before = telemetry.snapshot()
    _, records, captured = _run_forward(tmp_path, body, read_records=read_records)
    after = telemetry.snapshot()
    assert captured == body  # byte-for-byte same request body
    assert records[-1]["prompt_trim"] is None
    assert _counter_delta(before, after, "prompt_trim_skipped_total")['reason="disabled"'] == 1

    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "no-match",
                "when": {"role": "system", "content_regex": "zzz"},
                "then": {"action": "replace", "content": "y"},
            }
        ],
    )
    _, records, captured = _run_forward(
        tmp_path, body, read_records=read_records, rules_path=rules_path
    )
    assert captured == body  # byte-for-byte same request body
    assert records[-1]["prompt_trim"] is None


def test_integration_records_both_prompt_trim_and_overflow(tmp_path, read_records):
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "trim-system",
                "when": {"role": "system", "content_regex": "SECRET"},
                "then": {"action": "replace", "content": "PUBLIC"},
            }
        ],
    )
    body = json.dumps(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "SECRET"},
                {"role": "user", "content": "task"},
                {"role": "tool", "tool_call_id": "c1", "content": "x" * 100},
            ],
        }
    ).encode()

    _, records, captured = _run_forward(
        tmp_path,
        body,
        read_records=read_records,
        rules_path=rules_path,
        extra_settings={
            "message_overflow_enabled": True,
            "message_overflow_threshold": 10,
            "message_overflow_truncate": True,
            "message_overflow_warning": "WARNING",
        },
    )

    final = records[-1]
    assert json.loads(captured)["messages"][0]["content"] == "PUBLIC"
    assert json.loads(captured)["messages"][2]["content"] == "WARNING\n" + "x" * 10 + "..."
    assert final["prompt_trim"][0]["rule"] == "trim-system"
    assert final["message_overflow"][0]["index"] == 2


def test_prompt_trimming_runs_before_the_overflow_guard(tmp_path, read_records):
    # The rule shortens the tool result below the overflow threshold. If the
    # overflow guard ran first, it would warn/truncate it and record a change.
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "shorten-tool",
                "when": {"role": "tool", "content_regex": "^abcdefghij$"},
                "then": {"action": "truncate", "chars": 5},
            }
        ],
    )
    body = json.dumps(
        {
            "model": "test",
            "messages": [
                {"role": "user", "content": "task"},
                {"role": "tool", "tool_call_id": "c1", "content": "abcdefghij"},
            ],
        }
    ).encode()

    _, records, captured = _run_forward(
        tmp_path,
        body,
        read_records=read_records,
        rules_path=rules_path,
        extra_settings={
            "prompt_trim_truncate_marker": "..",
            "message_overflow_enabled": True,
            "message_overflow_threshold": 10,
            "message_overflow_warning": "WARNING",
        },
    )

    assert json.loads(captured)["messages"][1]["content"] == "abc.."
    assert records[-1]["prompt_trim"][0]["action"] == "truncate"
    assert records[-1]["message_overflow"] is None


def test_integration_streaming_path_forwards_trimmed_body(tmp_path, read_records):
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "trim-system",
                "when": {"role": "system", "content_regex": "SECRET"},
                "then": {"action": "replace", "content": "PUBLIC"},
            }
        ],
    )
    body = json.dumps(
        {
            "model": "test",
            "stream": True,
            "messages": [
                {"role": "system", "content": "SECRET preamble"},
                {"role": "user", "content": "task"},
            ],
        }
    ).encode()
    specs = [
        {
            "chunks": [
                chunk({"role": "assistant", "content": "ok"}),
                chunk({}, finish_reason="stop"),
            ]
        }
    ]
    before = telemetry.snapshot()

    with MockUpstream(specs) as mock:
        with settings_override(
            **{**_QUIET, "loop_detection_enabled": True},
            llm_base_url=mock.url,
            prompt_trim_enabled=True,
            prompt_trim_rules_file=str(rules_path),
            message_overflow_enabled=False,
        ):
            recorder = Recorder(tmp_path / "prompt_trim_stream.jsonl")
            proxy = Proxy(recorder)

            async def run():
                request = await make_request(body)
                return await proxy.forward(request, "v1/chat/completions")

            response = asyncio.run(run())
            captured = json.loads(mock.requests[0])
            records = read_records(tmp_path / "prompt_trim_stream.jsonl")

    after = telemetry.snapshot()
    assert response.status_code == 200
    assert captured["messages"][0]["content"] == "PUBLIC"
    assert b"ok" in response.body
    assert records[-1]["prompt_trim"][0]["rule"] == "trim-system"
    delta = _counter_delta(before, after, "prompt_trim_rules_total")
    assert delta['action="replace",mode="apply",rule="trim-system"'] == 1


def test_invalid_file_logs_once_disables_trimming_and_still_serves(tmp_path, caplog, read_records):
    path = write_rules(
        tmp_path,
        [
            {
                "id": "rule",
                "when": {"role": "system", "content_regex": "foo"},
                "then": {"action": "replace", "content": "foo"},
            }
        ],
    )
    body = _raw_body({"model": "test", "messages": [{"role": "system", "content": "foo"}]})

    with MockUpstream(
        [{"status": 200, "content_type": "application/json", "body": _OK_BODY}]
    ) as mock:
        with settings_override(
            **_QUIET,
            llm_base_url=mock.url,
            prompt_trim_enabled=True,
            prompt_trim_rules_file=str(path),
            message_overflow_enabled=False,
        ):
            with caplog.at_level(logging.ERROR, logger="llm_gauze.prompt_trim"):
                recorder = Recorder(tmp_path / "invalid.jsonl")
                proxy = Proxy(recorder)

                async def run():
                    request = await make_request(body)
                    return await proxy.forward(request, "v1/chat/completions")

                response = asyncio.run(run())
            records = read_records(tmp_path / "invalid.jsonl")

    assert response.status_code == 200
    assert mock.requests[0] == body  # byte-for-byte same request body
    assert records[-1]["prompt_trim"] is None
    assert caplog.text.count("invalid prompt-trim rule file") == 1


def test_integration_scan_bound_counts_a_skip(tmp_path, read_records):
    rules_path = write_rules(
        tmp_path,
        [
            {
                "id": "never-matches",
                "when": {"role": "system", "content_regex": "^x+$"},
                "then": {"action": "replace", "content": "y"},
            }
        ],
    )
    body = _raw_body({"model": "test", "messages": [{"role": "system", "content": "x" * 20}]})
    before = telemetry.snapshot()

    _, records, captured = _run_forward(
        tmp_path,
        body,
        read_records=read_records,
        rules_path=rules_path,
        extra_settings={"prompt_trim_max_scan_chars": 5},
    )
    after = telemetry.snapshot()

    assert captured == body  # byte-for-byte same request body
    assert records[-1]["prompt_trim"] is None
    assert _counter_delta(before, after, "prompt_trim_skipped_total")['reason="scan_bound"'] == 1
