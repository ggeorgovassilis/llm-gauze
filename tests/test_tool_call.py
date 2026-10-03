"""Tool-call syntax enforcement — unit + integration.

Unit tests cover `ToolCallGuard.validate` (valid pass-through, deterministic
repair of truncated ``function.arguments``, and flagging of unfixable/mangled
calls). Integration tests drive the real `Proxy._forward_streaming` against a
mock SSE upstream that emits a tool call with truncated arguments, proving the
gateway repairs it (and does not crash on unfixable input).

    docker compose exec -T gateway python - < tests/test_tool_call.py
"""

import asyncio
import json
import traceback

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.tool_call import ToolCallGuard
from mock_upstream import MockUpstream, chunk, settings_override


def _guard():
    return ToolCallGuard()


def _call(arguments=None, name="get_weather", type_="function"):
    fn = {"name": name}
    if arguments is not None:
        fn["arguments"] = arguments
    call = {"id": "call_1", "type": type_, "function": fn}
    return call


# --- unit tests -------------------------------------------------------


def test_valid_arguments_untouched():
    changes = _guard().validate([_call('{"city": "Paris"}')])
    assert changes == [], changes


def test_empty_arguments_repaired_to_object():
    calls = [_call("")]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert calls[0]["function"]["arguments"] == "{}", calls


def test_missing_arguments_repaired_to_object():
    calls = [_call(None)]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert calls[0]["function"]["arguments"] == "{}", calls


def test_truncated_string_repaired():
    # Cut off mid-string: the closing quote and brace can be restored.
    calls = [_call('{"city": "London')]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "London"}


def test_trailing_comma_repaired():
    calls = [_call('{"a": 1,')]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1}


def test_array_trailing_comma_repaired():
    calls = [_call("[1, 2,")]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert json.loads(calls[0]["function"]["arguments"]) == [1, 2]


def test_unbalanced_brackets_flagged():
    calls = [_call('{"city": "London"}}')]
    changes = _guard().validate(calls)
    assert changes == [
        {"kind": "flagged_malformed", "index": 0, "reason": "unfixable_arguments"}
    ], changes
    # Left as-is, not crashed, not silently mutated.
    assert calls[0]["function"]["arguments"] == '{"city": "London"}}', calls


def test_missing_function_flagged():
    calls = [{"id": "call_1", "type": "function"}]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "flagged_malformed", "index": 0, "reason": "missing_function"}], (
        changes
    )


def test_missing_name_flagged():
    calls = [_call("{}", name="")]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "flagged_malformed", "index": 0, "reason": "missing_name"}], changes


def test_arguments_not_string_flagged():
    calls = [_call({"city": "Paris"})]
    changes = _guard().validate(calls)
    assert changes == [
        {"kind": "flagged_malformed", "index": 0, "reason": "arguments_not_string"}
    ], changes


def test_repairs_second_call_and_flags_first():
    calls = [_call("not json at all"), _call('{"b": 2')]
    changes = _guard().validate(calls)
    assert changes == [
        {"kind": "flagged_malformed", "index": 0, "reason": "unfixable_arguments"},
        {"kind": "repaired_arguments", "index": 1},
    ], changes
    assert json.loads(calls[1]["function"]["arguments"]) == {"b": 2}


def test_escaped_quote_repaired():
    calls = [_call('{"msg": "he said \\"hi')]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert json.loads(calls[0]["function"]["arguments"]) == {"msg": 'he said "hi'}, calls


def test_dangling_escape_flagged():
    calls = [_call('{"a": "foo\\')]
    changes = _guard().validate(calls)
    assert changes == [
        {"kind": "flagged_malformed", "index": 0, "reason": "unfixable_arguments"}
    ], changes


def test_nested_array_truncation_repaired():
    calls = [_call("[[1,2]")]
    changes = _guard().validate(calls)
    assert changes == [{"kind": "repaired_arguments", "index": 0}], changes
    assert json.loads(calls[0]["function"]["arguments"]) == [[1, 2]], calls


def test_mismatched_closing_bracket_flagged():
    calls = [_call('{"a": 1]')]
    changes = _guard().validate(calls)
    assert changes == [
        {"kind": "flagged_malformed", "index": 0, "reason": "unfixable_arguments"}
    ], changes


def test_non_object_call_flagged():
    changes = _guard().validate(["not a dict"])
    assert changes == [{"kind": "flagged_malformed", "index": 0, "reason": "not_an_object"}], (
        changes
    )


def test_drop_trailing_comma_skips_whitespace():
    from app.remediation.tool_call import _drop_trailing_comma

    out = list('{"a": 1,  ')
    _drop_trailing_comma(out)
    # The comma is dropped after skipping the whitespace that followed it.
    assert "".join(out) == '{"a": 1' + "  "


# --- integration harness ---------------------------------------------


_chunk = chunk


def _run_forward(chunks):
    with MockUpstream([{"chunks": chunks}]) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            tool_call_guard_enabled=True,
        ):
            recorder = Recorder("/tmp/tool_call_integration.jsonl")
            proxy = Proxy(recorder)
            req = {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
            }
            body = json.dumps(req).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="toolcall-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "toolcall-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [
                    json.loads(line)
                    for line in open("/tmp/tool_call_integration.jsonl")
                    if line.strip()
                ]
                return resp, records

            return asyncio.run(run())


def test_truncated_tool_call_is_repaired_end_to_end():
    # A tool call whose arguments were cut off mid-string across two chunks.
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London',
                        },
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp, records = _run_forward(chunks)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    message = data["choices"][0]["message"]
    call = message["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"city": "London"}, call

    # The repair is recorded, never silent.
    final = records[-1]
    assert final["tool_repair"] == [{"kind": "repaired_arguments", "index": 0}], final


def test_unfixable_tool_call_is_flagged_not_crashed():
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}}',
                        },
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp, records = _run_forward(chunks)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    data = json.loads(resp.body)
    call = data["choices"][0]["message"]["tool_calls"][0]
    # Passed through unchanged, but the malformation is on the record.
    assert call["function"]["arguments"] == '{"city": "London"}}', call
    final = records[-1]
    assert final["tool_repair"] == [
        {"kind": "flagged_malformed", "index": 0, "reason": "unfixable_arguments"}
    ], final


def test_valid_tool_call_recorded_without_repair():
    chunks = [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    resp, records = _run_forward(chunks)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert records[-1].get("tool_repair") is None, records[-1]


def _run_all() -> int:
    tests = [
        value
        for key, value in sorted(globals().items())
        if key.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
