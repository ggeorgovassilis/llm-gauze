"""Message-overflow protection (oversized tool results) — unit + integration.

Unit tests cover `MessageOverflowGuard.process` (deterministic trigger, warn +
truncate, warn-only, non-tool messages untouched, non-string content, threshold
boundary, input immutability). Integration tests drive the real `Proxy.forward`
through both the buffered path (loop detection off) and the streaming path
(loop detection on), asserting the mock upstream received the warned/truncated
body and the recorder captured the intervention — plus a disabled-switch no-op.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.overflow import MessageOverflowGuard
from mock_upstream import MockUpstream, make_request, settings_override

_WARNING = "PLEASE_WORK_AROUND_THIS"


def _chunk(delta: dict, finish_reason=None) -> dict:
    return {
        "id": "chatcmpl-overflow",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


# --- unit tests -------------------------------------------------------


def _tool_messages(content):
    return {
        "model": "test",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "c1", "content": content},
        ],
    }


def test_oversized_tool_result_truncated_and_warned():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    content = "first line\n" + "x" * 1000
    out, changes = guard.process(_tool_messages(content))
    assert len(changes) == 1, changes
    assert changes[0] == {
        "kind": "tool_overflow",
        "index": 1,
        "size": len(content),
        "truncated": True,
    }, changes
    rewritten = out["messages"][1]["content"]
    assert rewritten == f"{_WARNING}\nfirst line", rewritten
    # Input is not mutated.
    assert "x" * 1000 in _tool_messages(content)["messages"][1]["content"]


def test_warn_only_keeps_full_content():
    guard = MessageOverflowGuard(threshold=10, truncate=False, warning=_WARNING)
    content = "first line\n" + "x" * 1000
    out, changes = guard.process(_tool_messages(content))
    assert len(changes) == 1, changes
    assert changes[0]["truncated"] is False
    assert out["messages"][1]["content"] == f"{_WARNING}\n{content}", out


def test_single_line_blob_capped_at_threshold():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    content = "y" * 5000  # no newline -> minified-style blob
    out, _ = guard.process(_tool_messages(content))
    rewritten = out["messages"][1]["content"]
    assert rewritten == f"{_WARNING}\n{'y' * 10}...", rewritten


def test_threshold_boundary_not_flagged():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    out, changes = guard.process(_tool_messages("a" * 10))
    assert changes == [], changes
    assert out["messages"][1]["content"] == "a" * 10


def test_non_tool_messages_untouched():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [
            {"role": "system", "content": "big" * 100},
            {"role": "user", "content": "huge" * 100},
            {"role": "assistant", "content": "vast" * 100},
        ],
    }
    out, changes = guard.process(body)
    assert changes == [], changes
    assert out is body, out


def test_non_string_tool_content_skipped():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [{"role": "tool", "tool_call_id": "c1", "content": ["list"]}],
    }
    out, changes = guard.process(body)
    assert changes == [], changes
    assert out is body, out


def test_multiple_tool_messages_flag_only_oversized():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [
            {"role": "tool", "tool_call_id": "c1", "content": "small"},
            {"role": "tool", "tool_call_id": "c2", "content": "b" * 100},
            {"role": "tool", "tool_call_id": "c3", "content": "c" * 100},
        ],
    }
    out, changes = guard.process(body)
    assert [c["index"] for c in changes] == [1, 2], changes
    assert out["messages"][0]["content"] == "small"
    assert out["messages"][1]["content"].startswith(_WARNING)
    assert out["messages"][2]["content"].startswith(_WARNING)


def test_non_dict_body_untouched():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    out, changes = guard.process("not a dict")
    assert changes == [], changes
    assert out == "not a dict"


def test_non_list_messages_untouched():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {"model": "test", "messages": "not a list"}
    out, changes = guard.process(body)
    assert changes == [], changes
    assert out is body


def test_non_dict_message_skipped():
    guard = MessageOverflowGuard(threshold=10, truncate=True, warning=_WARNING)
    body = {
        "model": "test",
        "messages": [
            "not a dict",
            {"role": "tool", "tool_call_id": "c1", "content": "b" * 100},
        ],
    }
    out, changes = guard.process(body)
    assert [c["index"] for c in changes] == [1], changes
    assert out["messages"][0] == "not a dict"
    assert out["messages"][1]["content"].startswith(_WARNING)


# --- integration harness ---------------------------------------------


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


def _overflow_mock(loop_enabled):
    if loop_enabled:
        specs = [
            {
                "chunks": [
                    _chunk({"role": "assistant", "content": "ok"}),
                    _chunk({}, finish_reason="stop"),
                ]
            }
        ]
    else:
        specs = [{"status": 200, "content_type": "application/json", "body": _OK_BODY}]
    return MockUpstream(specs)


def _run_forward(loop_enabled, content, enabled=True):
    with _overflow_mock(loop_enabled) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=loop_enabled,
            stall_detection_enabled=False,
            runaway_reasoning_enabled=False,
            think_cleanup_enabled=False,
            extract_enabled=False,
            coast_detection_enabled=False,
            tool_call_guard_enabled=False,
            message_overflow_enabled=enabled,
            message_overflow_threshold=10,
            message_overflow_truncate=True,
            message_overflow_warning=_WARNING,
        ):
            recorder = Recorder("/tmp/overflow_integration.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(
                {
                    "model": "test",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "tool",
                            "tool_call_id": "c1",
                            "content": content,
                        },
                    ],
                }
            ).encode()

            async def run():
                request = await make_request(body)
                return await proxy.forward(request, "v1/chat/completions")

            resp = asyncio.run(run())
            records = [
                json.loads(line) for line in open("/tmp/overflow_integration.jsonl") if line.strip()
            ]
            captured = mock.json_bodies()[0]
            return resp, records, captured


def test_buffered_forward_truncates_before_upstream():
    resp, records, captured = _run_forward(False, "first line\n" + "x" * 1000)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    assert "stream" not in captured
    tool_content = captured["messages"][1]["content"]
    assert tool_content == f"{_WARNING}\nfirst line", tool_content
    final = records[-1]
    assert final["message_overflow"][0]["index"] == 1, final
    assert final["message_overflow"][0]["size"] == len("first line\n" + "x" * 1000)


def test_streaming_forward_truncates_before_upstream():
    resp, records, captured = _run_forward(True, "first line\n" + "x" * 1000)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    tool_content = captured["messages"][1]["content"]
    assert tool_content == f"{_WARNING}\nfirst line", tool_content
    assert records[-1]["message_overflow"][0]["index"] == 1, records[-1]


def test_disabled_is_noop():
    resp, records, captured = _run_forward(False, "first line\n" + "x" * 1000, enabled=False)
    assert resp.status_code == 200, (resp.status_code, resp.body)
    # The upstream received the full, unmodified tool result.
    assert captured["messages"][1]["content"] == "first line\n" + "x" * 1000
    assert records[-1].get("message_overflow") is None, records[-1]
