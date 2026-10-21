"""Full-cycle extraction integration test.

One end-to-end test drives the real ``Proxy._forward_streaming`` through the
entire extraction cycle against a mock SSE upstream that serves a distinct
spec per request, so every turn is deterministic:

1. **p1 — initial request**: a think-only turn (``finish_reason: stop``, empty
   content, no tool call, non-empty reasoning).
2. **p2 — gauze re-submits a remediation request**: the seed body carrying the
   extraction instruction and the ``gauze_read`` tool definition.
3. **p3 — the model pages through p1's reasoning**: two successive
   ``gauze_read`` tool-call turns, each answered by a read-exec that replays
   the assistant tool-call message plus the matching ``role: tool`` slice.
4. **p4 — the model responds with a decision**: a final turn with non-empty
   visible content.
5. **p5 — gauze responds**: the reconstructed chat.completion returned to the
   client.
"""

import asyncio
import json

from app.proxy import Proxy
from app.recorder import Recorder
from mock_upstream import MockUpstream, chunk, settings_override

_INSTRUCTION = "PLEASE_PAGE_AND_ANSWER"
_TOOL_NAME = "gauze_read"
_RESOURCE = "reasoning"

# The reasoning the model emits on the first turn. The answer already lives
# inside it, so it must be recovered by paging through bounded windows — never
# by re-submitting the full stream as context.
_REASONING = "\n".join(
    [
        "Let me think carefully about this question.",
        "The capital of France is Paris.",
        "I can page through my reasoning to recover the visible answer.",
        "The answer the user wants is on the second line.",
    ]
)
_DECISION = "The capital of France is Paris."

_chunk = chunk


def _think_turn():
    """p1: stop, empty content, no tool call, non-empty reasoning."""
    return [
        _chunk({"role": "assistant", "reasoning_content": _REASONING}),
        _chunk({}, finish_reason="stop"),
    ]


def _read_turn(call_id, line_from, line_count):
    """p3: a `gauze_read` tool-call turn paging one window of the resource."""
    return [
        _chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": _TOOL_NAME,
                            "arguments": json.dumps(
                                {
                                    "resource": _RESOURCE,
                                    "line_from": line_from,
                                    "line_count": line_count,
                                }
                            ),
                        },
                    }
                ],
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]


def _answer_turn():
    """p4: the final visible decision."""
    return [
        _chunk({"role": "assistant", "content": _DECISION}),
        _chunk({}, finish_reason="stop"),
    ]


def _run_full_cycle(tmp_path):
    """Drive the full cycle against a distinct-spec ``MockUpstream``."""
    record_path = tmp_path / "extract_cycle_integration.jsonl"
    specs = [
        {"chunks": _think_turn()},  # p1: think-only -> seed
        {"chunks": _read_turn("call_read_1", 1, 1)},  # p3 read 1 (after seed)
        {"chunks": _read_turn("call_read_2", 2, 1)},  # p3 read 2 (multi-window)
        {"chunks": _answer_turn()},  # p4: decision
    ]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            stall_detection_enabled=False,
            runaway_reasoning_enabled=False,
            think_cleanup_enabled=True,
            extract_enabled=True,
            extract_instruction_text=_INSTRUCTION,
            extract_max_attempts=5,
            coast_detection_enabled=False,
            tool_call_guard_enabled=False,
            loop_retry_enabled=False,
        ):
            recorder = Recorder(record_path)
            proxy = Proxy(recorder)
            body = json.dumps(
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
            ).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="extract-cycle-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "extract-cycle-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [json.loads(line) for line in open(record_path) if line.strip()]
                return resp, records

            resp, records = asyncio.run(run())
            return resp, records, mock.request_count, mock.json_bodies()


def test_full_extraction_cycle_pages_reasoning_to_decision(tmp_path):
    resp, records, count, bodies = _run_full_cycle(tmp_path)

    # p5: a 200 chat.completion whose visible content is the decision the
    # model produced on the final turn.
    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message["content"] == _DECISION, message

    # Exact upstream request count: initial (p1) + seed (p2) + two read-exec
    # re-submissions (p3). The decision (p4) is the response to the second
    # read-exec request, so it does not add a further upstream request.
    assert count == 4, count

    # The full reasoning stream must never appear in any single captured
    # request body — context stays O(window), never O(stream).
    escaped = json.dumps(_REASONING)[1:-1]
    for body in bodies:
        assert escaped not in json.dumps(body), body

    # The seed re-submission (p2) carries the extraction instruction and the
    # `gauze_read` tool definition.
    seed = bodies[1]
    assert seed["messages"][-1] == {"role": "user", "content": _INSTRUCTION}, seed
    assert seed["tools"][-1]["function"]["name"] == _TOOL_NAME, seed

    # The read-exec re-submissions (p3) each replay the assistant tool_calls
    # message followed by the matching `role: tool` result slice.
    read_bodies = [
        b for b in bodies if b.get("messages") and b["messages"][-1].get("role") == "tool"
    ]
    assert len(read_bodies) == 2, read_bodies
    for body, call_id, expected_slice in (
        (read_bodies[0], "call_read_1", "Let me think carefully about this question."),
        (read_bodies[1], "call_read_2", "The capital of France is Paris."),
    ):
        assert body["messages"][-2]["role"] == "assistant", body
        assert body["messages"][-2].get("content") is None, body
        assert body["messages"][-2]["tool_calls"][0]["function"]["name"] == _TOOL_NAME, body
        assert body["messages"][-1] == {
            "role": "tool",
            "tool_call_id": call_id,
            "content": expected_slice,
        }, body

    # The final record's extract entry reports the full cycle succeeded with
    # seed + two reads = three remediation passes.
    final = records[-1]
    assert final["extract"] == {"attempts": 3, "outcome": "succeeded"}, final
