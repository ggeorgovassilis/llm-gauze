"""Common remediation protocol + composable ladder — unit + composition.

These tests pin the new contract introduced by #88:

* every re-submission policy implements the :class:`Remediation` protocol;
* every content/body guard implements the :class:`Transform` protocol;
* the ladder is an ordered, composable list of steps driven uniformly;
* two rungs (extract, then coast) compose on one request, in ladder order.
"""

import asyncio
import json

from app.proxy import Proxy, _find_step, _first_applicable_step, _remediation_outcome
from app.recorder import Recorder
from app.remediation.base import Remediation, Transform, Turn
from app.remediation.coast import CoastPolicy
from app.remediation.extract import ExtractionPolicy
from app.remediation.loop_retry import LoopRetryPolicy
from app.remediation.overflow import MessageOverflowGuard
from app.remediation.runaway import RunawayReasoningPolicy
from app.remediation.think import ThinkContentGuard
from app.remediation.tool_call import ToolCallGuard
from mock_upstream import MockUpstream, chunk, settings_override

_INSTRUCTION = "PLEASE_PAGE_AND_ANSWER"
_COAST_TEXT = "PLEASE_MAKE_THE_TOOL_CALL"
_COASTED_CONTENT = "Chunk 5 done: 12 products, 0 picks. Pulling next chunk."

_chunk = chunk


# --- protocol conformance ---------------------------------------------


def test_policies_implement_remediation_protocol():
    for cls, inst in (
        (ExtractionPolicy, ExtractionPolicy.from_settings()),
        (CoastPolicy, CoastPolicy.from_settings()),
        (RunawayReasoningPolicy, RunawayReasoningPolicy.from_settings()),
        (LoopRetryPolicy, LoopRetryPolicy.from_settings()),
    ):
        assert issubclass(cls, Remediation), cls
        assert isinstance(inst.name, str) and inst.name, cls
        assert isinstance(inst.max_attempts, int) and inst.max_attempts >= 1, cls
        assert callable(inst.applies), cls
        assert callable(inst.apply), cls


def test_transforms_implement_transform_protocol():
    for cls, inst in (
        (ThinkContentGuard, ThinkContentGuard.from_settings()),
        (ToolCallGuard, ToolCallGuard.from_settings()),
        (MessageOverflowGuard, MessageOverflowGuard.from_settings()),
    ):
        assert issubclass(cls, Transform), cls
        assert callable(inst.apply), cls


def test_remediation_names_are_stable():
    # The recorder keys and attempt counters hang off these names.
    assert ExtractionPolicy.from_settings().name == "extract"
    assert CoastPolicy.from_settings().name == "coast"
    assert RunawayReasoningPolicy.from_settings().name == "runaway"
    assert LoopRetryPolicy.from_settings().name == "loop_retry"


# --- ladder composition (helpers) -------------------------------------


def test_ladder_ordering():
    proxy = Proxy(Recorder("/tmp/test_remediation_order.jsonl"))
    with settings_override(
        runaway_reasoning_enabled=True,
        loop_retry_enabled=True,
        extract_enabled=True,
        coast_detection_enabled=True,
    ):
        ladder = proxy._build_remediation_ladder()
    assert [step.name for step in ladder] == ["runaway", "loop_retry", "extract", "coast"]


def test_ladder_omits_disabled_rungs():
    proxy = Proxy(Recorder("/tmp/test_remediation_disabled.jsonl"))
    with settings_override(
        runaway_reasoning_enabled=False,
        loop_retry_enabled=False,
        extract_enabled=False,
        coast_detection_enabled=False,
    ):
        assert proxy._build_remediation_ladder() == []


def test_first_applicable_step_picks_first_match_in_order():
    ladder = [
        ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=2),
        CoastPolicy(text=_COAST_TEXT, max_attempts=2),
    ]
    attempts = {"extract": 0, "coast": 0}
    turn = Turn(finish_reason="stop", reasoning="thinking hard")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    # Only the extract rung matches an empty turn; coast needs content + tools.
    step = _first_applicable_step(ladder, attempts, turn, body)
    assert step is not None and step.name == "extract"


def test_first_applicable_step_skips_exhausted_rung():
    ladder = [ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=1)]
    attempts = {"extract": 1}
    turn = Turn(finish_reason="stop", reasoning="thinking hard")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert _first_applicable_step(ladder, attempts, turn, body) is None


def test_find_step_returns_rung_by_name():
    ladder = [
        ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=2),
        CoastPolicy(text=_COAST_TEXT, max_attempts=2),
    ]
    assert _find_step(ladder, "coast") is not None
    assert _find_step(ladder, "loop_retry") is None


def test_remediation_outcome_exhausted():
    step = ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=1)
    turn = Turn(finish_reason="stop", reasoning="still thinking")
    body = {"messages": []}
    assert _remediation_outcome(step, {"extract": 1}, turn, body) == "exhausted"


def test_remediation_outcome_succeeded():
    step = ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=1)
    turn = Turn(finish_reason="stop", content="the answer")
    body = {"messages": []}
    assert _remediation_outcome(step, {"extract": 1}, turn, body) == "succeeded"


def test_remediation_outcome_untouched_is_none():
    step = ExtractionPolicy(instruction=_INSTRUCTION, max_attempts=1)
    turn = Turn(finish_reason="stop", content="the answer")
    body = {"messages": []}
    assert _remediation_outcome(step, {"extract": 0}, turn, body) is None


# --- composition integration ------------------------------------------


def _loop_request():
    return {
        "model": "test",
        "messages": [
            {"role": "user", "content": "vet the products"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "next_chunk", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "done"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "next_chunk", "description": "next chunk"},
            }
        ],
    }


_EMPTY_TURN = [
    _chunk({"role": "assistant", "reasoning_content": "thinking hard"}),
    _chunk({}, finish_reason="stop"),
]
_COAST_TURN = [
    _chunk(
        {
            "role": "assistant",
            "content": _COASTED_CONTENT,
            "reasoning_content": _COASTED_CONTENT,
        }
    ),
    _chunk({}, finish_reason="stop"),
]
_TOOL_CALL_TURN = [
    _chunk(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_next",
                    "type": "function",
                    "function": {"name": "next_chunk", "arguments": "{}"},
                }
            ],
        }
    ),
    _chunk({}, finish_reason="tool_calls"),
]


def test_extract_and_coast_compose_on_one_request():
    specs = [{"chunks": turn} for turn in [_EMPTY_TURN, _COAST_TURN, _TOOL_CALL_TURN]]
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            loop_detection_enabled=True,
            think_cleanup_enabled=True,
            extract_enabled=True,
            extract_instruction=_INSTRUCTION,
            extract_max_attempts=2,
            coast_detection_enabled=True,
            coast_nudge_text=_COAST_TEXT,
            coast_max_attempts=2,
        ):
            recorder = Recorder("/tmp/test_remediation_compose.jsonl")
            proxy = Proxy(recorder)
            body = json.dumps(_loop_request()).encode()

            async def run():
                resp = await proxy._forward_streaming(
                    request_id="compose-itest",
                    method="POST",
                    url="/v1/chat/completions",
                    query="",
                    body=body,
                    headers={"content-type": "application/json"},
                    base_entry={
                        "request_id": "compose-itest",
                        "method": "POST",
                        "path": "/v1/chat/completions",
                    },
                )
                records = [
                    json.loads(line)
                    for line in open("/tmp/test_remediation_compose.jsonl")
                    if line.strip()
                ]
                return resp, records

            resp, records = asyncio.run(run())

    assert resp.status_code == 200, (resp.status_code, resp.body)
    message = json.loads(resp.body)["choices"][0]["message"]
    assert message.get("tool_calls"), message
    assert message["tool_calls"][0]["function"]["name"] == "next_chunk", message

    # Three upstream requests: empty (extracted), coasted (re-prompted), answer.
    assert mock.request_count == 3, mock.request_count

    # Both rungs succeeded, in ladder order (extract first, then coast).
    final = records[-1]
    assert final["extract"] == {"attempts": 1, "outcome": "succeeded"}, final
    assert final["coast"] == {"attempts": 1, "outcome": "succeeded"}, final

    # Each rung recorded exactly one triggered pass.
    extract_triggers = [r for r in records if r.get("extract", {}).get("outcome") == "triggered"]
    coast_triggers = [r for r in records if r.get("coast", {}).get("outcome") == "triggered"]
    assert len(extract_triggers) == 1, records
    assert len(coast_triggers) == 1, records
