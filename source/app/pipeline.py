"""The streaming pipeline runner.

Owns the streaming orchestration that used to live inline in ``proxy.py``: it
opens the upstream SSE stream, reads and decodes each ``data:`` frame, merges
metadata and tool-call fragments, drives the content watchdogs through the
uniform :class:`ContentWatchdog` contract, bounds each read by the stall budget,
and assembles the raw turn (content, reasoning, tool calls, metadata, SSE
lines) plus any verdict the watchdogs raised.

Transport/HTTP retry, recording, telemetry, and remediation finalisation all
stay with the proxy; this module is only about *one* streamed exchange.
"""

import asyncio
import json
from dataclasses import dataclass, field

import httpx

from app.config import settings
from app.remediation.base import StreamVerdict, VerdictKind
from app.remediation.loop import ThinkingLoopDetector
from app.remediation.runaway import RunawayReasoningDetector
from app.remediation.stall import StallDetector


@dataclass
class StreamOutcome:
    """Raw result of one streamed exchange with the upstream.

    ``verdict`` is set when a content watchdog tripped (loop/stall/runaway);
    otherwise the text fields carry the assembled turn for the proxy to
    finalise. ``error_body`` is populated when the upstream answered with an
    HTTP error status.
    """

    status: int | None = None
    resp_headers: dict = field(default_factory=dict)
    error_body: bytes = b""
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    sse_lines: list = field(default_factory=list)
    verdict: StreamVerdict | None = None
    verdict_stream: str | None = None


def _extract_stream_chunk(
    chunk: dict,
) -> tuple[str | None, str | None, list | None, dict]:
    """Pull (content, reasoning, tool_calls, meta) from one SSE payload.

    Tolerates the common OpenAI-compatible shapes: ``delta`` vs ``message``,
    ``reasoning_content`` / ``reasoning`` / ``thinking`` for the thinking
    stream, and streaming ``tool_calls`` fragments for tool use.
    """
    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or choice.get("message") or {}
    content = delta.get("content")
    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking")
    tool_calls = delta.get("tool_calls")
    meta = {
        "id": chunk.get("id"),
        "model": chunk.get("model"),
        "created": chunk.get("created"),
        "role": delta.get("role"),
        "finish_reason": choice.get("finish_reason"),
        "usage": chunk.get("usage"),
    }
    return content, reasoning, tool_calls, meta


def _merge_meta(meta: dict, chunk_meta: dict) -> None:
    """Merge non-None metadata fields into the accumulator (last wins)."""
    for key, value in chunk_meta.items():
        if value is not None:
            meta[key] = value


def _merge_tool_calls(accumulator: list, deltas: list | None) -> None:
    """Accumulate OpenAI streaming tool-call fragments by ``index``.

    The first fragment carries ``id`` / ``type`` / ``function.name``; later
    fragments append to ``function.arguments``. Fragments for the same index
    are merged into one entry.
    """
    for fragment in deltas or []:
        if not isinstance(fragment, dict):
            continue
        index = fragment.get("index", 0)
        while len(accumulator) <= index:
            accumulator.append({})
        slot = accumulator[index]
        for key, value in fragment.items():
            if key == "index":
                continue
            if key == "function" and isinstance(value, dict):
                fn = slot.setdefault("function", {})
                for fkey, fval in value.items():
                    if fkey == "arguments":
                        if fval is not None:
                            fn["arguments"] = fn.get("arguments", "") + fval
                    elif fval is not None:
                        fn[fkey] = fval
            elif value is not None:
                slot[key] = value


class StreamingPipeline:
    """Runs one streamed exchange, driving the content watchdogs uniformly.

    Constructed once per ``Proxy`` (it holds the shared ``httpx`` client) and
    called per attempt. Stateful watchdogs are created fresh inside ``run`` so
    no state leaks between attempts or requests.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def run(
        self,
        method: str,
        url: str,
        params: str,
        stream_body: bytes,
        headers: dict,
    ) -> StreamOutcome:
        """Open the upstream stream and read one exchange into an outcome.

        Transport errors propagate to the caller (the proxy classifies them for
        retry); this method is only concerned with reading a healthy stream.
        """
        outcome = StreamOutcome()
        async with self.client.stream(
            method,
            url,
            params=params,
            content=stream_body,
            headers=headers,
        ) as upstream:
            outcome.status = upstream.status_code
            outcome.resp_headers = dict(upstream.headers)
            if outcome.status >= 400:
                outcome.error_body = await upstream.aread()
                return outcome
            await self._read_stream(upstream, outcome)
        return outcome

    async def _read_stream(self, upstream, outcome: StreamOutcome) -> None:
        """Read SSE frames, feed the watchdogs, and assemble the raw turn."""
        # One detector per concern. The loop detector observes reasoning and
        # content in separate windows (so a reasoning loop and a response loop
        # are distinguished), so it is instantiated twice.
        # The loop detector is gated by its own switch so that entering the
        # streaming path for another feature (stall, nudge, ...) does not
        # silently re-enable loop detection the operator switched off.
        thinking = ThinkingLoopDetector.from_settings() if settings.loop_detection_enabled else None
        response = ThinkingLoopDetector.from_settings() if settings.loop_detection_enabled else None
        runaway = (
            RunawayReasoningDetector.from_settings() if settings.runaway_reasoning_enabled else None
        )
        stall = StallDetector.from_settings() if settings.stall_detection_enabled else None

        lines = upstream.aiter_lines()
        while True:
            # Bound each read by the stall budget. Keepalive/comment lines
            # arrive without resetting the timer, so a connection kept open by
            # pings still trips the watchdog once the model goes silent.
            timeout = max(0.0, stall.remaining()) if stall else None
            try:
                line = await asyncio.wait_for(anext(lines), timeout=timeout)
            except asyncio.TimeoutError:
                # A timeout can only occur when `stall` set a budget; with
                # stall detection disabled the wait is unbounded and never
                # raises here.
                assert stall is not None
                outcome.verdict = stall.check() or stall.verdict()
                outcome.verdict_stream = "stalled"
                return
            except StopAsyncIteration:
                break

            if not line or not line.startswith("data:"):
                continue
            # Keep the raw SSE line so a streaming client can be served the
            # upstream's exact event framing.
            outcome.sse_lines.append(line)
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            delta_content, delta_reasoning, delta_tool_calls, chunk_meta = _extract_stream_chunk(
                chunk
            )
            _merge_meta(outcome.meta, chunk_meta)
            _merge_tool_calls(outcome.tool_calls, delta_tool_calls)

            # Any model-produced delta proves the model is alive; feed the
            # watchdogs uniformly. Tool-call fragments count too: a model
            # slowly streaming arguments is working, not stalled.
            if delta_reasoning or delta_content or delta_tool_calls:
                if stall is not None:
                    stall.note(
                        reasoning=delta_reasoning,
                        content=delta_content,
                        tool_calls=delta_tool_calls,
                    )
                if runaway is not None:
                    runaway.note(
                        reasoning=delta_reasoning,
                        content=delta_content,
                        tool_calls=delta_tool_calls,
                    )

            if delta_reasoning:
                outcome.reasoning += delta_reasoning
                if thinking is not None:
                    thinking.note(reasoning=delta_reasoning)
            if delta_content:
                outcome.content += delta_content
                if response is not None:
                    response.note(content=delta_content)

            # Check the watchdogs in the same order as before: reasoning loop,
            # response loop, then runaway reasoning.
            if thinking is not None:
                verdict = thinking.check()
                if verdict is not None:
                    outcome.verdict = verdict
                    outcome.verdict_stream = "thinking"
                    return
            if response is not None:
                verdict = response.check()
                if verdict is not None:
                    outcome.verdict = verdict
                    outcome.verdict_stream = "response"
                    return
            if runaway is not None:
                verdict = runaway.check()
                if verdict is not None:
                    outcome.verdict = verdict
                    outcome.verdict_stream = "reasoning"
                    return

        # End of stream: run the trailing-text check, then the terminal
        # ``finish_reason == "length"`` classification.
        if outcome.verdict is None and response is not None:
            outcome.verdict = response.check()
            if outcome.verdict is not None:
                outcome.verdict_stream = "response"
        if outcome.verdict is None and thinking is not None:
            outcome.verdict = thinking.check()
            if outcome.verdict is not None:
                outcome.verdict_stream = "thinking"
        if outcome.verdict is None:
            self._classify_length(outcome)

    def _classify_length(self, outcome: StreamOutcome) -> None:
        """Classify a terminal ``finish_reason == "length"`` turn.

        The upstream filled the output window without the model finishing.
        llama.cpp reports this as ``n_tokens = 8191, truncated = 1`` and
        LiteLLM surfaces it as ``finish_reason: "length"``. That is a
        definitive runaway signal — the model never stopped — so treat it as a
        loop regardless of whether the content looked repetitive. This catches
        *drift* loops (near-identical but not verbatim) which the
        compression-ratio check misses. When the window was spent *reasoning*
        with no content or tool calls at all, it is classified as
        runaway-reasoning instead (#17), which gets a stop-thinking nudge
        rather than varied sampling.
        """
        if outcome.meta.get("finish_reason") != "length":
            return
        if (
            settings.runaway_reasoning_enabled
            and (outcome.reasoning or "").strip()
            and not (outcome.content or "").strip()
            and not outcome.tool_calls
        ):
            outcome.verdict = StreamVerdict(
                kind=VerdictKind.RUNAWAY_REASONING,
                reason=(
                    "output window exhausted while reasoning (finish_reason=length, no content)"
                ),
                details={
                    "finish_reason": "length",
                    "reasoning_tokens": int(len(outcome.reasoning) / 4),
                },
            )
            outcome.verdict_stream = "reasoning"
        elif settings.loop_detection_enabled:
            outcome.verdict = StreamVerdict(
                kind=VerdictKind.LOOP,
                reason=("output window exhausted (finish_reason=length)"),
                details={"finish_reason": "length"},
            )
            outcome.verdict_stream = "response"
