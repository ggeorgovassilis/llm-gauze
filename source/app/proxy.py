"""Reverse-proxy forwarding to the upstream local LLM provider.

Every request is forwarded to the configured upstream, with each attempt
recorded for later analysis. Failures are routed through the remediation
pipeline: the understanding layer classifies them, and the action layer
(retry with backoff) decides whether and when to try again.
"""

import asyncio
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field

import httpx
from fastapi import Request
from fastapi.responses import Response

from app.config import settings
from app.pipeline import (  # noqa: F401  (_merge_tool_calls re-exported for tests)
    StreamingPipeline,
    _merge_tool_calls,
)
from app.recorder import Recorder
from app.remediation.base import (
    Detector,
    DiagnosisCode,
    Remediation,
    RetryPolicy,
    StreamVerdict,
    Turn,
)
from app.remediation.coast import CoastPolicy
from app.remediation.context import ContextWindowDetector
from app.remediation.loop_retry import LoopRetryPolicy
from app.remediation.nudge import NudgePolicy
from app.remediation.overflow import MessageOverflowGuard
from app.remediation.retry import ExponentialBackoff, RetryableDetector
from app.remediation.runaway import RunawayReasoningPolicy
from app.remediation.think import ThinkContentGuard
from app.remediation.tool_call import ToolCallGuard
from app.remediation.verdicts import route_for
from app.telemetry import telemetry

logger = logging.getLogger("llm_gauze.proxy")

# Headers that must not be forwarded verbatim (RFC 7230 hop-by-hop headers).
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}

# Request headers whose values are secrets: forwarded, never logged or recorded.
_CREDENTIAL_HEADERS = {"authorization", "proxy-authorization"}

# Headers we manage ourselves on the outbound response.
_OWNED_RESPONSE_HEADERS = {"content-length", "content-type"}


@dataclass
class _StreamResult:
    """Outcome of one streamed exchange with the upstream.

    Either ``response`` is set (a terminal outcome: loop/stall abort,
    context-window overflow, transport failure, or HTTP error) or the success
    fields carry the assembled turn for the caller to finalise (nudge, then
    the placeholder floor, then reconstruct).
    """

    response: Response | None = None
    # Set when the exchange ended in a content verdict (loop or stall); lets
    # the caller decide whether to remediate (loop) or abort outright (stall).
    loop_verdict: StreamVerdict | None = None
    loop_stream: str | None = None
    content: str = ""
    reasoning: str = ""
    tool_calls: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    sse_lines: list = field(default_factory=list)
    finish_reason: str = "stop"
    relocate_changes: list = field(default_factory=list)
    # Recording scaffolding for the success path.
    status: int | None = None
    resp_headers: dict = field(default_factory=dict)
    attempt: int = 1
    max_attempts: int = 1
    duration: float = 0.0
    # Terminal outcome label (stable vocabulary), set when ``response`` is set.
    outcome: str | None = None


def _decode(data: bytes) -> str | None:
    """Decode a body for recording, falling back to replacement chars."""
    if not data:
        return None
    return data.decode("utf-8", errors="replace")


def _diagnosis_entry(diagnosis) -> dict | None:
    """Serialize a ``Diagnosis`` for the recorder."""
    if not diagnosis:
        return None
    entry = {"retryable": diagnosis.retryable, "reason": diagnosis.reason}
    if diagnosis.code:
        entry["code"] = diagnosis.code
    return entry


def _attempt_outcome(status: int | None, error: dict | None, diagnosis) -> str:
    """Classify a recorded exchange into the stable ``outcome`` vocabulary.

    Persists the same labels telemetry already emits as the ``outcome`` label
    on ``requests_total``, so a consumer can route a record by ``outcome``
    instead of key-sniffing which fields happen to be present. Verdict aborts
    do not use this helper — they take ``route.outcome`` directly.
    """
    if diagnosis is not None and diagnosis.code == DiagnosisCode.CONTEXT_WINDOW_EXCEEDED:
        return DiagnosisCode.CONTEXT_WINDOW_EXCEEDED.value
    if error is not None:
        return "upstream_failure"
    if status is None:
        return "upstream_failure"
    if status >= 400:
        return "http_error"
    return "success"


def _is_chat_completion(path: str) -> bool:
    """Whether this path is an OpenAI chat-completion request."""
    return path.rstrip("/").endswith("/chat/completions")


def _streaming_feature_enabled() -> bool:
    """Whether any streaming-path feature is enabled.

    The streaming path hosts loop detection plus stall detection,
    runaway-reasoning detection, nudge, coast, think-cleanup, and the
    tool-call guard. Each has its own switch, and the path must be entered
    whenever any one of them is on — not only when loop detection is on.

    ``loop_retry_enabled`` is deliberately absent: loop-retry only ever fires
    on a loop verdict, which requires loop detection, so it never needs to
    trigger the streaming path on its own.
    """
    return any(
        (
            settings.loop_detection_enabled,
            settings.stall_detection_enabled,
            settings.runaway_reasoning_enabled,
            settings.think_nudge_enabled,
            settings.coast_detection_enabled,
            settings.think_cleanup_enabled,
            settings.tool_call_guard_enabled,
        )
    )


def _ensure_stream(body: bytes) -> tuple[bytes, str]:
    """Rewrite a JSON request body to force ``stream: true`` upstream."""
    if not body:
        return b'{"stream": true}', "application/json"
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body, "application/json"
    if not isinstance(data, dict):
        return body, "application/json"
    data["stream"] = True
    return json.dumps(data).encode("utf-8"), "application/json"


def _client_wants_stream(body: bytes) -> bool:
    """Whether the client asked for a streaming (SSE) response.

    Clients such as Copilot always send ``stream: true`` and expect the
    response back as SSE ``data:`` events. If we answer a streaming request
    with a plain JSON body, the client's SSE parser yields zero completions.
    """
    if not body:
        return False
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    return bool(data.get("stream"))


def _apply_overflow_guard(body: bytes) -> tuple[list, bytes]:
    """Trim oversized tool results before forwarding (request-side guard).

    Returns ``(changes, body)``: the recorder change records (empty when nothing
    was flagged or the feature is disabled) and the possibly-rewritten body.
    The original bytes are returned untouched when nothing changed, so the
    caller's Content-Length stays valid.
    """
    if not settings.message_overflow_enabled:
        return [], body
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return [], body
    if not isinstance(data, dict):
        return [], body
    guard = MessageOverflowGuard.from_settings()
    data, changes = guard.process(data)
    if not changes:
        return [], body
    return changes, json.dumps(data).encode("utf-8")


def _reconstruct_chat_completion(
    meta: dict,
    content: str,
    reasoning: str,
    tool_calls: list | None = None,
) -> dict:
    """Reassemble a non-streaming ``chat.completion`` from streamed deltas."""
    message: dict = {"role": meta.get("role") or "assistant"}
    if content:
        message["content"] = content
    if reasoning:
        message["reasoning_content"] = reasoning
    if tool_calls:
        message["tool_calls"] = tool_calls
    body = {
        "id": meta.get("id") or uuid.uuid4().hex,
        "object": "chat.completion",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model") or "",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": meta.get("finish_reason") or "stop",
                "logprobs": None,
            }
        ],
    }
    if meta.get("usage"):
        body["usage"] = meta["usage"]
    return body


def _reconstruct_sse_lines(
    meta: dict,
    content: str,
    reasoning: str,
    tool_calls: list | None = None,
) -> list[str]:
    """Rebuild SSE ``data:`` lines from reconstructed (post-cleanup) values.

    Used for a ``stream: true`` client when the think-tag guard changed the
    output: the raw upstream lines still carry the leaked tags, so they cannot
    be passed through verbatim. The reconstructed stream mirrors the same
    cleaned values that the non-streaming client receives.
    """
    base = {
        "id": meta.get("id") or uuid.uuid4().hex,
        "object": "chat.completion.chunk",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model") or "",
    }
    lines: list[str] = []

    def _emit(delta: dict, finish_reason: str | None = None) -> None:
        chunk = {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        lines.append("data: " + json.dumps(chunk))

    _emit({"role": meta.get("role") or "assistant"})
    if reasoning:
        _emit({"reasoning_content": reasoning})
    if content:
        _emit({"content": content})
    for index, call in enumerate(tool_calls or []):
        _emit({"tool_calls": [{**call, "index": index}]})
    _emit({}, finish_reason=meta.get("finish_reason") or "stop")
    lines.append("data: [DONE]")
    return lines


def _filtered_headers(headers: dict) -> tuple[dict, str | None]:
    """Strip hop-by-hop and self-managed headers; return (headers, ctype)."""
    content_type = headers.get("content-type")
    out = {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() not in _OWNED_RESPONSE_HEADERS
    }
    return out, content_type


def _redact_credentials(headers: dict) -> dict:
    """Copy of ``headers`` with credential values masked, for logs and records."""
    return {k: "[redacted]" if k.lower() in _CREDENTIAL_HEADERS else v for k, v in headers.items()}


def _abort_error_response(verdict: StreamVerdict) -> Response:
    """Build the client-facing response for an aborted stream (loop/stall)."""
    route = route_for(verdict.kind)
    body = json.dumps(
        {
            "error": {
                "message": route.message,
                "type": f"{route.kind.value}_detected",
                "reason": verdict.reason,
                "details": verdict.details,
            }
        }
    ).encode("utf-8")
    return Response(
        content=body,
        status_code=route.abort_status,
        media_type="application/json",
    )


def _context_window_error_response(
    status: int | None,
    body: bytes | None,
    headers: dict | None,
    diagnosis,
) -> Response:
    """Pass through the upstream's context-window overflow response verbatim.

    The upstream told us the request cannot fit its context window; retrying
    cannot help, so we stop retrying. But the upstream's own error body carries
    exactly what the user needs to see — the token count, ``n_ctx``, and how to
    fix it — so we forward it unchanged rather than translating it into our own
    message. Only when there is no upstream body to forward (e.g. the overflow
    arrived as an exception message) do we synthesise a minimal named error.
    """
    if body:
        filtered_headers, ctype = _filtered_headers(headers or {})
        return Response(
            content=body,
            status_code=status or settings.context_window_abort_status,
            headers=filtered_headers,
            media_type=ctype or "application/json",
        )
    synthesized = json.dumps(
        {
            "error": {
                "message": "context window exceeded",
                "type": DiagnosisCode.CONTEXT_WINDOW_EXCEEDED,
                "reason": diagnosis.reason,
                "details": {"upstream_status": status},
            }
        }
    ).encode("utf-8")
    return Response(
        content=synthesized,
        status_code=settings.context_window_abort_status,
        media_type="application/json",
    )


def _build_retry_policy() -> RetryPolicy:
    """Assemble the retry policy (understanding + action) from settings."""
    statuses = {int(s.strip()) for s in settings.retryable_status_codes.split(",") if s.strip()}
    detector: Detector = RetryableDetector(statuses)
    if settings.context_window_detection_enabled:
        detector = ContextWindowDetector.from_settings(detector)
    backoff = ExponentialBackoff(
        initial=settings.retry_backoff_initial,
        base=settings.retry_backoff_base,
        maximum=settings.retry_backoff_max,
        jitter=settings.retry_backoff_jitter,
    )
    return RetryPolicy(
        max_attempts=settings.retry_max_attempts,
        detector=detector,
        backoff=backoff,
    )


def _first_applicable_step(
    ladder: list[Remediation],
    attempts: dict[str, int],
    turn: Turn,
    request_body: dict,
) -> Remediation | None:
    """Return the first ladder rung whose trigger fires and has budget left.

    Rungs are consulted in list order; the first match wins. A rung whose
    ``max_attempts`` budget is already spent is skipped.
    """
    for step in ladder:
        if attempts.get(step.name, 0) >= step.max_attempts:
            continue
        if step.applies(turn, request_body):
            return step
    return None


def _find_step(ladder: list[Remediation], name: str) -> Remediation | None:
    """Return the ladder rung with the given ``name``, else ``None``."""
    for step in ladder:
        if step.name == name:
            return step
    return None


def _remediation_outcome(
    step: Remediation | None,
    attempts: dict[str, int],
    turn: Turn,
    request_body: dict,
) -> str | None:
    """Classify a rung's final disposition: ``succeeded`` / ``exhausted`` / None.

    A rung that still triggers on the final turn exhausted its budget; one that
    fired earlier but no longer triggers succeeded. A rung that never fired has
    no outcome.
    """
    if step is None:
        return None
    if step.applies(turn, request_body):
        return "exhausted"
    if attempts.get(step.name, 0) > 0:
        return "succeeded"
    return None


class Proxy:
    """Forwards requests to the upstream, retrying transient failures."""

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder
        self.client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.request_timeout,
                write=settings.request_timeout,
                pool=settings.connect_timeout,
            ),
        )
        self.retry_policy = _build_retry_policy()
        self.pipeline = StreamingPipeline(self.client)

    async def aclose(self) -> None:
        """Release the upstream client and its connection pool.

        Called from the FastAPI lifespan on shutdown so the pool is not
        abandoned at process exit.
        """
        await self.client.aclose()

    def _log_completion(
        self, method: str, path: str, status: int, outcome: str, duration: float
    ) -> None:
        """Log the one-line per-request completion summary (INFO)."""
        logger.info(
            "%s %s status=%s duration=%.3fs outcome=%s",
            method,
            path,
            status,
            duration,
            outcome,
        )

    def _emit_attempt_telemetry(
        self,
        status: int | None,
        error: dict | None,
        duration: float,
    ) -> None:
        """Emit the per-attempt telemetry shared by both request shapes."""
        telemetry.incr("attempts_total")
        telemetry.observe_latency(duration)
        if error is not None:
            telemetry.incr("upstream_errors_total", {"type": error["type"]})
            telemetry.set_upstream_down(True)
        elif status is not None:
            telemetry.set_upstream_down(False)

    async def _retry_decision(
        self,
        diagnosis,
        attempt: int,
        max_attempts: int,
    ) -> str | None:
        """Decide the next step for a failed attempt, shared by both shapes.

        Returns ``"context_window"`` (terminal overflow — caller builds its
        bespoke response), ``"retry"`` (backed off and slept; caller continues
        the loop), or ``None`` (final: caller settles on the outcome or gives
        up). Centralising ``should_retry``, backoff/sleep, and the
        context-window special case here keeps the buffered and streaming
        retry loops from drifting apart.
        """
        policy = self.retry_policy
        if diagnosis is not None and diagnosis.code == DiagnosisCode.CONTEXT_WINDOW_EXCEEDED:
            telemetry.incr(
                "requests_total",
                {"outcome": DiagnosisCode.CONTEXT_WINDOW_EXCEEDED.value},
            )
            telemetry.incr("context_window_aborts_total")
            return "context_window"
        if policy.should_retry(diagnosis, attempt):
            delay = policy.backoff.delay(attempt)
            telemetry.incr("retries_total")
            logger.info(
                "retrying upstream request (attempt %d/%d)",
                attempt + 1,
                max_attempts,
            )
            logger.debug(
                "backoff delay %.3fs before attempt %d (%s)",
                delay,
                attempt + 1,
                diagnosis,
            )
            await asyncio.sleep(delay)
            return "retry"
        return None

    async def forward(self, request: Request, path: str) -> Response:
        started = time.time()
        method = request.method
        body = await request.body()
        url = f"/{path}" if path else "/"
        query = request.url.query
        request_id = uuid.uuid4().hex

        logger.debug(
            "incoming request method=%s path=%s headers=%r body=%r",
            method,
            url,
            _redact_credentials(dict(request.headers)),
            _decode(body),
        )

        # Request-side guard: trim oversized tool results before forwarding, so
        # a single runaway ``role: "tool"`` message cannot silently eat the
        # model's context window. Chat completions only — the only shape that
        # carries ``messages``.
        overflow_changes: list = []
        if _is_chat_completion(url):
            overflow_changes, body = _apply_overflow_guard(body)
            if overflow_changes:
                logger.warning(
                    "message overflow trimmed %d oversized tool result(s)",
                    len(overflow_changes),
                )

        base_entry = {
            "request_id": request_id,
            "method": method,
            "path": url,
            "query": query,
            "request_headers": _redact_credentials(dict(request.headers)),
            "request_body": _decode(body),
            "message_overflow": overflow_changes or None,
        }

        # The client's Authorization header is forwarded to the upstream unchanged.
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}
        if overflow_changes:
            # The body was rewritten, so any inbound Content-Length is stale;
            # let httpx recompute it from the actual body (see the streaming
            # path's comment on the same h11 failure mode).
            headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}

        # Chat completions get the streaming path whenever any streaming
        # feature is enabled; everything else goes through the buffered
        # forwarder unchanged. Entering the streaming path must not depend on
        # loop detection alone, otherwise disabling loop detection to silence
        # one false positive would silently disable every other streaming
        # feature regardless of its own switch.
        if _streaming_feature_enabled() and _is_chat_completion(url):
            return await self._forward_streaming(
                request_id, method, url, query, body, headers, base_entry, started
            )

        policy = self.retry_policy
        final_status: int
        final_headers: dict
        final_body: bytes
        outcome = "unknown"

        for attempt in range(1, policy.max_attempts + 1):
            attempt_started = time.time()
            status: int | None = None
            resp_headers: dict = {}
            resp_body: bytes = b""
            error: dict | None = None
            diagnosis = None

            try:
                upstream = await self.client.request(
                    method,
                    url,
                    params=query,
                    content=body,
                    headers=headers,
                )
                status = upstream.status_code
                resp_headers = dict(upstream.headers)
                resp_body = upstream.content
                diagnosis = policy.detector.diagnose_status(status, body=resp_body)
            except Exception as exc:  # noqa: BLE001 - capture everything
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                diagnosis = policy.detector.diagnose_exception(exc)
                logger.warning(
                    "upstream attempt %d/%d failed: %s",
                    attempt,
                    policy.max_attempts,
                    diagnosis,
                )

            self._emit_attempt_telemetry(status, error, time.time() - attempt_started)

            # Action: retry with backoff, or settle on the final outcome.
            decision = await self._retry_decision(diagnosis, attempt, policy.max_attempts)

            # Data collection: record every attempt. `response_body` holds only
            # what is actually delivered to the client — a retried attempt's
            # body is discarded, so it is recorded as null.
            await self.recorder.record(
                {
                    **base_entry,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "outcome": _attempt_outcome(status, error, diagnosis),
                    "status": status,
                    "response_headers": resp_headers,
                    "response_body": _decode(resp_body) if decision != "retry" else None,
                    "error": error,
                    "diagnosis": _diagnosis_entry(diagnosis),
                    "duration": time.time() - attempt_started,
                }
            )

            if decision == "retry":
                continue
            if decision == "context_window":
                resp = _context_window_error_response(status, resp_body, resp_headers, diagnosis)
                self._log_completion(
                    method, url, resp.status_code, "context_window_exceeded", time.time() - started
                )
                return resp

            if status is None:
                final_status = 502
                final_headers = {"content-type": "application/json"}
                final_body = b'{"error": "upstream failure"}'
                outcome = "upstream_failure"
                logger.error(
                    "terminal upstream failure for %s %s after %d attempt(s)",
                    method,
                    url,
                    attempt,
                )
            else:
                final_status = status
                final_headers = resp_headers
                final_body = resp_body
                outcome = "success" if status < 400 else "http_error"
            telemetry.incr("requests_total", {"outcome": outcome})
            break

        content_type = final_headers.get("content-type")
        out_headers = {
            k: v
            for k, v in final_headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS and k.lower() not in _OWNED_RESPONSE_HEADERS
        }

        resp = Response(
            content=final_body,
            status_code=final_status,
            headers=out_headers,
            media_type=content_type,
        )
        self._log_completion(method, url, final_status, outcome, time.time() - started)
        logger.debug(
            "response status=%s headers=%r body=%r",
            final_status,
            dict(final_headers),
            _decode(final_body),
        )
        return resp

    async def _forward_streaming(
        self,
        request_id: str,
        method: str,
        url: str,
        query: str,
        body: bytes,
        headers: dict,
        base_entry: dict,
        started: float | None = None,
    ) -> Response:
        """Stream the upstream response; drive the remediation ladder.

        The upstream is asked for ``stream: true`` so the output can be
        observed incrementally. Two independent detectors run — one over the
        thinking (reasoning) stream, one over the visible response.

        Remediation is an *ordered, composable ladder* of :class:`Remediation`
        steps (runaway nudge, loop retry, empty-turn nudge, coast re-prompt),
        each with its own trigger and attempt budget. After each exchange the
        ladder is walked in order and the first rung whose trigger fires
        re-submits; when no rung fires the outcome is final (abort on a
        verdict, or finalise the assembled turn). A *stall* is never remediated
        — retrying just re-waits for a silent model.
        """
        # `_ensure_stream` rewrites the body (stream forced on, JSON
        # re-serialised), so any inbound Content-Length is stale. Drop it and
        # let httpx recompute it from the actual body; forwarding the stale
        # value makes h11 raise "Too much data for declared Content-Length".
        if started is None:
            started = time.time()
        out_headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
        content_type = _ensure_stream(body)[1]
        if content_type:
            out_headers["content-type"] = content_type

        # A streaming client (stream: true) needs SSE back; a non-streaming
        # client gets the reconstructed chat.completion. See `_client_wants_stream`.
        client_wants_stream = _client_wants_stream(body)

        guard = ThinkContentGuard.from_settings() if settings.think_cleanup_enabled else None
        ladder = self._build_remediation_ladder()
        attempts: dict[str, int] = {step.name: 0 for step in ladder}

        current_body = body
        while True:
            stream_body, _ = _ensure_stream(current_body)
            request_payload = json.loads(current_body)
            result = await self._stream_once(
                request_id,
                method,
                url,
                query,
                stream_body,
                out_headers,
                base_entry,
            )

            turn = Turn(
                finish_reason=result.finish_reason,
                content=result.content,
                reasoning=result.reasoning,
                tool_calls=result.tool_calls,
                verdict=result.loop_verdict,
            )

            step = _first_applicable_step(ladder, attempts, turn, request_payload)
            if step is not None:
                logger.warning(
                    "remediation triggered: %s (attempt %d/%d)",
                    step.name,
                    attempts[step.name] + 1,
                    step.max_attempts,
                )
                resubmitted = step.apply(turn, request_payload)
                await self._record_remediation_pass(
                    step,
                    base_entry,
                    result,
                    client_wants_stream,
                    attempts[step.name],
                    resubmitted,
                )
                attempts[step.name] += 1
                current_body = json.dumps(resubmitted).encode("utf-8")
                continue

            # No rung fired: the outcome is final. A verdict (loop/stall/
            # runaway) aborts; a clean turn is finalised (nudge/coast floors).
            if result.response is not None:
                resp = result.response
                self._log_completion(
                    method,
                    url,
                    resp.status_code,
                    result.outcome or "unknown",
                    time.time() - started,
                )
                logger.debug(
                    "response status=%s headers=%r body=%r",
                    resp.status_code,
                    dict(resp.headers),
                    resp.body,
                )
                return resp

            nudge_outcome = _remediation_outcome(
                _find_step(ladder, "nudge"), attempts, turn, request_payload
            )
            coast_outcome = _remediation_outcome(
                _find_step(ladder, "coast"), attempts, turn, request_payload
            )
            resp = await self._finalize_success(
                base_entry,
                result,
                guard,
                client_wants_stream,
                attempts.get("nudge", 0),
                nudge_outcome,
                attempts.get("coast", 0),
                coast_outcome,
            )
            self._log_completion(method, url, resp.status_code, "success", time.time() - started)
            logger.debug(
                "response status=%s headers=%r body=%r",
                resp.status_code,
                dict(resp.headers),
                resp.body,
            )
            return resp

    def _build_remediation_ladder(self) -> list[Remediation]:
        """Assemble the ordered, composable remediation ladder from settings.

        Order matters: verdict remediations (runaway nudge, then loop retry)
        run first, then the empty-turn rungs (nudge, then coast). A disabled
        rung is simply left out — the ladder is a plain list of enabled steps.
        """
        ladder: list[Remediation] = []
        if settings.runaway_reasoning_enabled:
            ladder.append(RunawayReasoningPolicy.from_settings())
        if settings.loop_retry_enabled:
            ladder.append(LoopRetryPolicy.from_settings())
        if settings.think_nudge_enabled:
            ladder.append(NudgePolicy.from_settings())
        if settings.coast_detection_enabled:
            ladder.append(CoastPolicy.from_settings())
        return ladder

    async def _record_remediation_pass(
        self,
        step: Remediation,
        base_entry: dict,
        result: _StreamResult,
        client_wants_stream: bool,
        attempt: int,
        resubmitted_body: dict,
    ) -> None:
        """Record a re-submission that a ladder rung just triggered.

        Dispatches to the rung-specific recorder hook so each entry keeps its
        bespoke shape (``nudge``/``coast``/``runaway``/``loop_retry``).
        """
        if step.name == "nudge":
            await self._record_nudge_pass(base_entry, result, client_wants_stream, attempt)
        elif step.name == "coast":
            await self._record_coast_pass(base_entry, result, client_wants_stream, attempt)
        elif step.name == "runaway":
            await self._record_runaway_pass(base_entry, result, attempt, resubmitted_body)
        elif step.name == "loop_retry":
            await self._record_loop_retry_pass(base_entry, result, attempt, resubmitted_body)
        else:  # pragma: no cover - defensive; all ladder names are known
            raise ValueError(f"unknown remediation step: {step.name}")

    async def _stream_once(
        self,
        request_id: str,
        method: str,
        url: str,
        query: str,
        stream_body: bytes,
        out_headers: dict,
        base_entry: dict,
    ) -> _StreamResult:
        """One retrying streamed exchange against the upstream.

        Transport/HTTP failures are retried here (with backoff), exactly as the
        buffered path does; a loop/stall/context-window verdict is terminal.
        On success the assembled turn is returned for the caller to finalise —
        no record is written here for the success case.
        """
        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.time()
            status: int | None = None
            resp_headers: dict = {}
            error_body = b""
            invalid_response = False
            error: dict | None = None
            diagnosis = None
            content = ""
            reasoning = ""
            tool_calls: list = []
            sse_lines: list[str] = []
            meta: dict = {}
            loop_verdict: StreamVerdict | None = None
            loop_stream: str | None = None

            try:
                outcome = await self.pipeline.run(method, url, query, stream_body, out_headers)
                status = outcome.status
                resp_headers = outcome.resp_headers
                error_body = outcome.error_body
                invalid_response = outcome.invalid_response
                loop_verdict = outcome.verdict
                loop_stream = outcome.verdict_stream
                content = outcome.content
                reasoning = outcome.reasoning
                tool_calls = outcome.tool_calls
                meta = outcome.meta
                sse_lines = outcome.sse_lines
            except Exception as exc:  # noqa: BLE001 - capture everything
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                diagnosis = policy.detector.diagnose_exception(exc)
                logger.warning(
                    "upstream stream attempt %d/%d failed: %s",
                    attempt,
                    policy.max_attempts,
                    diagnosis,
                )

            self._emit_attempt_telemetry(status, error, time.time() - started)

            # Content verdict (loop or stalled) -> terminal; never retry.
            # Retrying just re-enters the loop, or re-waits for a silent model.
            if loop_verdict is not None:
                route = route_for(loop_verdict.kind)
                await self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "outcome": route.outcome,
                        "status": status,
                        "response_body": None,
                        "abort_kind": loop_verdict.kind,
                        "abort_stream": loop_stream,
                        "abort_reason": loop_verdict.reason,
                        "abort_details": loop_verdict.details,
                        # Preserve whatever the model produced before the
                        # abort so the hang/loop can be diagnosed post-hoc.
                        "partial_content": content or None,
                        "partial_reasoning": reasoning or None,
                        "partial_tool_calls": tool_calls or None,
                        "duration": time.time() - started,
                    }
                )
                telemetry.incr("requests_total", {"outcome": route.outcome})
                if route.stream_labeled:
                    telemetry.incr(route.abort_metric, {"stream": loop_stream or "unknown"})
                else:
                    telemetry.incr(route.abort_metric)
                logger.error(
                    "%s DETECTED request_id=%s stream=%s reason=%s",
                    loop_verdict.kind.upper(),
                    request_id,
                    loop_stream,
                    loop_verdict.reason,
                )
                return _StreamResult(
                    response=_abort_error_response(loop_verdict),
                    loop_verdict=loop_verdict,
                    loop_stream=loop_stream,
                    outcome=route.outcome,
                )

            # Transport error -> retry or give up (mirrors the buffered path).
            if error is not None:
                # `error` is only ever set by the exception handler above,
                # which also classifies the failure, so `diagnosis` is set.
                assert diagnosis is not None
                await self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "outcome": _attempt_outcome(status, error, diagnosis),
                        "status": status,
                        "response_headers": resp_headers,
                        "response_body": None,
                        "error": error,
                        "diagnosis": _diagnosis_entry(diagnosis),
                        "duration": time.time() - started,
                    }
                )
                decision = await self._retry_decision(diagnosis, attempt, policy.max_attempts)
                if decision == "retry":
                    continue
                if decision == "context_window":
                    return _StreamResult(
                        outcome="context_window_exceeded",
                        response=_context_window_error_response(
                            status, error_body, resp_headers, diagnosis
                        ),
                    )
                telemetry.incr("requests_total", {"outcome": "upstream_failure"})
                logger.error(
                    "terminal upstream failure for %s %s after %d attempt(s)",
                    method,
                    url,
                    attempt,
                )
                return _StreamResult(
                    outcome="upstream_failure",
                    response=Response(
                        content=b'{"error": "upstream failure"}',
                        status_code=502,
                        media_type="application/json",
                    ),
                )

            # HTTP error status -> retry or return it.
            if status is not None and status >= 400:
                diagnosis = (
                    None
                    if invalid_response
                    else policy.detector.diagnose_status(status, body=error_body)
                )
                decision = (
                    None
                    if invalid_response
                    else await self._retry_decision(diagnosis, attempt, policy.max_attempts)
                )
                await self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "outcome": _attempt_outcome(status, None, diagnosis),
                        "status": status,
                        "response_headers": resp_headers,
                        "response_body": _decode(error_body) if decision != "retry" else None,
                        "diagnosis": _diagnosis_entry(diagnosis),
                        "duration": time.time() - started,
                    }
                )
                if decision == "retry":
                    continue
                if decision == "context_window":
                    return _StreamResult(
                        outcome="context_window_exceeded",
                        response=_context_window_error_response(
                            status, error_body, resp_headers, diagnosis
                        ),
                    )
                telemetry.incr("requests_total", {"outcome": "http_error"})
                err_headers, ctype = _filtered_headers(resp_headers)
                return _StreamResult(
                    outcome="http_error",
                    response=Response(
                        content=error_body,
                        status_code=status,
                        headers=err_headers,
                        media_type=ctype,
                    ),
                )

            # Success: relocate leaked think tags (the caller decides whether
            # to nudge or apply the placeholder floor before reconstructing).
            relocate_changes: list = []
            if settings.think_cleanup_enabled:
                guard = ThinkContentGuard.from_settings()
                content, reasoning, relocate_changes = guard.relocate(content, reasoning)
                if relocate_changes:
                    logger.warning(
                        "think-cleanup relocated %d leaked thinking tag(s)",
                        len(relocate_changes),
                    )

            return _StreamResult(
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
                meta=meta,
                sse_lines=sse_lines,
                finish_reason=meta.get("finish_reason") or "stop",
                relocate_changes=relocate_changes,
                status=status,
                resp_headers=resp_headers,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                duration=time.time() - started,
            )

        # Exhausted retries (all attempts errored) — defensive fallback.
        return _StreamResult(
            outcome="upstream_failure",
            response=Response(
                content=b'{"error": "upstream failure"}',
                status_code=502,
                media_type="application/json",
            ),
        )

    async def _record_nudge_pass(
        self,
        base_entry: dict,
        result: _StreamResult,
        client_wants_stream: bool,
        nudge_attempt: int,
    ) -> None:
        """Record an empty-text turn that triggered a nudge re-submission."""
        turn_body = json.dumps(
            _reconstruct_chat_completion(
                result.meta, result.content, result.reasoning, result.tool_calls
            )
        ).encode("utf-8")
        await self.recorder.record(
            {
                **base_entry,
                "attempt": result.attempt,
                "max_attempts": result.max_attempts,
                "outcome": "success",
                "status": result.status,
                "response_headers": result.resp_headers,
                "response_body": None,
                "pre_remediation_body": _decode(turn_body),
                "streamed": client_wants_stream,
                "think_cleanup": result.relocate_changes or None,
                "nudge": {"attempt": nudge_attempt, "outcome": "triggered"},
                "diagnosis": None,
                "duration": result.duration,
            }
        )

    async def _record_coast_pass(
        self,
        base_entry: dict,
        result: _StreamResult,
        client_wants_stream: bool,
        coast_attempt: int,
    ) -> None:
        """Record a coasted turn that triggered a re-prompt re-submission."""
        turn_body = json.dumps(
            _reconstruct_chat_completion(
                result.meta, result.content, result.reasoning, result.tool_calls
            )
        ).encode("utf-8")
        await self.recorder.record(
            {
                **base_entry,
                "attempt": result.attempt,
                "max_attempts": result.max_attempts,
                "outcome": "success",
                "status": result.status,
                "response_headers": result.resp_headers,
                "response_body": None,
                "pre_remediation_body": _decode(turn_body),
                "streamed": client_wants_stream,
                "think_cleanup": result.relocate_changes or None,
                "coast": {"attempt": coast_attempt, "outcome": "triggered"},
                "diagnosis": None,
                "duration": result.duration,
            }
        )

    async def _record_runaway_pass(
        self,
        base_entry: dict,
        result: _StreamResult,
        runaway_attempt: int,
        resubmitted_body: dict,
    ) -> None:
        """Record a runaway-reasoning turn that triggered a stop-thinking nudge.

        ``resubmitted_body`` is the exact request body sent upstream on the
        re-submission, so the appended instruction is auditable against the
        client's original ``request_body`` (already in the entry).
        """
        await self.recorder.record(
            {
                **base_entry,
                "attempt": result.attempt,
                "max_attempts": result.max_attempts,
                "outcome": "runaway_reasoning_aborted",
                "status": result.status,
                "response_headers": result.resp_headers,
                "response_body": None,
                "runaway": {
                    "attempt": runaway_attempt,
                    "outcome": "triggered",
                    "resubmitted_body": resubmitted_body,
                },
                "diagnosis": None,
                "duration": result.duration,
            }
        )

    async def _record_loop_retry_pass(
        self,
        base_entry: dict,
        result: _StreamResult,
        loop_retry_attempt: int,
        resubmitted_body: dict,
    ) -> None:
        """Record a looped turn that triggered a varied-sampling re-submission.

        ``resubmitted_body`` is the exact request body sent upstream on the
        re-submission, so the overridden sampling parameters are auditable
        against the client's original ``request_body`` (already in the entry).
        """
        await self.recorder.record(
            {
                **base_entry,
                "attempt": result.attempt,
                "max_attempts": result.max_attempts,
                "outcome": "loop_aborted",
                "status": result.status,
                "response_headers": result.resp_headers,
                "response_body": None,
                "loop_retry": {
                    "attempt": loop_retry_attempt,
                    "outcome": "triggered",
                    "resubmitted_body": resubmitted_body,
                },
                "diagnosis": None,
                "duration": result.duration,
            }
        )

    async def _finalize_success(
        self,
        base_entry: dict,
        result: _StreamResult,
        guard,
        client_wants_stream: bool,
        nudge_attempt: int,
        nudge_outcome: str | None,
        coast_attempt: int,
        coast_outcome: str | None,
    ) -> Response:
        """Apply the placeholder floor, record, and return the final response."""
        content = result.content
        reasoning = result.reasoning
        cleanup_changes = list(result.relocate_changes)
        if guard is not None:
            content, guard_changes = guard.guard_empty(content, reasoning, result.tool_calls)
            cleanup_changes.extend(guard_changes)
            if guard_changes:
                logger.warning("think-cleanup applied placeholder to empty visible turn")

        tool_changes: list = []
        if settings.tool_call_guard_enabled and result.tool_calls:
            tool_changes = ToolCallGuard.from_settings().validate(result.tool_calls)
            if tool_changes:
                logger.warning("tool-call repair changed %d tool call(s)", len(tool_changes))

        final_body = json.dumps(
            _reconstruct_chat_completion(result.meta, content, reasoning, result.tool_calls)
        ).encode("utf-8")

        nudge_entry = None
        if nudge_outcome is not None:
            nudge_entry = {"attempts": nudge_attempt, "outcome": nudge_outcome}

        coast_entry = None
        if coast_outcome is not None:
            coast_entry = {"attempts": coast_attempt, "outcome": coast_outcome}

        await self.recorder.record(
            {
                **base_entry,
                "attempt": result.attempt,
                "max_attempts": result.max_attempts,
                "outcome": "success",
                "status": result.status,
                "response_headers": result.resp_headers,
                "response_body": _decode(final_body),
                "streamed": client_wants_stream,
                "think_cleanup": cleanup_changes or None,
                "tool_repair": tool_changes or None,
                "nudge": nudge_entry,
                "coast": coast_entry,
                "diagnosis": None,
                "duration": result.duration,
            }
        )
        telemetry.incr("requests_total", {"outcome": "success"})

        mutated = bool(cleanup_changes or tool_changes)
        if client_wants_stream:
            if mutated:
                # The raw SSE lines still carry the pre-remediation values
                # (leaked tags or malformed tool-call arguments); rebuild the
                # stream from the cleaned/repaired values instead.
                lines = _reconstruct_sse_lines(result.meta, content, reasoning, result.tool_calls)
            else:
                lines = list(result.sse_lines)
                # Guarantee a clean SSE termination even if the upstream
                # omits the ``[DONE]`` sentinel (some providers do).
                if not lines or lines[-1].strip() != "data: [DONE]":
                    lines.append("data: [DONE]")
            sse_body = "".join(f"{line}\n\n" for line in lines).encode("utf-8")
            return Response(
                content=sse_body,
                status_code=200,
                media_type="text/event-stream",
            )
        return Response(
            content=final_body,
            status_code=200,
            media_type="application/json",
        )
