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

import httpx
from fastapi import Request
from fastapi.responses import Response

from app.config import settings
from app.recorder import Recorder
from app.remediation.base import RetryPolicy, StreamVerdict
from app.remediation.loop import ThinkingLoopDetector
from app.remediation.retry import ExponentialBackoff, RetryableDetector

logger = logging.getLogger("bandaid.proxy")

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

# Headers we manage ourselves on the outbound response.
_OWNED_RESPONSE_HEADERS = {"content-length", "content-type"}


def _decode(data: bytes) -> str | None:
    """Decode a body for recording, falling back to replacement chars."""
    if not data:
        return None
    return data.decode("utf-8", errors="replace")


def _diagnosis_entry(diagnosis) -> dict | None:
    """Serialize a ``Diagnosis`` for the recorder."""
    if not diagnosis:
        return None
    return {"retryable": diagnosis.retryable, "reason": diagnosis.reason}


def _is_chat_completion(path: str) -> bool:
    """Whether this path is an OpenAI chat-completion request."""
    return path.rstrip("/").endswith("/chat/completions")


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


def _extract_stream_chunk(chunk: dict) -> tuple[str | None, str | None, dict]:
    """Pull (content, reasoning, meta) out of one SSE ``data`` payload.

    Tolerates the common OpenAI-compatible shapes: ``delta`` vs ``message``,
    and ``reasoning_content`` / ``reasoning`` / ``thinking`` for the thinking
    stream.
    """
    choices = chunk.get("choices") or []
    choice = choices[0] if choices else {}
    delta = choice.get("delta") or choice.get("message") or {}
    content = delta.get("content")
    reasoning = (
        delta.get("reasoning_content")
        or delta.get("reasoning")
        or delta.get("thinking")
    )
    meta = {
        "id": chunk.get("id"),
        "model": chunk.get("model"),
        "created": chunk.get("created"),
        "role": delta.get("role"),
        "finish_reason": choice.get("finish_reason"),
        "usage": chunk.get("usage"),
    }
    return content, reasoning, meta


def _merge_meta(meta: dict, chunk_meta: dict) -> None:
    """Merge non-None metadata fields into the accumulator (last wins)."""
    for key, value in chunk_meta.items():
        if value is not None:
            meta[key] = value


def _reconstruct_chat_completion(
    meta: dict, content: str, reasoning: str
) -> dict:
    """Reassemble a non-streaming ``chat.completion`` from streamed deltas."""
    message: dict = {"role": meta.get("role") or "assistant"}
    if content:
        message["content"] = content
    if reasoning:
        message["reasoning_content"] = reasoning
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


def _filtered_headers(headers: dict) -> tuple[dict, str | None]:
    """Strip hop-by-hop and self-managed headers; return (headers, ctype)."""
    content_type = headers.get("content-type")
    out = {
        k: v
        for k, v in headers.items()
        if k.lower() not in HOP_BY_HOP_HEADERS
        and k.lower() not in _OWNED_RESPONSE_HEADERS
    }
    return out, content_type


def _loop_error_response(verdict: StreamVerdict) -> Response:
    """Build the client-facing response for an aborted loop."""
    body = json.dumps(
        {
            "error": {
                "message": (
                    "The model entered a repetitive loop and the request "
                    "was aborted."
                ),
                "type": "loop_detected",
                "reason": verdict.reason,
                "details": verdict.details,
            }
        }
    ).encode("utf-8")
    return Response(
        content=body,
        status_code=settings.loop_abort_status,
        media_type="application/json",
    )


def _build_retry_policy() -> RetryPolicy:
    """Assemble the retry policy (understanding + action) from settings."""
    statuses = {
        int(s.strip())
        for s in settings.retryable_status_codes.split(",")
        if s.strip()
    }
    detector = RetryableDetector(statuses)
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

    async def forward(self, request: Request, path: str) -> Response:
        method = request.method
        body = await request.body()
        url = f"/{path}" if path else "/"
        query = request.url.query
        request_id = uuid.uuid4().hex

        base_entry = {
            "request_id": request_id,
            "method": method,
            "path": url,
            "query": query,
            "request_headers": dict(request.headers),
            "request_body": _decode(body),
        }

        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }

        # Chat completions get the streaming loop-detection path; everything
        # else goes through the buffered forwarder unchanged.
        if settings.loop_detection_enabled and _is_chat_completion(url):
            return await self._forward_streaming(
                request_id, method, url, query, body, headers, base_entry
            )

        policy = self.retry_policy
        final_status: int
        final_headers: dict
        final_body: bytes

        for attempt in range(1, policy.max_attempts + 1):
            started = time.time()
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
                diagnosis = policy.detector.diagnose_status(status)
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

            # Data collection: record every attempt, whatever the outcome.
            self.recorder.record(
                {
                    **base_entry,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "status": status,
                    "response_headers": resp_headers,
                    "response_body": _decode(resp_body),
                    "error": error,
                    "diagnosis": (
                        {
                            "retryable": diagnosis.retryable,
                            "reason": diagnosis.reason,
                        }
                        if diagnosis
                        else None
                    ),
                    "duration": time.time() - started,
                }
            )

            # Action: retry with backoff, or settle on the final outcome.
            if policy.should_retry(diagnosis, attempt):
                delay = policy.backoff.delay(attempt)
                logger.warning(
                    "retrying in %.2fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    policy.max_attempts,
                )
                await asyncio.sleep(delay)
                continue

            if status is None:
                final_status = 502
                final_headers = {"content-type": "application/json"}
                final_body = b'{"error": "upstream failure"}'
            else:
                final_status = status
                final_headers = resp_headers
                final_body = resp_body
            break

        content_type = final_headers.get("content-type")
        out_headers = {
            k: v
            for k, v in final_headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
            and k.lower() not in _OWNED_RESPONSE_HEADERS
        }

        return Response(
            content=final_body,
            status_code=final_status,
            headers=out_headers,
            media_type=content_type,
        )

    async def _forward_streaming(
        self,
        request_id: str,
        method: str,
        url: str,
        query: str,
        body: bytes,
        headers: dict,
        base_entry: dict,
    ) -> Response:
        """Stream the upstream response and abort on a thinking/output loop.

        The upstream is asked for ``stream: true`` so the output can be
        observed incrementally. Two independent detectors run — one over the
        thinking (reasoning) stream, one over the visible response — and a
        loop in either aborts the request (never retried: retrying just
        re-enters the loop).
        """
        stream_body, content_type = _ensure_stream(body)
        out_headers = dict(headers)
        if content_type:
            out_headers["content-type"] = content_type

        policy = self.retry_policy
        for attempt in range(1, policy.max_attempts + 1):
            started = time.time()
            status: int | None = None
            resp_headers: dict = {}
            error_body = b""
            error: dict | None = None
            diagnosis = None
            content = ""
            reasoning = ""
            meta: dict = {}
            loop_verdict: StreamVerdict | None = None
            loop_stream: str | None = None

            thinking = ThinkingLoopDetector.from_settings()
            response = ThinkingLoopDetector.from_settings()

            try:
                async with self.client.stream(
                    method,
                    url,
                    params=query,
                    content=stream_body,
                    headers=out_headers,
                ) as upstream:
                    status = upstream.status_code
                    resp_headers = dict(upstream.headers)

                    if status < 400:
                        async for line in upstream.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if not payload or payload == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue

                            delta_content, delta_reasoning, chunk_meta = (
                                _extract_stream_chunk(chunk)
                            )
                            _merge_meta(meta, chunk_meta)

                            if delta_reasoning:
                                reasoning += delta_reasoning
                                verdict = thinking.feed(delta_reasoning)
                                if verdict is not None:
                                    loop_verdict = verdict
                                    loop_stream = "thinking"
                                    break
                            if delta_content:
                                content += delta_content
                                verdict = response.feed(delta_content)
                                if verdict is not None:
                                    loop_verdict = verdict
                                    loop_stream = "response"
                                    break
                    else:
                        error_body = await upstream.aread()

                if loop_verdict is None:
                    loop_verdict = response.flush()
                    if loop_verdict is not None:
                        loop_stream = "response"
                    else:
                        loop_verdict = thinking.flush()
                        if loop_verdict is not None:
                            loop_stream = "thinking"
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

            # Loop -> terminal (never retry; retrying just re-enters the loop).
            if loop_verdict is not None:
                self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "status": status,
                        "loop_detected": True,
                        "loop_stream": loop_stream,
                        "loop_reason": loop_verdict.reason,
                        "loop_details": loop_verdict.details,
                        "duration": time.time() - started,
                    }
                )
                logger.error(
                    "LOOP DETECTED request_id=%s stream=%s reason=%s",
                    request_id,
                    loop_stream,
                    loop_verdict.reason,
                )
                return _loop_error_response(loop_verdict)

            # Transport error -> retry or give up (mirrors the buffered path).
            if error is not None:
                self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "status": status,
                        "response_headers": resp_headers,
                        "error": error,
                        "diagnosis": _diagnosis_entry(diagnosis),
                        "duration": time.time() - started,
                    }
                )
                if policy.should_retry(diagnosis, attempt):
                    delay = policy.backoff.delay(attempt)
                    logger.warning(
                        "retrying in %.2fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        policy.max_attempts,
                    )
                    await asyncio.sleep(delay)
                    continue
                return Response(
                    content=b'{"error": "upstream failure"}',
                    status_code=502,
                    media_type="application/json",
                )

            # HTTP error status -> retry or return it.
            if status is not None and status >= 400:
                diagnosis = policy.detector.diagnose_status(status)
                self.recorder.record(
                    {
                        **base_entry,
                        "attempt": attempt,
                        "max_attempts": policy.max_attempts,
                        "status": status,
                        "response_headers": resp_headers,
                        "diagnosis": _diagnosis_entry(diagnosis),
                        "duration": time.time() - started,
                    }
                )
                if policy.should_retry(diagnosis, attempt):
                    delay = policy.backoff.delay(attempt)
                    logger.warning(
                        "retrying in %.2fs (attempt %d/%d)",
                        delay,
                        attempt + 1,
                        policy.max_attempts,
                    )
                    await asyncio.sleep(delay)
                    continue
                out_headers, ctype = _filtered_headers(resp_headers)
                return Response(
                    content=error_body,
                    status_code=status,
                    headers=out_headers,
                    media_type=ctype,
                )

            # Success: reconstruct a non-streaming chat.completion.
            final_body = json.dumps(
                _reconstruct_chat_completion(meta, content, reasoning)
            ).encode("utf-8")
            self.recorder.record(
                {
                    **base_entry,
                    "attempt": attempt,
                    "max_attempts": policy.max_attempts,
                    "status": status,
                    "response_headers": resp_headers,
                    "response_body": _decode(final_body),
                    "diagnosis": None,
                    "duration": time.time() - started,
                }
            )
            return Response(
                content=final_body,
                status_code=200,
                media_type="application/json",
            )

        # Exhausted retries (all attempts errored) — defensive fallback.
        return Response(
            content=b'{"error": "upstream failure"}',
            status_code=502,
            media_type="application/json",
        )
