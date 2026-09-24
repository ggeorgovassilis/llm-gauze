"""Reverse-proxy forwarding to the upstream local LLM provider.

Every request is forwarded to the configured upstream, with each attempt
recorded for later analysis. Failures are routed through the remediation
pipeline: the understanding layer classifies them, and the action layer
(retry with backoff) decides whether and when to try again.
"""

import asyncio
import logging
import time
import traceback
import uuid

import httpx
from fastapi import Request
from fastapi.responses import Response

from app.config import settings
from app.recorder import Recorder
from app.remediation.base import RetryPolicy
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
