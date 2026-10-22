"""Tests for the header-phase and total-exchange wall-clock deadlines.

Two acceptance criteria from #148 are covered here:

* a connection that is accepted but never sends response headers aborts after
  the header deadline and is recorded/telemetered as ``header_timeout``;
* a request exceeding the total deadline aborts as ``total_deadline`` even when
  it is trickling, and disabling the total deadline restores unbounded-total
  behaviour.
"""

import asyncio
import json

import pytest
from app.proxy import Proxy
from app.recorder import Recorder
from app.remediation.deadlines import (
    HeaderTimeoutError,
    TotalDeadlineError,
    wait_for_header,
    wait_for_total,
)
from app.telemetry import telemetry
from mock_upstream import MockUpstream, chunk, settings_override, sse

_chunk = chunk
_sse = sse


def _run_streaming(tmp_path, mock, *, body, overrides):
    with settings_override(llm_base_url=mock.url, **overrides):
        recorder = Recorder(tmp_path / "deadlines.jsonl")
        proxy = Proxy(recorder)

        async def run():
            return await proxy._forward_streaming(
                request_id="deadlines-itest",
                method="POST",
                url="/v1/chat/completions",
                query="",
                body=body,
                headers={"content-type": "application/json"},
                base_entry={
                    "request_id": "deadlines-itest",
                    "method": "POST",
                    "path": "/v1/chat/completions",
                },
            )

        return asyncio.run(run())


def _chat_body() -> bytes:
    return json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()


def _records(tmp_path):
    return [json.loads(line) for line in open(tmp_path / "deadlines.jsonl") if line.strip()]


def _error_counter() -> dict:
    return telemetry.snapshot()["counters"].get("upstream_errors_total", {})


def test_wait_for_header_converts_timeout():
    async def never():
        await asyncio.sleep(10)

    with pytest.raises(HeaderTimeoutError) as exc:
        asyncio.run(wait_for_header(never(), 0.05))
    assert exc.value.reason == "header_timeout"


def test_wait_for_total_converts_timeout():
    async def never():
        await asyncio.sleep(10)

    with pytest.raises(TotalDeadlineError) as exc:
        asyncio.run(wait_for_total(never(), 0.05))
    assert exc.value.reason == "total_deadline"


def test_wait_for_helpers_pass_fast_coroutine_through():
    async def fast():
        return "ok"

    assert asyncio.run(wait_for_header(fast(), 1.0)) == "ok"
    assert asyncio.run(wait_for_total(fast(), 1.0)) == "ok"


def test_header_timeout_is_retried_and_recorded(tmp_path):
    """Headers never arrive: each attempt hits the header deadline and is retried."""
    before = _error_counter()
    spec = {"header_delay": 0.6, "steps": [(0.0, _sse(_chunk({"content": "hi"})))]}
    with MockUpstream([spec]) as mock:
        resp = _run_streaming(
            tmp_path,
            mock,
            body=_chat_body(),
            overrides={
                "header_timeout_enabled": True,
                "header_timeout_seconds": 0.3,
                "total_deadline_enabled": False,
                "retry_max_attempts": 2,
                "retry_backoff_initial": 0.01,
                "retry_backoff_jitter": False,
            },
        )
        assert resp.status_code == 502, (resp.status_code, resp.body)
        assert mock.request_count == 2, mock.request_count

    errors = [r for r in _records(tmp_path) if r.get("error")]
    assert len(errors) == 2, errors
    for entry in errors:
        assert entry["error"]["type"] == "header_timeout", entry
        assert entry["diagnosis"]["reason"] == "header_timeout", entry
        assert entry["diagnosis"]["retryable"] is True, entry

    after = _error_counter()
    assert after.get('type="header_timeout"', 0) > before.get('type="header_timeout"', 0)


def test_total_deadline_aborts_trickling_stream(tmp_path):
    """A stream that trickles past the total deadline aborts as total_deadline."""
    before = _error_counter()
    # ~50 tokens/s (well above the stall floor) for ~1.2s — the total deadline
    # must fire even though the stream is not stalled.
    steps = [(0.02, _sse(_chunk({"content": "word"}))) for _ in range(60)]
    with MockUpstream([{"steps": steps}]) as mock:
        resp = _run_streaming(
            tmp_path,
            mock,
            body=_chat_body(),
            overrides={
                "header_timeout_enabled": False,
                "total_deadline_enabled": True,
                "total_deadline_seconds": 0.4,
                "retry_max_attempts": 1,
            },
        )
        assert resp.status_code == 502, (resp.status_code, resp.body)

    errors = [r for r in _records(tmp_path) if r.get("error")]
    assert len(errors) == 1, errors
    assert errors[0]["error"]["type"] == "total_deadline", errors[0]
    assert errors[0]["diagnosis"]["reason"] == "total_deadline", errors[0]
    assert errors[0]["diagnosis"]["retryable"] is True, errors[0]

    after = _error_counter()
    assert after.get('type="total_deadline"', 0) > before.get('type="total_deadline"', 0)


def test_total_deadline_disabled_is_unbounded(tmp_path):
    """With the total deadline off, a stream longer than the (unused) bound succeeds."""
    steps = [(0.02, _sse(_chunk({"content": "word"}))) for _ in range(12)] + [
        (0.0, _sse(_chunk({}, finish_reason="stop")))
    ]
    with MockUpstream([{"steps": steps}]) as mock:
        resp = _run_streaming(
            tmp_path,
            mock,
            body=_chat_body(),
            overrides={
                "header_timeout_enabled": False,
                "total_deadline_enabled": False,
                "total_deadline_seconds": 0.05,  # deliberately tiny, but disabled
                "retry_max_attempts": 1,
            },
        )
        assert resp.status_code == 200, (resp.status_code, resp.body)
        data = json.loads(resp.body)
        assert data["object"] == "chat.completion", data
