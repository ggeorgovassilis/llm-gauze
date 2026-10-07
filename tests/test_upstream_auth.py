"""Tests for ``Authorization`` pass-through to the upstream (#119).

The client's ``Authorization`` header must reach the upstream unchanged on
buffered, streaming and retried requests, be absent when the client sends none,
and never leak into logs or recordings.
"""

import asyncio
import json
import logging

import pytest
from app.proxy import Proxy, _redact_credentials
from app.recorder import Recorder
from mock_upstream import MockUpstream, chunk, make_request, settings_override

TOKEN = "s3cr3t-token-12345"
AUTH = f"Bearer {TOKEN}"

_OK = {"status": 200, "body": b'{"ok": true}'}
_FAIL = {"status": 500, "body": b'{"error": "boom"}'}
_STREAM_OK = {
    "chunks": [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "hello"}),
        chunk({}, finish_reason="stop"),
    ]
}


def _forward(specs, record_path, *, path="v1/completions", auth=AUTH, **overrides):
    """Drive ``Proxy.forward`` against a mock upstream; return it for inspection."""
    body = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
    client_headers = {"Authorization": auth} if auth is not None else {}
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            retry_max_attempts=3,
            retry_backoff_initial=0.0,
            retry_backoff_jitter=False,
            **overrides,
        ):
            proxy = Proxy(Recorder(record_path))

            async def run():
                request = await make_request(body, path=f"/{path}", headers=client_headers)
                return await proxy.forward(request, path)

            resp = asyncio.run(run())
    return resp, mock


def _auth_values(mock):
    return [h.get("authorization") for h in mock.request_headers]


def test_buffered_request_carries_header(tmp_path):
    resp, mock = _forward([_OK], tmp_path / "r.jsonl")
    assert resp.status_code == 200
    assert _auth_values(mock) == [AUTH]


def test_streaming_request_carries_header(tmp_path):
    resp, mock = _forward(
        [_STREAM_OK],
        tmp_path / "r.jsonl",
        path="v1/chat/completions",
        loop_detection_enabled=True,
    )
    assert resp.status_code == 200, resp.body
    assert _auth_values(mock) == [AUTH]


@pytest.mark.parametrize("base_path", ["/api", "/api/"])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_completion_preserves_api_prefix_and_auth(tmp_path, base_path, stream):
    async def run():
        with MockUpstream([_STREAM_OK]) as upstream:
            with settings_override(
                llm_base_url=upstream.url + base_path,
                loop_detection_enabled=True,
            ):
                proxy = Proxy(Recorder(tmp_path / "r.jsonl"))
                try:
                    request = await make_request(
                        json.dumps({"model": "test", "messages": [], "stream": stream}).encode(),
                        headers={"Authorization": AUTH},
                    )
                    response = await proxy.forward(request, "v1/chat/completions")
                finally:
                    await proxy.aclose()
            assert response.status_code == 200, response.body
            assert upstream.request_paths == ["/api/v1/chat/completions"]
            assert _auth_values(upstream) == [AUTH]
            assert upstream.json_bodies()[0]["stream"] is True

    asyncio.run(run())


def test_retried_buffered_requests_carry_header(tmp_path):
    resp, mock = _forward([_FAIL, _OK], tmp_path / "r.jsonl")
    assert resp.status_code == 200
    assert _auth_values(mock) == [AUTH, AUTH]


def test_retried_streaming_requests_carry_header(tmp_path):
    resp, mock = _forward(
        [_FAIL, _STREAM_OK],
        tmp_path / "r.jsonl",
        path="v1/chat/completions",
        loop_detection_enabled=True,
    )
    assert resp.status_code == 200, resp.body
    assert _auth_values(mock) == [AUTH, AUTH]


def test_no_header_when_client_sends_none(tmp_path):
    resp, mock = _forward([_OK], tmp_path / "r.jsonl", auth=None)
    assert resp.status_code == 200
    assert _auth_values(mock) == [None]


def test_token_not_logged_or_recorded(tmp_path, caplog):
    record_path = tmp_path / "r.jsonl"
    with caplog.at_level(logging.DEBUG):
        resp, _ = _forward([_OK], record_path)
    assert resp.status_code == 200
    assert TOKEN not in caplog.text
    assert TOKEN not in record_path.read_text()


def test_token_not_logged_or_recorded_when_streaming(tmp_path, caplog):
    record_path = tmp_path / "r.jsonl"
    with caplog.at_level(logging.DEBUG):
        resp, _ = _forward(
            [_STREAM_OK],
            record_path,
            path="v1/chat/completions",
            loop_detection_enabled=True,
        )
    assert resp.status_code == 200, resp.body
    assert TOKEN not in caplog.text
    assert TOKEN not in record_path.read_text()


@pytest.mark.parametrize("name", ["Authorization", "authorization", "Proxy-Authorization"])
def test_redact_credentials_masks_only_credentials(name):
    out = _redact_credentials({name: AUTH, "X-Custom": "yes"})
    assert out == {name: "[redacted]", "X-Custom": "yes"}
