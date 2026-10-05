"""Tests for upstream bearer authentication (``LLM_BEARER_HEADER``, #119).

The configured header line must reach the upstream on buffered, streaming and
retried requests, be absent when unset, and never leak into logs or recordings.
"""

import asyncio
import json
import logging

import pytest
from app.config import Settings
from app.proxy import Proxy, _with_upstream_auth
from app.recorder import Recorder
from mock_upstream import MockUpstream, chunk, make_request, settings_override
from pydantic import SecretStr, ValidationError

TOKEN = "s3cr3t-token-12345"
HEADER = f"Authorization: Bearer {TOKEN}"
EXPECTED = f"Bearer {TOKEN}"

_OK = {"status": 200, "body": b'{"ok": true}'}
_FAIL = {"status": 500, "body": b'{"error": "boom"}'}
_STREAM_OK = {
    "chunks": [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "hello"}),
        chunk({}, finish_reason="stop"),
    ]
}


def _forward(specs, record_path, *, path="v1/completions", bearer=HEADER, **overrides):
    """Drive ``Proxy.forward`` against a mock upstream; return it for inspection."""
    body = json.dumps({"model": "test", "messages": [{"role": "user", "content": "hi"}]}).encode()
    with MockUpstream(specs) as mock:
        with settings_override(
            llm_base_url=mock.url,
            llm_bearer_header=SecretStr(bearer),
            retry_max_attempts=3,
            retry_backoff_initial=0.0,
            retry_backoff_jitter=False,
            **overrides,
        ):
            proxy = Proxy(Recorder(record_path))

            async def run():
                request = await make_request(body, path=f"/{path}")
                return await proxy.forward(request, path)

            resp = asyncio.run(run())
    return resp, mock


def _auth_values(mock):
    return [h.get("authorization") for h in mock.request_headers]


def test_buffered_request_carries_header(tmp_path):
    resp, mock = _forward([_OK], tmp_path / "r.jsonl")
    assert resp.status_code == 200
    assert _auth_values(mock) == [EXPECTED]


def test_streaming_request_carries_header(tmp_path):
    resp, mock = _forward(
        [_STREAM_OK],
        tmp_path / "r.jsonl",
        path="v1/chat/completions",
        loop_detection_enabled=True,
    )
    assert resp.status_code == 200, resp.body
    assert _auth_values(mock) == [EXPECTED]


def test_retried_buffered_requests_carry_header(tmp_path):
    resp, mock = _forward([_FAIL, _OK], tmp_path / "r.jsonl")
    assert resp.status_code == 200
    assert _auth_values(mock) == [EXPECTED, EXPECTED]


def test_retried_streaming_requests_carry_header(tmp_path):
    resp, mock = _forward(
        [_FAIL, _STREAM_OK],
        tmp_path / "r.jsonl",
        path="v1/chat/completions",
        loop_detection_enabled=True,
    )
    assert resp.status_code == 200, resp.body
    assert _auth_values(mock) == [EXPECTED, EXPECTED]


@pytest.mark.parametrize("unset", ["", "   "])
def test_no_header_when_unset(tmp_path, unset):
    resp, mock = _forward([_OK], tmp_path / "r.jsonl", bearer=unset)
    assert resp.status_code == 200
    assert _auth_values(mock) == [None]


def test_token_not_logged_or_recorded(tmp_path, caplog):
    record_path = tmp_path / "r.jsonl"
    with caplog.at_level(logging.DEBUG):
        resp, _ = _forward([_OK], record_path)
    assert resp.status_code == 200
    assert TOKEN not in caplog.text
    assert TOKEN not in record_path.read_text()


def test_with_upstream_auth_replaces_client_header_only():
    client = {"Authorization": "Bearer client", "X-Custom": "yes"}
    with settings_override(llm_bearer_header=SecretStr(HEADER)):
        out = _with_upstream_auth(client)
    assert out == {"X-Custom": "yes", "Authorization": EXPECTED}
    assert client == {"Authorization": "Bearer client", "X-Custom": "yes"}


def test_with_upstream_auth_unset_leaves_headers_untouched():
    client = {"Authorization": "Bearer client"}
    with settings_override(llm_bearer_header=SecretStr("")):
        assert _with_upstream_auth(client) == client


# --- configuration -----------------------------------------------------


def test_bearer_header_defaults_to_empty():
    assert Settings.model_validate({}).llm_bearer_header.get_secret_value() == ""


def test_bearer_header_is_not_exposed_in_repr_or_dump():
    cfg = Settings.model_validate({"llm_bearer_header": HEADER})
    assert TOKEN not in repr(cfg)
    assert TOKEN not in str(cfg.model_dump())


@pytest.mark.parametrize("bad", ["no-colon", ": v4lue", "Authorization:", "A: b\r\nX: y"])
def test_malformed_bearer_header_rejected_without_leaking_value(bad):
    with pytest.raises(ValidationError) as excinfo:
        Settings.model_validate({"llm_bearer_header": bad})
    assert bad not in str(excinfo.value)
