"""Unit tests for the loop-retry policy (varied-sampling re-submission)."""

import sys

sys.path.insert(0, "source")

from app.remediation.base import Turn
from app.remediation.loop_retry import LoopRetryPolicy

# Shared constructor kwargs for the three tests that use identical penalties.
_DEFAULT_KWARGS = dict(
    max_attempts=2,
    increment=0.1,
    temperature=1.2,
    repeat_penalty=1.2,
    presence_penalty=0.3,
    frequency_penalty=0.3,
)


def test_apply_sets_fallback_when_client_omits():
    """A parameter the client did not submit is hard-set to the fallback."""
    policy = LoopRetryPolicy(
        max_attempts=2,
        increment=0.1,
        temperature=1.2,
        repeat_penalty=1.3,
        presence_penalty=0.4,
        frequency_penalty=0.5,
    )
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    out = policy.apply(Turn(), body)
    assert out["temperature"] == 1.2, out
    assert out["repeat_penalty"] == 1.3, out
    assert out["presence_penalty"] == 0.4, out
    assert out["frequency_penalty"] == 0.5, out
    # stream flag preserved for _ensure_stream
    assert out["stream"] is True, out


def test_apply_bumps_client_submitted_values():
    """A parameter the client submitted is increased by the increment."""
    policy = LoopRetryPolicy(**_DEFAULT_KWARGS)
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "repeat_penalty": 1.1,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
    }
    out = policy.apply(Turn(), body)
    assert out["temperature"] == 0.8, out
    assert out["repeat_penalty"] == 1.2, out
    assert out["presence_penalty"] == 0.3, out
    assert out["frequency_penalty"] == 0.3, out


def test_apply_does_not_mutate_input():
    policy = LoopRetryPolicy(**_DEFAULT_KWARGS)
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    policy.apply(Turn(), body)
    assert body["temperature"] == 0.2, body
    assert "repeat_penalty" not in body, body


def test_apply_mixed_fallback_and_bump():
    """Submitting some parameters bumps them; the rest fall back."""
    policy = LoopRetryPolicy(**_DEFAULT_KWARGS)
    body = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
    }
    out = policy.apply(Turn(), body)
    # submitted -> bumped
    assert out["temperature"] == 0.1, out
    # omitted -> fallback
    assert out["repeat_penalty"] == 1.2, out
    assert body["temperature"] == 0.0, body
