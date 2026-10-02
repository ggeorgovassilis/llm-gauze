"""Unit tests for the retry capability (failure classification + backoff).

Covers ``RetryableDetector`` (which exception categories and statuses are
retryable) and ``ExponentialBackoff`` (exponential growth, the maximum cap, and
full jitter). Both were untested before the test audit (#39, finding F2): a bug
here causes retry storms or silently swallows transient failures.
"""

import httpx
import pytest
from app.remediation.retry import ExponentialBackoff, RetryableDetector


def test_exponential_backoff_grows_exponentially():
    backoff = ExponentialBackoff(initial=0.5, base=2.0, maximum=30.0, jitter=False)
    assert backoff.delay(1) == 0.5
    assert backoff.delay(2) == 1.0
    assert backoff.delay(3) == 2.0
    assert backoff.delay(4) == 4.0


def test_exponential_backoff_caps_at_maximum():
    backoff = ExponentialBackoff(initial=1.0, base=10.0, maximum=25.0, jitter=False)
    # attempt 1 -> 1.0, attempt 2 -> 10.0, attempt 3 -> 100.0 capped at 25.0.
    assert backoff.delay(1) == 1.0
    assert backoff.delay(2) == 10.0
    assert backoff.delay(3) == 25.0
    assert backoff.delay(4) == 25.0


def test_exponential_backoff_jitter_samples_within_delay(monkeypatch):
    backoff = ExponentialBackoff(initial=2.0, base=2.0, maximum=30.0, jitter=True)
    # Full jitter draws from [0, unjittered_delay); attempt 1 -> 2.0.
    monkeypatch.setattr("app.remediation.retry.random.uniform", lambda lo, hi: 1.25)
    assert backoff.delay(1) == 1.25
    # The range's lower bound is 0.
    monkeypatch.setattr("app.remediation.retry.random.uniform", lambda lo, hi: 0.0)
    assert backoff.delay(1) == 0.0


def test_exponential_backoff_without_jitter_never_randomises(monkeypatch):
    backoff = ExponentialBackoff(initial=2.0, base=2.0, maximum=30.0, jitter=False)

    def _boom(lo, hi):
        pytest.fail("random.uniform must not be called when jitter is disabled")

    monkeypatch.setattr("app.remediation.retry.random.uniform", _boom)
    assert backoff.delay(2) == 4.0


def test_retryable_exception_categories():
    detector = RetryableDetector({500, 502, 503})
    # httpx.RequestError subclasses (network/timeout) are retryable.
    assert detector.diagnose_exception(httpx.ConnectError("connection refused")).retryable
    assert detector.diagnose_exception(httpx.ReadTimeout("timed out")).retryable
    # Raw socket OSError subclasses are retryable.
    assert detector.diagnose_exception(ConnectionResetError("reset")).retryable
    # asyncio-level TimeoutError is retryable.
    assert detector.diagnose_exception(TimeoutError("async timeout")).retryable


def test_non_transient_exception_is_not_retryable():
    detector = RetryableDetector({500, 502, 503})
    diagnosis = detector.diagnose_exception(ValueError("a bug in our code"))
    assert not diagnosis.retryable


def test_status_classification():
    detector = RetryableDetector({408, 429, 500})
    assert detector.diagnose_status(500).retryable
    assert detector.diagnose_status(429).retryable
    assert not detector.diagnose_status(200).retryable
    assert not detector.diagnose_status(404).retryable
