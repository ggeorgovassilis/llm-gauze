"""Endpoint tests for the gateway's health and metrics routes.

Drives the real FastAPI app (``app.main``) through the in-process test client so
the observability surface — ``/health`` and ``/metrics`` — is covered without a
live upstream. ``main.py`` was at 0% coverage before these tests (test audit #39,
finding F1).
"""

from app.config import settings
from app.main import app
from fastapi.testclient import TestClient


def test_health_reports_ok_and_upstream():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        # Read the live setting rather than a hardcoded default: integration
        # tests elsewhere repoint `settings.llm_base_url` at their mock upstream,
        # so the endpoint must echo whatever is currently configured.
        assert resp.json() == {"status": "ok", "upstream": settings.llm_base_url}


def test_metrics_returns_json_when_accept_is_json():
    with TestClient(app) as client:
        resp = client.get("/metrics", headers={"accept": "application/json"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        body = resp.json()
        # Snapshot shape, independent of the live counter values.
        assert "uptime_seconds" in body
        assert "counters" in body
        assert "latency" in body
        assert "upstream_down" in body
        assert "upstream_down_since" in body


def test_metrics_returns_prometheus_by_default():
    with TestClient(app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        text = resp.text
        # Help/type metadata is emitted for every registered metric.
        assert "# HELP requests_total" in text
        assert "# TYPE requests_total counter" in text
        assert "# TYPE upstream_latency_seconds histogram" in text
        assert "# TYPE upstream_down gauge" in text
        # The histogram's +Inf bucket and the gauge are always present.
        assert 'upstream_latency_seconds_bucket{le="+Inf"}' in text
        assert "upstream_down " in text
