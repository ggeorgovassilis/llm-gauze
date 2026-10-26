"""Shared pytest fixtures for the test suite.

* ``_restore_settings`` (autouse) snapshots the global ``settings`` singleton
  before each test and restores it afterwards, so a test that mutates a field
  cannot leak into the next one.
* ``mock_upstream`` is a factory returning a :class:`mock_upstream.MockUpstream`
  (see that module); each server it creates is stopped after the test.
* ``read_records`` decodes a JSONL recorder file into a list of records, shared
  by the record-schema tests.
* ``analysis`` imports ``scripts/analyse-prompt-preambles`` once per session;
  ``copilot_body`` and ``make_record`` build the synthetic request bodies and
  recorded exchanges used by the prompt-trimming tests.
"""

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any

import pytest
from app.config import settings
from mock_upstream import MockUpstream

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPT = ROOT / "scripts" / "analyse-prompt-preambles"
COPILOT_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "copilot_request_body.json"


@pytest.fixture(scope="session")
def analysis() -> Any:
    """Import the extensionless analysis script as a module."""
    loader = SourceFileLoader("analyse_prompt_preambles", str(ANALYSIS_SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture
def copilot_body() -> dict[str, Any]:
    """A fresh copy of the synthetic Copilot-shaped request body."""
    return json.loads(COPILOT_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def make_record():
    """Factory building a recorded exchange (request + response) from a body."""

    def _make(
        request_id: str,
        body: dict[str, Any],
        *,
        session: str = "session-1",
        timestamp: float = 1.0,
    ) -> dict[str, Any]:
        response = {"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        return {
            "request_id": request_id,
            "timestamp": timestamp,
            "request_headers": {"x-interaction-id": session},
            "request_body": json.dumps(body, separators=(",", ":")),
            "response_body": json.dumps(response),
        }

    return _make


@pytest.fixture(autouse=True)
def _restore_settings():
    """Snapshot and restore the settings singleton around every test."""
    snapshot = settings.model_dump()
    yield
    for key, value in snapshot.items():
        setattr(settings, key, value)


@pytest.fixture
def mock_upstream():
    """Factory for :class:`MockUpstream`; created servers are auto-stopped."""
    servers = []

    def _make(specs):
        server = MockUpstream(specs)
        servers.append(server)
        return server

    yield _make
    for server in servers:
        server.stop()


@pytest.fixture
def read_records():
    """Return a helper that decodes a JSONL recorder file into records."""

    def _read(path):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    return _read
