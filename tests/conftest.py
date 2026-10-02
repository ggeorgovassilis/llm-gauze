"""Shared pytest fixtures for the test suite.

* ``_restore_settings`` (autouse) snapshots the global ``settings`` singleton
  before each test and restores it afterwards, so a test that mutates a field
  cannot leak into the next one.
* ``mock_upstream`` is a factory returning a :class:`mock_upstream.MockUpstream`
  (see that module); each server it creates is stopped after the test.
"""

import pytest
from app.config import settings
from mock_upstream import MockUpstream


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
