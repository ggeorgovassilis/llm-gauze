"""Wall-clock deadlines for upstream exchanges.

Two deadlines bound an upstream exchange independently of httpx's per-read
timeout (which a trickling upstream defeats):

* **header deadline** — a connection is accepted but no response headers
  arrive; headers must arrive quickly regardless of how slow generation later
  proves to be.
* **total deadline** — the whole per-attempt exchange (connect → headers →
  stream/body) is bounded by a generous wall-clock backstop.

Both surface as :class:`~app.remediation.deadlines.UpstreamDeadlineError`
subclasses carrying a distinct ``reason`` (``header_timeout`` /
``total_deadline``) so the recorder and telemetry can tell *why* an attempt
died. They are plain exceptions (not ``asyncio.TimeoutError``) so a nested
``asyncio.wait_for`` cannot mislabel one deadline as the other; the retry
detector recognises them explicitly and keeps them retryable.
"""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar("_T")


class UpstreamDeadlineError(Exception):
    """Base class for an upstream wall-clock deadline firing.

    Carries :attr:`reason` (``header_timeout`` or ``total_deadline``) so the
    retry classifier and the recorder can record a distinct diagnosis.
    """

    reason = "deadline"


class HeaderTimeoutError(UpstreamDeadlineError):
    """No upstream response headers arrived within the header deadline."""

    reason = "header_timeout"


class TotalDeadlineError(UpstreamDeadlineError):
    """The whole upstream exchange exceeded the total deadline."""

    reason = "total_deadline"


async def wait_for_header(coro: Awaitable[_T], timeout: float) -> _T:
    """Await ``coro``, converting a header-phase timeout to ``HeaderTimeoutError``."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise HeaderTimeoutError(f"no response headers within {timeout:g}s") from exc


async def wait_for_total(coro: Awaitable[_T], timeout: float) -> _T:
    """Await ``coro``, converting a total-deadline timeout to ``TotalDeadlineError``."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TotalDeadlineError(f"total exchange deadline {timeout:g}s exceeded") from exc
