"""The verdict routing registry.

The streaming pipeline's content watchdogs raise a :class:`StreamVerdict`
whose ``kind`` is a :class:`VerdictKind`. The proxy used to switch on the
kind's *string* in several places — the abort message map, the abort-status
selection, and the telemetry increments — so adding a verdict kind meant
touching every site and a typo failed at runtime.

This module holds the single registry that maps each :class:`VerdictKind` to
its abort message, HTTP status, ``requests_total`` outcome label, and abort
counter. The proxy (and tests) look the verdict up here instead of switching
on its string value, so a new kind is "add a :class:`VerdictRoute`", not
"edit every comparison".
"""

from dataclasses import dataclass

from app.config import settings
from app.remediation.base import VerdictKind


@dataclass(frozen=True)
class VerdictRoute:
    """Response and telemetry routing for one :class:`VerdictKind`.

    ``abort_status_attr`` names a ``settings`` field read lazily (so test
    overrides of e.g. ``loop_abort_status`` are honoured) rather than
    capturing the value at import time.
    """

    kind: VerdictKind
    message: str
    outcome: str
    abort_metric: str
    abort_status_attr: str
    #: Whether the abort counter carries the ``stream`` label (loop does).
    stream_labeled: bool = False

    @property
    def abort_status(self) -> int:
        """The HTTP status for this verdict, read from the live settings."""
        return int(getattr(settings, self.abort_status_attr))


VERDICT_ROUTES: dict[VerdictKind, VerdictRoute] = {
    VerdictKind.LOOP: VerdictRoute(
        kind=VerdictKind.LOOP,
        message="The model entered a repetitive loop and the request was aborted.",
        outcome="loop_aborted",
        abort_metric="loop_aborts_total",
        abort_status_attr="loop_abort_status",
        stream_labeled=True,
    ),
    VerdictKind.STALLED: VerdictRoute(
        kind=VerdictKind.STALLED,
        message="The model stopped producing output and the request was aborted.",
        outcome="stalled",
        abort_metric="stall_aborts_total",
        abort_status_attr="stall_abort_status",
    ),
    VerdictKind.RUNAWAY_REASONING: VerdictRoute(
        kind=VerdictKind.RUNAWAY_REASONING,
        message=(
            "The model kept reasoning without producing an answer and the request was aborted."
        ),
        outcome="runaway_reasoning_aborted",
        abort_metric="runaway_reasoning_aborts_total",
        abort_status_attr="runaway_reasoning_abort_status",
    ),
}


def route_for(kind: VerdictKind | str) -> VerdictRoute:
    """Return the route for ``kind``; an unknown kind raises ``ValueError``.

    ``VerdictKind(kind)`` normalises a raw string so a stray untyped value
    still resolves to its route, while a genuinely unknown kind fails loudly
    instead of silently falling through to a default branch.
    """
    return VERDICT_ROUTES[VerdictKind(kind)]
