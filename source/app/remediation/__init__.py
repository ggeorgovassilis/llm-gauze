"""Remediation pipeline.

Separates three concerns so new capabilities can be added without touching the
proxy:

1. **Data collection** — the `Recorder` persists every exchange/attempt.
2. **Understanding** — `Detector`s classify a failure into a `Diagnosis`.
3. **Action** — `Backoff`/`RetryPolicy` turn a `Diagnosis` into behaviour.

Later phases (sloppy-response cleanup, loop detection, context-window
detection, ...) implement these interfaces rather than editing the proxy.
"""

from app.remediation.base import (
    Backoff,
    ContentWatchdog,
    Detector,
    Diagnosis,
    Remediation,
    RetryPolicy,
    StreamVerdict,
    Transform,
    Turn,
)
from app.remediation.coast import CoastPolicy
from app.remediation.context import CONTEXT_WINDOW_CODE, ContextWindowDetector
from app.remediation.loop import ThinkingLoopDetector
from app.remediation.nudge import NudgePolicy
from app.remediation.overflow import MessageOverflowGuard
from app.remediation.retry import ExponentialBackoff, RetryableDetector
from app.remediation.runaway import RunawayReasoningDetector, RunawayReasoningPolicy
from app.remediation.stall import StallDetector
from app.remediation.think import ThinkContentGuard
from app.remediation.tool_call import ToolCallGuard

__all__ = [
    "Backoff",
    "Detector",
    "Diagnosis",
    "RetryPolicy",
    "ContentWatchdog",
    "StreamVerdict",
    "Remediation",
    "Transform",
    "Turn",
    "ExponentialBackoff",
    "RetryableDetector",
    "ThinkingLoopDetector",
    "StallDetector",
    "ContextWindowDetector",
    "CONTEXT_WINDOW_CODE",
    "ThinkContentGuard",
    "NudgePolicy",
    "CoastPolicy",
    "MessageOverflowGuard",
    "RunawayReasoningDetector",
    "RunawayReasoningPolicy",
    "ToolCallGuard",
]
