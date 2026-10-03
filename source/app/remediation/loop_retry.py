"""Loop retry — re-submit a looped request with varied sampling.

A detected loop (thinking or output) is often a *fixed point* of the sampler:
the same deterministic reasoning cycle repeats because, given the same prompt
and the same sampling parameters, the model re-walks the same path. Retrying
the identical request reproduces the loop. Re-submitting with a higher
``temperature`` and stronger repeat penalties perturbs the trajectory enough to
break the cycle — without touching the prompt, so the task itself is unchanged.

This mirrors the #14 nudge rung in shape, but for loops rather than empty
turns:

* ``apply`` mutates a copy of the request body: a sampling parameter the
  client already submitted is nudged up by ``increment`` (0.1); one it did not
  submit is set to the configured fallback. Affects ``temperature``,
  ``repeat_penalty`` (llama.cpp native) plus ``presence_penalty`` /
  ``frequency_penalty`` (OpenAI). Pure and unit-testable without the gateway.
"""

from app.config import settings
from app.remediation.base import Remediation, Turn


class LoopRetryPolicy(Remediation):
    """Build the re-submission body for a looped request."""

    name = "loop_retry"

    def __init__(
        self,
        max_attempts: int | None = None,
        increment: float | None = None,
        temperature: float | None = None,
        repeat_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
    ) -> None:
        self.max_attempts = (
            max_attempts if max_attempts is not None else settings.loop_retry_max_attempts
        )
        self.increment = increment if increment is not None else settings.loop_retry_increment
        self.temperature = (
            temperature if temperature is not None else settings.loop_retry_temperature
        )
        self.repeat_penalty = (
            repeat_penalty if repeat_penalty is not None else settings.loop_retry_repeat_penalty
        )
        self.presence_penalty = (
            presence_penalty
            if presence_penalty is not None
            else settings.loop_retry_presence_penalty
        )
        self.frequency_penalty = (
            frequency_penalty
            if frequency_penalty is not None
            else settings.loop_retry_frequency_penalty
        )

    @classmethod
    def from_settings(cls) -> "LoopRetryPolicy":
        return cls()

    def applies(self, turn: Turn, request_body: dict) -> bool:
        """The ladder's trigger: a ``loop`` verdict (not a stall)."""
        return turn.verdict is not None and turn.verdict.kind == "loop"

    def apply(self, turn: Turn, request_body: dict) -> dict:
        """Return a copy of the request body with varied sampling parameters.

        Model-appropriate sampling is the client's and endpoint's domain, so
        llm-gauze never invents values: a parameter the client *did* submit is
        nudged upward by ``increment`` (0.1); one it did *not* submit is set to
        the configured fallback. ``repeat_penalty`` is a llama.cpp-native knob
        surfaced through the upstream's OpenAI-compatible API; setting it
        alongside the standard OpenAI ``presence_penalty``/``frequency_penalty``
        covers both common backends. The ``stream`` flag is left untouched —
        the gateway forces it on itself in ``_ensure_stream``.
        """
        body = dict(request_body)
        self._nudge(body, "temperature", self.temperature)
        self._nudge(body, "repeat_penalty", self.repeat_penalty)
        self._nudge(body, "presence_penalty", self.presence_penalty)
        self._nudge(body, "frequency_penalty", self.frequency_penalty)
        return body

    def _nudge(self, body: dict, key: str, fallback: float) -> None:
        """Raise a client-submitted value by ``increment``, else set fallback."""
        if key in body:
            body[key] = round(body[key] + self.increment, 4)
        else:
            body[key] = fallback
