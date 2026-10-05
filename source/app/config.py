"""Application configuration, loaded from the environment / .env file.

Every setting is declared here with a ``description`` and validation
constraints (numeric ranges, enum-like HTTP status codes). ``.env.example`` and
``docs/configuration.md`` are generated from this module by
``scripts/generate_config.py`` — add a new setting here and re-run that script
rather than editing those files by hand.
"""

from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _field(
    default: Any,
    description: str,
    section: str,
    **constraints: Any,
) -> Any:
    """Declare a setting: default value, doc string, and validation.

    ``section`` groups the setting for the generated ``.env.example`` and
    ``docs/configuration.md``. Any extra keyword arguments are pydantic
    ``Field`` constraints (``gt``, ``ge``, ``le``, ``lt``, ``min_length``, …).
    """
    return Field(
        default=default,
        description=description,
        json_schema_extra={"section": section},
        **constraints,
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Upstream local LLM provider (OpenAI-compatible API).
    llm_base_url: str = _field(
        "http://host.docker.internal:14434",
        "Upstream local LLM (OpenAI-compatible API).",
        "Upstream & gateway",
        min_length=1,
    )

    # Complete header line sent on every upstream request, e.g.
    # "Authorization: Bearer <token>". A secret: kept as SecretStr so it never
    # shows up in repr/model_dump, logs, or validation errors.
    llm_bearer_header: SecretStr = _field(
        SecretStr(""),
        "Optional header line sent on every upstream request, e.g. "
        "`Authorization: Bearer <token>` (the full line, not just the token). "
        "This is a secret: keep it out of version control. Unset or empty sends no header.",
        "Upstream & gateway",
    )

    # A pydantic validator would echo the raw input in its error, so the header is
    # validated here instead and checked once at startup (see below).
    @property
    def upstream_auth_header(self) -> tuple[str, str] | None:
        """``(name, value)`` parsed from ``llm_bearer_header``, or None when unset."""
        line = self.llm_bearer_header.get_secret_value().strip()
        if not line:
            return None
        if "\r" in line or "\n" in line:
            raise ValueError("LLM_BEARER_HEADER must be a single line")
        name, sep, value = line.partition(":")
        if not sep or not name.strip() or not value.strip():
            raise ValueError("LLM_BEARER_HEADER must be a full header line: 'Name: value'")
        return name.strip(), value.strip()

    # Upstream read/write timeout, in seconds (how long to wait for a
    # response once connected — generation can be slow).
    request_timeout: float = _field(
        300.0,
        "Upstream read/write timeout in seconds once connected — generation can be slow.",
        "Upstream & gateway",
        gt=0,
    )

    # Upstream connection timeout, in seconds (how long to wait to establish
    # a TCP connection). Kept short so a dead/blackholed endpoint fails fast
    # instead of hanging until request_timeout.
    connect_timeout: float = _field(
        10.0,
        "Upstream connection timeout in seconds; kept short so a dead/blackholed "
        "endpoint fails fast.",
        "Upstream & gateway",
        gt=0,
    )

    # Where requests/responses are recorded.
    data_dir: str = _field(
        "/data",
        "Directory where exchanges are recorded.",
        "Upstream & gateway",
    )
    record_file: str = _field(
        "records.jsonl",
        "Recording filename (appended, rotated).",
        "Upstream & gateway",
        min_length=1,
    )

    # --- Logging & recording -----------------------------------------
    # Verbosity of the gateway's log stream. One of the standard Python
    # levels; DEBUG for troubleshooting, WARNING/ERROR for quiet production.
    log_level: str = _field(
        "INFO",
        "Log verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
        "Logging & recording",
    )

    # Format string applied to every log line (Python logging format syntax).
    log_format: str = _field(
        "%(asctime)s %(name)s %(levelname)s %(message)s",
        "Format string for log lines (Python logging format syntax).",
        "Logging & recording",
        min_length=1,
    )

    # Master switch for JSONL exchange recording. When false the recorder is
    # still constructed but writes nothing (and touches no files).
    recording_enabled: bool = _field(
        True,
        "Master switch for JSONL exchange recording.",
        "Logging & recording",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        level = str(value).strip().upper()
        if level not in allowed:
            raise ValueError(
                f"invalid log level {value!r}; expected one of {', '.join(sorted(allowed))}"
            )
        return level

    # --- Retries / backoff --------------------------------------------
    # Maximum number of attempts per upstream request (1 = no retry).
    retry_max_attempts: int = _field(
        3,
        "Maximum attempts per upstream request (1 = no retry).",
        "Retries / backoff",
        ge=1,
    )

    # Initial backoff delay, in seconds (before the first retry).
    retry_backoff_initial: float = _field(
        0.5,
        "Initial backoff delay, in seconds (before the first retry).",
        "Retries / backoff",
        ge=0,
    )

    # Exponential backoff base: delay = initial * base^(attempt-1).
    retry_backoff_base: float = _field(
        2.0,
        "Exponential backoff base: delay = initial * base^(attempt-1).",
        "Retries / backoff",
        ge=1,
    )

    # Upper bound on any single backoff delay, in seconds.
    retry_backoff_max: float = _field(
        30.0,
        "Upper bound on any single backoff delay, in seconds.",
        "Retries / backoff",
        gt=0,
    )

    # Apply full jitter to the computed delay.
    retry_backoff_jitter: bool = _field(
        True,
        "Apply full jitter to the computed delay.",
        "Retries / backoff",
    )

    # Comma-separated HTTP status codes that warrant a retry.
    retryable_status_codes: str = _field(
        "408,429,500,502,503,504",
        "Comma-separated HTTP status codes that warrant a retry.",
        "Retries / backoff",
    )

    @field_validator("retryable_status_codes")
    @classmethod
    def _validate_retryable_status_codes(cls, value: str) -> str:
        codes = [part.strip() for part in value.split(",") if part.strip()]
        if not codes:
            raise ValueError("must list at least one HTTP status code")
        for code in codes:
            if not code.isdigit() or not 100 <= int(code) <= 599:
                raise ValueError(f"invalid HTTP status code {code!r}")
        return ",".join(codes)

    # --- Loop detection (thinking/output loops) ----------------------
    # Master switch for the loop-detection feature.
    loop_detection_enabled: bool = _field(
        True,
        "Master switch for loop detection.",
        "Loop detection",
    )

    # Size of the sliding output window, in bytes, over which repetition is
    # measured. Approximates an 8192-token output window at ~4 bytes/token.
    loop_window_bytes: int = _field(
        32768,
        "Sliding output window (bytes) over which repetition is measured; "
        "approximates an 8192-token window at ~4 bytes/token.",
        "Loop detection",
        gt=0,
    )

    # Fraction of the output window that must be produced before loop
    # detection arms. Below this the stream is still "warming up" — e.g. a
    # model enumerating a short list of near-identical tool results — and is
    # not treated as a loop. A genuine loop keeps emitting repetitive output
    # and is caught once it crosses this threshold.
    loop_min_output_fraction: float = _field(
        0.25,
        "Fraction of the output window that must be produced before loop detection arms.",
        "Loop detection",
        ge=0,
        le=1,
    )

    # Compression ratio below which the window is considered low-entropy
    # (repetitive). Lower = more repetition. Tuned low (0.15) so genuine varied
    # reasoning (~0.24+) stays above it while a verbatim loop (~0.004) trips.
    loop_compression_ratio: float = _field(
        0.15,
        "Compression ratio below which the window is considered low-entropy (repetitive).",
        "Loop detection",
        ge=0,
        le=1,
    )

    # HTTP status returned to the client when a loop is detected and every
    # remediation re-submission (see loop-retry below) has also looped.
    loop_abort_status: int = _field(
        502,
        "HTTP status returned when a loop is detected and not remediated.",
        "Loop detection",
        ge=100,
        le=599,
    )

    # --- Loop remediation (re-submit with varied sampling) -----------
    # When a loop is detected, rather than aborting immediately the request
    # is re-submitted with a higher temperature and repeat penalties. A loop
    # is often a fixed point of the sampler: the same deterministic reasoning
    # cycle repeats verbatim. Perturbing the sampling breaks the cycle without
    # changing the prompt. Only *loop* verdicts are remediated — a *stall*
    # (silent model) is a different failure and is aborted outright.
    #
    # Master switch for the loop-retry feature.
    loop_retry_enabled: bool = _field(
        True,
        "Master switch for re-submitting a looped request with varied sampling.",
        "Loop remediation",
    )

    # Maximum number of loop re-submissions before giving up and aborting.
    loop_retry_max_attempts: int = _field(
        5,
        "Maximum number of loop re-submissions before giving up.",
        "Loop remediation",
        ge=0,
    )

    # By how much to *increase* a sampling parameter the client already
    # submitted. Model-appropriate sampling is the client's and endpoint's
    # domain, so llm-gauze never invents values — it only nudges the client's
    # own values upward by this delta.
    loop_retry_increment: float = _field(
        0.1,
        "Amount added to a sampling parameter the client already submitted.",
        "Loop remediation",
        ge=0,
    )

    # Fallback sampling parameters used *only* when the client did not submit
    # that parameter at all. ``temperature`` (more variety), ``repeat_penalty``
    # (llama.cpp native) and ``presence_penalty``/``frequency_penalty``
    # (OpenAI) discourage repeating the tokens/phrases that produced the loop.
    loop_retry_temperature: float = _field(
        1.2,
        "Fallback temperature when the client submitted none.",
        "Loop remediation",
        gt=0,
    )
    loop_retry_repeat_penalty: float = _field(
        1.2,
        "Fallback llama.cpp repeat penalty when the client submitted none.",
        "Loop remediation",
        gt=0,
    )
    loop_retry_presence_penalty: float = _field(
        0.3,
        "Fallback OpenAI presence penalty when the client submitted none.",
        "Loop remediation",
        ge=0,
    )
    loop_retry_frequency_penalty: float = _field(
        0.3,
        "Fallback OpenAI frequency penalty when the client submitted none.",
        "Loop remediation",
        ge=0,
    )

    # --- Stall detection (silently hung streams) ---------------------
    # Master switch for the stalled-stream detection feature.
    stall_detection_enabled: bool = _field(
        True,
        "Master switch for stall detection.",
        "Stall detection",
    )

    # Max wall-clock seconds to wait for the FIRST content-bearing token
    # (thinking or response delta) after the upstream returns headers. Prefill
    # on a large prompt is legitimately slow, so this is generous.
    stall_ttft_seconds: float = _field(
        120.0,
        "Max seconds to wait for the first content-bearing token after headers.",
        "Stall detection",
        gt=0,
    )

    # Max wall-clock seconds between content-bearing tokens once generation has
    # started. SSE keepalives/comments do NOT reset this — only real tokens do.
    stall_gap_seconds: float = _field(
        60.0,
        "Max seconds between content-bearing tokens once generation has started.",
        "Stall detection",
        gt=0,
    )

    # HTTP status returned to the client when a stall is detected.
    stall_abort_status: int = _field(
        502,
        "HTTP status returned when a stall is detected.",
        "Stall detection",
        ge=100,
        le=599,
    )

    # --- Context-window overflow -------------------------------------
    # Master switch for recognising the upstream's context-window-fill error
    # and failing fast (instead of retrying a doomed request).
    context_window_detection_enabled: bool = _field(
        True,
        "Master switch for context-window overflow detection.",
        "Context-window overflow",
    )

    # Comma-separated substrings (matched case-insensitively) that the upstream
    # emits when the context window fills. Kept exact to avoid false positives;
    # override for other providers/versions.
    context_window_markers: str = _field(
        "exceeded the context window,"
        "context window exceeded,"
        "context length exceeded,"
        "maximum context length,"
        "prompt is too long,"
        "prompt is longer than the context,"
        "input is too long,"
        "ran out of context,"
        "exceeds the available context size",
        "Comma-separated substrings (case-insensitive) signalling a full context window.",
        "Context-window overflow",
        min_length=1,
    )

    # HTTP status returned to the client when a context-window overflow is
    # recognised but no upstream body is available to pass through (e.g. the
    # overflow arrived as an exception message). When the upstream returned a
    # body, its own status and body are forwarded verbatim instead.
    context_window_abort_status: int = _field(
        413,
        "HTTP status returned when an overflow is recognised but no upstream "
        "body is available to pass through.",
        "Context-window overflow",
        ge=100,
        le=599,
    )

    # --- Think-tag cleanup -------------------------------------------
    # Master switch for relocating leaked thinking tags out of the visible
    # content and guaranteeing a non-empty response to the client.
    think_cleanup_enabled: bool = _field(
        True,
        "Master switch for think-tag cleanup.",
        "Think-tag cleanup",
    )

    # Comma-separated tag names (matched case-insensitively) whose inner text
    # is relocated out of ``content`` into ``reasoning_content`` — e.g.
    # ``<think>…</think>``, ``<reasoning>…</reasoning>``. Exact tag names only;
    # no fuzzy heuristics.
    think_tags: str = _field(
        "think,thinking,reasoning",
        "Comma-separated tag names (case-insensitive) whose inner text is "
        "relocated into reasoning_content.",
        "Think-tag cleanup",
        min_length=1,
    )

    # Placeholder message emitted as the visible ``content`` when the model
    # produced only a thinking tag and nothing else. Kept short and explicit so
    # the client receives a non-empty turn instead of aborting the flow.
    think_empty_response_placeholder: str = _field(
        "The model replied inside a thinking tag; see reasoning_content.",
        "Placeholder emitted as visible content when the model produced only a thinking tag.",
        "Think-tag cleanup",
    )

    # --- Nudge (re-prompt empty-text turns) --------------------------
    # Master switch for re-submitting a turn that produced only reasoning
    # (finish_reason=stop, no content, no tool calls) with a short re-prompt.
    think_nudge_enabled: bool = _field(
        True,
        "Master switch for re-prompting a turn that produced only reasoning.",
        "Nudge (re-prompt empty turns)",
    )

    # Nudge text appended as a ``user`` message on the re-submitted request.
    think_nudge_text: str = _field(
        "Your previous reply contained no visible text and no tool call. "
        "Reply with a visible answer, or call a tool if the task requires one.",
        "Nudge text appended as a user message on the re-submitted request.",
        "Nudge (re-prompt empty turns)",
    )

    # Maximum number of nudge re-submissions before falling back to the
    # placeholder floor (see ``think_empty_response_placeholder``).
    think_nudge_max_attempts: int = _field(
        2,
        "Max nudge re-submissions before falling back to the placeholder floor.",
        "Nudge (re-prompt empty turns)",
        ge=0,
    )

    # --- Coast detection (announced-but-absent tool call) -------------
    # Master switch for re-submitting a turn that produced non-empty visible
    # content but no tool call, even though a tool call was possible (the
    # request carried ``tools`` and the model had already been driving a tool
    # loop) and the model's reasoning collapsed to be byte-identical with its
    # visible content. This is the "silent workflow stop" described in #16:
    # the model regurgitated its status line instead of generating the call.
    coast_detection_enabled: bool = _field(
        True,
        "Master switch for coast detection.",
        "Coast detection",
    )

    # Re-prompt text appended as a ``user`` message on the re-submitted
    # request. The coasted assistant turn is replayed immediately before it so
    # the re-prompt refers to something the model actually said.
    coast_nudge_text: str = _field(
        "You announced a tool call but did not make one. "
        "Call the tool now, or if the task is complete, say so explicitly.",
        "Re-prompt text appended as a user message on the re-submitted request.",
        "Coast detection",
    )

    # Maximum number of coast re-submissions before giving up and returning
    # the coasted turn as-is (visible content, no tool call, logged outcome).
    coast_max_attempts: int = _field(
        2,
        "Max coast re-submissions before returning the coasted turn as-is.",
        "Coast detection",
        ge=0,
    )

    # --- Runaway-reasoning detection (thinks endlessly, never answers) ---
    # Master switch for flagging a turn that keeps emitting reasoning tokens
    # while never producing visible content — the model "thinks endlessly" and
    # either trips a proactive token budget or exhausts the output window with
    # nothing to show (see #17). Distinct from loop (repetitive) and stall
    # (silent): here tokens flow continuously but never become an answer.
    runaway_reasoning_enabled: bool = _field(
        True,
        "Master switch for runaway-reasoning detection.",
        "Runaway-reasoning detection",
    )

    # Reasoning-token budget above which a turn is flagged as runaway, provided
    # no content (or tool call) has appeared yet. Token count is approximated
    # at ~4 characters/token (see the loop detector's window comment).
    runaway_reasoning_token_threshold: int = _field(
        2000,
        "Reasoning-token budget (~4 chars/token) above which a turn is flagged, "
        "provided no content has appeared.",
        "Runaway-reasoning detection",
        gt=0,
    )

    # Re-prompt text appended as a ``user`` message on the re-submitted
    # request, telling the model to stop analysing and answer now.
    runaway_reasoning_nudge_text: str = _field(
        "Stop thinking and produce your final answer now, without further "
        "analysis. If the task requires a tool call, make it.",
        "Re-prompt text appended as a user message on the re-submitted request.",
        "Runaway-reasoning detection",
    )

    # Maximum number of runaway re-submissions before aborting the request.
    runaway_reasoning_max_attempts: int = _field(
        2,
        "Max runaway re-submissions before aborting the request.",
        "Runaway-reasoning detection",
        ge=0,
    )

    # HTTP status returned to the client when a runaway turn is detected and
    # every re-submission has also run away.
    runaway_reasoning_abort_status: int = _field(
        502,
        "HTTP status returned when a runaway turn is detected and every "
        "re-submission also ran away.",
        "Runaway-reasoning detection",
        ge=100,
        le=599,
    )

    # --- Message-overflow protection (oversized tool results) ---------
    # Master switch for warning (and optionally truncating) oversized
    # ``role: "tool"`` results before forwarding the request upstream. A single
    # huge tool result can silently eat the model's context window; this guard
    # flags it and reclaims the context (see #15).
    message_overflow_enabled: bool = _field(
        True,
        "Master switch for message-overflow protection.",
        "Message overflow",
    )

    # Size threshold, in characters, above which a tool-result message is
    # treated as oversized. Characters (not bytes) so the trigger and the
    # truncation prefix share one unit; for ASCII they coincide with bytes.
    message_overflow_threshold: int = _field(
        4096,
        "Size threshold, in characters, above which a tool-result message is treated as oversized.",
        "Message overflow",
        gt=0,
    )

    # Whether to truncate oversized tool results to a bounded prefix (the
    # first line, capped at the threshold). When False the full content is
    # kept and only the warning is prepended — rarely useful, as it just adds
    # tokens to a message the model already sees.
    message_overflow_truncate: bool = _field(
        True,
        "Truncate oversized tool results to a bounded prefix.",
        "Message overflow",
    )

    # Warning text prepended to an oversized tool result, telling the model
    # the content was too large and to find a workaround.
    message_overflow_warning: str = _field(
        "This tool result is very large and may exceed the model's context "
        "window; the full result may have been truncated. Find a workaround "
        "rather than relying on the complete result.",
        "Warning text prepended to an oversized tool result.",
        "Message overflow",
    )

    # --- Tool-call syntax enforcement --------------------------------
    # Master switch for validating assembled tool calls: repair truncated
    # ``function.arguments`` where deterministic, flag the rest without
    # crashing the request.
    tool_call_guard_enabled: bool = _field(
        True,
        "Master switch for tool-call validation/repair.",
        "Tool-call syntax enforcement",
    )


settings = Settings()
settings.upstream_auth_header  # fail fast on a malformed LLM_BEARER_HEADER
