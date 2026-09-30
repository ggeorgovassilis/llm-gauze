"""Application configuration, loaded from the environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Upstream local LLM provider (OpenAI-compatible API).
    llm_base_url: str = "http://host.docker.internal:14434"

    # Gateway listen address/port.
    host: str = "0.0.0.0"
    port: int = 8000

    # Upstream read/write timeout, in seconds (how long to wait for a
    # response once connected — generation can be slow).
    request_timeout: float = 300.0

    # Upstream connection timeout, in seconds (how long to wait to establish
    # a TCP connection). Kept short so a dead/blackholed endpoint fails fast
    # instead of hanging until request_timeout.
    connect_timeout: float = 10.0

    # Where requests/responses are recorded.
    data_dir: str = "/data"
    record_file: str = "records.jsonl"

    # --- Retries / backoff --------------------------------------------
    # Maximum number of attempts per upstream request (1 = no retry).
    retry_max_attempts: int = 3

    # Initial backoff delay, in seconds (before the first retry).
    retry_backoff_initial: float = 0.5

    # Exponential backoff base: delay = initial * base^(attempt-1).
    retry_backoff_base: float = 2.0

    # Upper bound on any single backoff delay, in seconds.
    retry_backoff_max: float = 30.0

    # Apply full jitter to the computed delay.
    retry_backoff_jitter: bool = True

    # Comma-separated HTTP status codes that warrant a retry.
    retryable_status_codes: str = "408,429,500,502,503,504"

    # --- Loop detection (thinking/output loops) ----------------------
    # Master switch for the loop-detection feature.
    loop_detection_enabled: bool = True

    # Size of the sliding output window, in bytes, over which repetition is
    # measured. Approximates an 8192-token output window at ~4 bytes/token.
    loop_window_bytes: int = 32768

    # Fraction of the output window that must be produced before loop
    # detection arms. Below this the stream is still "warming up" — e.g. a
    # model enumerating a short list of near-identical tool results — and is
    # not treated as a loop. A genuine loop keeps emitting repetitive output
    # and is caught once it crosses this threshold.
    loop_min_output_fraction: float = 0.25

    # Compression ratio below which the window is considered low-entropy
    # (repetitive). Lower = more repetition. Tuned low (0.15) so genuine varied
    # reasoning (~0.24+) stays above it while a verbatim loop (~0.004) trips.
    loop_compression_ratio: float = 0.15

    # HTTP status returned to the client when a loop is detected and every
    # remediation re-submission (see loop-retry below) has also looped.
    loop_abort_status: int = 502

    # --- Loop remediation (re-submit with varied sampling) -----------
    # When a loop is detected, rather than aborting immediately the request
    # is re-submitted with a higher temperature and repeat penalties. A loop
    # is often a fixed point of the sampler: the same deterministic reasoning
    # cycle repeats verbatim. Perturbing the sampling breaks the cycle without
    # changing the prompt. Only *loop* verdicts are remediated — a *stall*
    # (silent model) is a different failure and is aborted outright.
    #
    # Master switch for the loop-retry feature.
    loop_retry_enabled: bool = True

    # Maximum number of loop re-submissions before giving up and aborting.
    loop_retry_max_attempts: int = 5

    # By how much to *increase* a sampling parameter the client already
    # submitted. Model-appropriate sampling is the client's and endpoint's
    # domain, so bandaid never invents values — it only nudges the client's
    # own values upward by this delta.
    loop_retry_increment: float = 0.1

    # Fallback sampling parameters used *only* when the client did not submit
    # that parameter at all. ``temperature`` (more variety), ``repeat_penalty``
    # (llama.cpp native) and ``presence_penalty``/``frequency_penalty``
    # (OpenAI) discourage repeating the tokens/phrases that produced the loop.
    loop_retry_temperature: float = 1.2
    loop_retry_repeat_penalty: float = 1.2
    loop_retry_presence_penalty: float = 0.3
    loop_retry_frequency_penalty: float = 0.3

    # --- Stall detection (silently hung streams) ---------------------
    # Master switch for the stalled-stream detection feature.
    stall_detection_enabled: bool = True

    # Max wall-clock seconds to wait for the FIRST content-bearing token
    # (thinking or response delta) after the upstream returns headers. Prefill
    # on a large prompt is legitimately slow, so this is generous.
    stall_ttft_seconds: float = 120.0

    # Max wall-clock seconds between content-bearing tokens once generation has
    # started. SSE keepalives/comments do NOT reset this — only real tokens do.
    stall_gap_seconds: float = 60.0

    # HTTP status returned to the client when a stall is detected.
    stall_abort_status: int = 502

    # --- Context-window overflow -------------------------------------
    # Master switch for recognising the upstream's context-window-fill error
    # and failing fast (instead of retrying a doomed request).
    context_window_detection_enabled: bool = True

    # Comma-separated substrings (matched case-insensitively) that the upstream
    # emits when the context window fills. Kept exact to avoid false positives;
    # override for other providers/versions.
    context_window_markers: str = (
        "exceeded the context window,"
        "context window exceeded,"
        "context length exceeded,"
        "maximum context length,"
        "prompt is too long,"
        "prompt is longer than the context,"
        "input is too long,"
        "ran out of context,"
        "exceeds the available context size"
    )

    # HTTP status returned to the client when a context-window overflow is
    # recognised but no upstream body is available to pass through (e.g. the
    # overflow arrived as an exception message). When the upstream returned a
    # body, its own status and body are forwarded verbatim instead.
    context_window_abort_status: int = 413

    # --- Think-tag cleanup -------------------------------------------
    # Master switch for relocating leaked thinking tags out of the visible
    # content and guaranteeing a non-empty response to the client.
    think_cleanup_enabled: bool = True

    # Comma-separated tag names (matched case-insensitively) whose inner text
    # is relocated out of ``content`` into ``reasoning_content`` — e.g.
    # ``<think>…</think>``, ``<reasoning>…</reasoning>``. Exact tag names only;
    # no fuzzy heuristics.
    think_tags: str = "think,thinking,reasoning"

    # Placeholder message emitted as the visible ``content`` when the model
    # produced only a thinking tag and nothing else. Kept short and explicit so
    # the client receives a non-empty turn instead of aborting the flow.
    think_empty_response_placeholder: str = (
        "The model replied inside a thinking tag; see reasoning_content."
    )

    # --- Nudge (re-prompt empty-text turns) --------------------------
    # Master switch for re-submitting a turn that produced only reasoning
    # (finish_reason=stop, no content, no tool calls) with a short re-prompt.
    think_nudge_enabled: bool = True

    # Nudge text appended as a ``user`` message on the re-submitted request.
    think_nudge_text: str = (
        "Your previous reply contained no visible text and no tool call. "
        "Reply with a visible answer, or call a tool if the task requires one."
    )

    # Maximum number of nudge re-submissions before falling back to the
    # placeholder floor (see ``think_empty_response_placeholder``).
    think_nudge_max_attempts: int = 2

    # --- Coast detection (announced-but-absent tool call) -------------
    # Master switch for re-submitting a turn that produced non-empty visible
    # content but no tool call, even though a tool call was possible (the
    # request carried ``tools`` and the model had already been driving a tool
    # loop) and the model's reasoning collapsed to be byte-identical with its
    # visible content. This is the "silent workflow stop" described in #16:
    # the model regurgitated its status line instead of generating the call.
    coast_detection_enabled: bool = True

    # Re-prompt text appended as a ``user`` message on the re-submitted
    # request. The coasted assistant turn is replayed immediately before it so
    # the re-prompt refers to something the model actually said.
    coast_nudge_text: str = (
        "You announced a tool call but did not make one. "
        "Call the tool now, or if the task is complete, say so explicitly."
    )

    # Maximum number of coast re-submissions before giving up and returning
    # the coasted turn as-is (visible content, no tool call, logged outcome).
    coast_max_attempts: int = 2

    # --- Runaway-reasoning detection (thinks endlessly, never answers) ---
    # Master switch for flagging a turn that keeps emitting reasoning tokens
    # while never producing visible content — the model "thinks endlessly" and
    # either trips a proactive token budget or exhausts the output window with
    # nothing to show (see #17). Distinct from loop (repetitive) and stall
    # (silent): here tokens flow continuously but never become an answer.
    runaway_reasoning_enabled: bool = True

    # Reasoning-token budget above which a turn is flagged as runaway, provided
    # no content (or tool call) has appeared yet. Token count is approximated
    # at ~4 characters/token (see the loop detector's window comment).
    runaway_reasoning_token_threshold: int = 2000

    # Re-prompt text appended as a ``user`` message on the re-submitted
    # request, telling the model to stop analysing and answer now.
    runaway_reasoning_nudge_text: str = (
        "Stop thinking and produce your final answer now, without further "
        "analysis. If the task requires a tool call, make it."
    )

    # Maximum number of runaway re-submissions before aborting the request.
    runaway_reasoning_max_attempts: int = 2

    # HTTP status returned to the client when a runaway turn is detected and
    # every re-submission has also run away.
    runaway_reasoning_abort_status: int = 502

    # --- Message-overflow protection (oversized tool results) ---------
    # Master switch for warning (and optionally truncating) oversized
    # ``role: "tool"`` results before forwarding the request upstream. A single
    # huge tool result can silently eat the model's context window; this guard
    # flags it and reclaims the context (see #15).
    message_overflow_enabled: bool = True

    # Size threshold, in characters, above which a tool-result message is
    # treated as oversized. Characters (not bytes) so the trigger and the
    # truncation prefix share one unit; for ASCII they coincide with bytes.
    message_overflow_threshold: int = 4096

    # Whether to truncate oversized tool results to a bounded prefix (the
    # first line, capped at the threshold). When False the full content is
    # kept and only the warning is prepended — rarely useful, as it just adds
    # tokens to a message the model already sees.
    message_overflow_truncate: bool = True

    # Warning text prepended to an oversized tool result, telling the model
    # the content was too large and to find a workaround.
    message_overflow_warning: str = (
        "This tool result is very large and may exceed the model's context "
        "window; the full result may have been truncated. Find a workaround "
        "rather than relying on the complete result."
    )

    # --- Tool-call syntax enforcement --------------------------------
    # Master switch for validating assembled tool calls: repair truncated
    # ``function.arguments`` where deterministic, flag the rest without
    # crashing the request.
    tool_call_guard_enabled: bool = True


settings = Settings()
