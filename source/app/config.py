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

    # Sliding window size (recent sentences compared against).
    loop_window_sentences: int = 20

    # Jaccard similarity above which two sentences count as "the same".
    loop_jaccard_threshold: float = 0.65

    # Number of similar sentences within the window that constitutes a loop.
    loop_min_loop_count: int = 3

    # Word n-gram size used for similarity.
    loop_ngram_size: int = 3

    # Compression ratio below which the window is considered low-entropy.
    loop_compression_ratio: float = 0.22

    # Minimum window length (chars) before the compression check applies.
    loop_compression_min_chars: int = 300

    # HTTP status returned to the client when a loop is detected.
    loop_abort_status: int = 502

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
        "ran out of context"
    )

    # HTTP status returned to the client when a context-window overflow is
    # recognised (413 Payload Too Large: the request cannot fit the model).
    context_window_abort_status: int = 413


settings = Settings()
