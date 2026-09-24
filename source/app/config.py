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

    # Upstream request timeout, in seconds.
    request_timeout: float = 300.0

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


settings = Settings()
