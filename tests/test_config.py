"""Tests for config validation and the generated config artefacts.

Covers two acceptance criteria from #95:

* ``Settings`` rejects out-of-range / malformed values at load time (ranges,
  enum-like HTTP status codes) instead of silently misbehaving at runtime.
* ``.env.example`` and ``docs/configuration.md`` are generated from ``config.py``
  and never drift from it.
"""

from pathlib import Path

import pytest
from app.config import Settings
from app.config_docs import configuration_md, env_example
from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_defaults_are_valid():
    # Every shipped default must pass the model's own validation. Validate the
    # declared defaults explicitly so the check is isolated from any ambient
    # `.env` file or environment variables and is therefore deterministic.
    defaults = {name: info.default for name, info in Settings.model_fields.items()}
    assert Settings.model_validate(defaults)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("retry_max_attempts", 0),
        ("retry_backoff_initial", -1),
        ("retry_backoff_base", 0.5),
        ("retry_backoff_max", 0),
        ("request_timeout", 0),
        ("connect_timeout", -1),
        ("loop_window_bytes", 0),
        ("loop_min_output_fraction", 1.5),
        ("loop_min_output_fraction", -0.1),
        ("loop_compression_ratio", 2),
        ("loop_abort_status", 99),
        ("loop_abort_status", 600),
        ("stall_ttft_seconds", 0),
        ("stall_gap_seconds", -1),
        ("stall_abort_status", 1000),
        ("context_window_abort_status", 0),
        ("runaway_reasoning_token_threshold", 0),
        ("runaway_reasoning_abort_status", 42),
        ("message_overflow_threshold", 0),
        ("llm_base_url", ""),
        ("record_file", ""),
        ("log_format", ""),
        ("log_level", ""),
        ("log_level", "VERBOSE"),
        ("log_level", "trace"),
        ("think_tags", ""),
    ],
)
def test_invalid_values_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        Settings.model_validate({field: bad_value})


@pytest.mark.parametrize(
    "bad_statuses",
    [
        "",  # no status codes
        "abc",  # not a number
        "500,99",  # out of HTTP range
        "500,600",  # out of HTTP range
    ],
)
def test_retryable_status_codes_reject_bad_values(bad_statuses):
    with pytest.raises(ValidationError):
        Settings.model_validate({"retryable_status_codes": bad_statuses})


def test_retryable_status_codes_normalises_whitespace():
    result = Settings.model_validate({"retryable_status_codes": " 500 , 503 "})
    assert result.retryable_status_codes == "500,503"


def test_log_level_normalises_case():
    result = Settings.model_validate({"log_level": "debug"})
    assert result.log_level == "DEBUG"


def test_env_example_is_up_to_date():
    assert (REPO_ROOT / ".env.example").read_text() == env_example()


def test_configuration_md_is_up_to_date():
    assert (REPO_ROOT / "docs" / "configuration.md").read_text() == configuration_md()


def _env_keys(text: str) -> set[str]:
    """Return the set of variable names declared in a rendered ``.env.example``."""
    return {
        line.split("=", 1)[0]
        for line in text.splitlines()
        if line and not line.startswith("#") and "=" in line
    }


def _md_tokens(text: str) -> set[str]:
    """Return the set of backtick-quoted variable tokens in ``configuration.md``."""
    return {line.split("|")[1].strip() for line in text.splitlines() if line.startswith("| `")}


def test_every_setting_is_documented():
    # Exact-token matching so an overlapping setting name cannot false-positive.
    env_keys = _env_keys(env_example())
    md_tokens = _md_tokens(configuration_md())
    for name in Settings.model_fields:
        var = name.upper()
        assert var in env_keys, f"{name} missing from .env.example"
        assert f"`{var}`" in md_tokens, f"{name} missing from docs/configuration.md"
