"""llm-gauze — an HTTP gateway that works around local LLM shortcomings."""

# Single source of truth for the release version. The scheme is a plain
# incrementing number (v1, v2, v3, …) — no semver compatibility semantics.
# Bump this before cutting a release; the publish workflow asserts the tag
# (e.g. `v7`) matches this value.
__version__ = "0"
