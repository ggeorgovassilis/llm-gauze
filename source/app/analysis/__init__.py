"""Modular, agent-facing analysis toolkit for recorded exchanges.

Small composable stages over the recorder's JSONL stream::

    from app.analysis import Digest, Filter, Project, Read, read_records

    records = read_records("data/records.jsonl")
    errors = list(Filter(outcome="http_error").apply(records))
    summary = (Read("data/records.jsonl") | Filter(kind="abort") | Digest("stuck")).run()

See ``docs/analysis.md`` for the stage semantics and the four digests.
"""

from __future__ import annotations

from app.analysis.classify import (
    content_text,
    decode_body,
    diagnosis_reason,
    error_text,
    has_tool_calls,
    is_abort,
    is_attempt,
    is_final,
    is_pass,
    normalise_tool_call,
    pass_rung,
    payload_objects,
    record_kind,
    record_rungs,
    request_messages,
    response_messages,
    tool_calls,
)
from app.analysis.digests import DIGESTS, endpoint, narrative, remediation, run_digest, stuck
from app.analysis.filters import build_predicate, filter_records
from app.analysis.pipeline import Digest, Filter, Pipeline, Project, Read
from app.analysis.projection import project, project_record
from app.analysis.reader import (
    RecordFile,
    discover_record_files,
    iter_records,
    read_records,
)

__all__ = [
    "DIGESTS",
    "Digest",
    "Filter",
    "Pipeline",
    "Project",
    "Read",
    "RecordFile",
    "build_predicate",
    "content_text",
    "decode_body",
    "diagnosis_reason",
    "discover_record_files",
    "endpoint",
    "error_text",
    "filter_records",
    "has_tool_calls",
    "is_abort",
    "is_attempt",
    "is_final",
    "is_pass",
    "iter_records",
    "narrative",
    "normalise_tool_call",
    "pass_rung",
    "payload_objects",
    "project",
    "project_record",
    "read_records",
    "record_kind",
    "record_rungs",
    "remediation",
    "request_messages",
    "response_messages",
    "run_digest",
    "stuck",
    "tool_calls",
]
