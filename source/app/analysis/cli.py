"""Command-line wrapper for the analysis toolkit.

Usage::

    scripts/analyse-records 'read | digest stuck'
    scripts/analyse-records 'read | filter outcome=http_error | digest endpoint'
    scripts/analyse-records 'read | filter kind=abort | project id,abort_kind,abort_reason'
    scripts/analyse-records 'read | filter request_id=abc | digest narrative' --format jsonl

The pipeline is a quoted string of ``|``-separated stages.  Output is JSON (or
one JSON object per line with ``--format jsonl``) — never prose — so the agent
can pipe it into ``jq`` or read it directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.analysis.pipeline import Digest, Filter, Pipeline, Project, Read, default_records_path

_FILTER_KEYS = {
    "outcome",
    "status",
    "attempt",
    "error",
    "reason",
    "abort_kind",
    "path",
    "query",
    "since",
    "until",
    "rung",
    "has_tool_calls",
    "request_id",
    "kind",
    "retryable",
}
_BOOL_FILTER_KEYS = {"has_tool_calls", "retryable"}


def _parse_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _pairs(tokens: Iterable[str]) -> list[tuple[str, str]]:
    """Split ``key=value`` tokens (also comma-separated) into pairs."""
    pairs = []
    for token in tokens:
        for part in token.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"expected key=value in pipeline stage argument: {part!r}")
            key, _, value = part.partition("=")
            pairs.append((key.strip(), value.strip()))
    return pairs


def _collapse(values: list[Any]) -> Any:
    return values[0] if len(values) == 1 else values


def _parse_stage(text: str) -> tuple[str, list[str]]:
    tokens = text.strip().split()
    if not tokens:
        raise ValueError("empty pipeline stage")
    return tokens[0].lower(), tokens[1:]


def parse_stage(text: str) -> Any:
    """Parse one stage expression (``filter …``, ``project …``, ``digest …``)."""
    name, args = _parse_stage(text)
    if name == "read":
        target = None
        include_rotated = True
        option_args = list(args)
        if option_args and "=" not in option_args[0]:
            target = option_args.pop(0)
        for key, value in _pairs(option_args):
            if key == "path":
                target = value
            elif key == "include_rotated":
                include_rotated = bool(_parse_value(value))
            else:
                raise ValueError(f"unknown read option {key!r}")
        return Read(target, include_rotated=include_rotated)
    if name == "filter":
        grouped: dict[str, list[Any]] = {}
        for key, value in _pairs(args):
            if key not in _FILTER_KEYS:
                raise ValueError(
                    f"unknown filter {key!r}; choose from {', '.join(sorted(_FILTER_KEYS))}"
                )
            grouped.setdefault(key, []).append(_parse_value(value))
        criteria = {}
        for key, values in grouped.items():
            if key in _BOOL_FILTER_KEYS:
                criteria[key] = bool(values[-1])
            else:
                criteria[key] = _collapse(values)
        return Filter(**criteria)
    if name == "project":
        projection: dict[str, str] = {}
        has_alias = False
        for token in args:
            for part in token.split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    projection[part] = part
                    continue
                alias, _, path = part.partition("=")
                alias, path = alias.strip(), path.strip()
                if not alias or not path:
                    raise ValueError(f"expected alias=path in project stage: {part!r}")
                if alias in projection:
                    raise ValueError(f"duplicate project field {alias!r}")
                projection[alias] = path
                has_alias = True
        if not projection:
            raise ValueError("project requires at least one field")
        return Project(projection if has_alias else list(projection))
    if name == "digest":
        if not args:
            raise ValueError("digest requires a name: stuck, endpoint, remediation or narrative")
        digest = args[0]
        options = {key: _parse_value(value) for key, value in _pairs(args[1:])}
        return Digest(digest, **options)
    raise ValueError(f"unknown pipeline stage {name!r}; choose read, filter, project or digest")


def build_pipeline(
    spec: str,
    *,
    records: str | Path | None = None,
    include_rotated: bool = True,
) -> Pipeline:
    """Build a :class:`Pipeline` from a ``read | filter … | digest …`` string."""
    parts = [part.strip() for part in spec.split("|") if part.strip()]
    if not parts:
        raise ValueError("empty pipeline")
    stages = [parse_stage(part) for part in parts]
    if not isinstance(stages[0], Read):
        stages.insert(0, Read(records, include_rotated=include_rotated))
    else:
        first = stages[0]
        assert isinstance(first, Read)
        if first.target is None:
            first.target = records
        first.include_rotated = include_rotated and first.include_rotated
    for stage in stages[:-1]:
        if isinstance(stage, Digest):
            raise ValueError("digest must be the last pipeline stage")
    return Pipeline(*stages)


def render(result: Any, *, output_format: str = "json", compact: bool = False) -> str:
    """Render a pipeline result as JSON or JSONL."""
    if output_format == "jsonl":
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, default=str)
        return "\n".join(json.dumps(record, ensure_ascii=False, default=str) for record in result)
    payload = result if isinstance(result, dict) else list(result)
    indent = None if compact else 2
    return json.dumps(payload, indent=indent, ensure_ascii=False, default=str)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analyse-records",
        description="Composable analysis over llm-gauze recorded exchanges.",
    )
    parser.add_argument(
        "pipeline",
        nargs="*",
        help="quoted pipeline, e.g. 'read | filter outcome=http_error | digest endpoint'",
    )
    parser.add_argument("-p", "--pipeline", "--pipeline-spec", dest="pipeline_option")
    parser.add_argument("--records", help="live records.jsonl (rotated siblings are included)")
    parser.add_argument("--data-dir", help="directory containing records.jsonl")
    parser.add_argument("--format", choices=("json", "jsonl"), default="json")
    parser.add_argument("--compact", action="store_true", help="single-line JSON output")
    parser.add_argument("--no-rotated", action="store_true", help="read only the live file")
    args = parser.parse_args(argv)

    spec = args.pipeline_option or " ".join(args.pipeline)
    if not spec:
        parser.error("a pipeline is required, e.g. 'read | digest stuck'")

    if args.data_dir:
        records: str | Path | None = Path(args.data_dir) / "records.jsonl"
    elif args.records:
        records = Path(args.records)
    else:
        records = default_records_path()

    try:
        pipeline = build_pipeline(spec, records=records, include_rotated=not args.no_rotated)
        result = pipeline.run()
    except ValueError as exc:
        parser.error(str(exc))

    sys.stdout.write(render(result, output_format=args.format, compact=args.compact))
    sys.stdout.write("\n")
    return 0
