"""Composable analysis pipeline: ``read | filter | project | digest``.

Stages are small objects with an ``apply`` method.  ``Pipeline`` chains them
and supports ``|`` so a caller can compose one in Python the same way the CLI
does on the command line::

    pipeline = Read("data/records.jsonl") | Filter(outcome="http_error") | Digest("endpoint")
    summary = pipeline.run()

A pipeline starts with a reader (or any iterable of records) and ends in either
a record stream (filter/project) or a digest summary dict.  Nothing in a stage
prints or formats prose; the CLI is the only place that renders JSON.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator

from app.analysis.digests import run_digest
from app.analysis.filters import filter_records
from app.analysis.projection import project
from app.analysis.reader import iter_records


class Stage:
    """Base class: a stage transforms the previous stage's output."""

    def apply(self, data: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def __or__(self, other: Stage) -> Pipeline:
        return Pipeline(self, other)


class Read(Stage):
    """Stream records from a recording (rotated files plus the live one)."""

    def __init__(self, target: str | Path | None = None, *, include_rotated: bool = True) -> None:
        self.target = target
        self.include_rotated = include_rotated

    def apply(self, data: Any = None) -> Iterator[dict]:
        target = self.target if self.target is not None else default_records_path()
        return iter_records(target, include_rotated=self.include_rotated)


class Filter(Stage):
    """Keep records matching every criterion (see :func:`filter_records`)."""

    def __init__(self, **criteria: Any) -> None:
        self.criteria = criteria

    def apply(self, data: Iterable[dict]) -> Iterator[dict]:
        return filter_records(data, **self.criteria)


class Project(Stage):
    """Reduce each record to the requested fields."""

    def __init__(self, fields: Iterable[str] | dict[str, str]) -> None:
        self.fields = fields

    def apply(self, data: Iterable[dict]) -> Iterator[dict]:
        return project(data, self.fields)


class Digest(Stage):
    """Run a named digest and return its summary dict."""

    def __init__(self, name: str, **options: Any) -> None:
        self.name = name
        self.options = options

    def apply(self, data: Iterable[dict]) -> dict:
        return run_digest(self.name, data, **self.options)


class Pipeline:
    """An ordered chain of stages."""

    def __init__(self, *stages: Stage) -> None:
        self.stages: list[Stage] = list(stages)

    def __or__(self, stage: Stage) -> Pipeline:
        return Pipeline(*self.stages, stage)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Pipeline({', '.join(type(stage).__name__ for stage in self.stages)})"

    def run(self) -> Any:
        """Run every stage in order and return the last stage's output."""
        data: Any = None
        for stage in self.stages:
            data = stage.apply(data)
        return data

    def records(self) -> list[dict]:
        """Materialise the pipeline's record stream (when it ends in records)."""
        result = self.run()
        if isinstance(result, dict):
            raise TypeError("pipeline ends in a digest summary, not a record stream")
        return list(result)


def default_records_path() -> Path:
    """The recording the CLI reads when no path is supplied.

    ``DATA_DIR``/``RECORD_FILE`` are honoured first (the same environment the
    gateway uses); otherwise the repository's local ``data/records.jsonl``.
    """
    data_dir = os.environ.get("DATA_DIR")
    record_file = os.environ.get("RECORD_FILE", "records.jsonl")
    if data_dir:
        return Path(data_dir) / record_file
    return Path("data") / record_file
