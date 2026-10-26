"""Change-record schema (#151).

Every silent mutation a guard or the proxy reports is a *change record*. This
file is the executable copy of `docs/record-schema.md`: it parses the per-kind
table and asserts each emitter produces `kind` plus only the keys documented
for that kind, so a new undocumented field fails the suite.
"""

import ast
import asyncio
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from app.proxy import Proxy, _StreamResult
from app.recorder import Recorder
from app.remediation.overflow import MessageOverflowGuard
from app.remediation.think import ThinkContentGuard
from app.remediation.tool_call import ToolCallGuard

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = REPO_ROOT / "docs" / "record-schema.md"
# Optional on every kind per the schema prose, not repeated per row in the table.
GENERIC_CHANGE_KEYS = {"index", "role"}


def _documented_change_keys() -> dict[str, set[str]]:
    """Parse the per-kind table from the change-record schema section.

    The kind-specific column is located by its header rather than by position,
    so adding or reordering columns in the schema does not silently make the
    guard parse the wrong cell.
    """
    text = SCHEMA.read_text(encoding="utf-8")
    section = text.split("## Change records", 1)[1].split("\n## ", 1)[0]
    documented: dict[str, set[str]] = {}
    fields_column: int | None = None
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if fields_column is None:
            if "Fields besides kind" in (cell.replace("`", "") for cell in cells):
                fields_column = next(
                    index
                    for index, cell in enumerate(cells)
                    if cell.replace("`", "") == "Fields besides kind"
                )
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        kind = cells[0].strip("`")
        documented[kind] = set(re.findall(r"`([a-z0-9_]+)`", cells[fields_column]))
    return documented


# Calls whose ``kind=`` keyword emits a change record. Restricting the scan to
# these constructors keeps an unrelated ``kind=`` keyword from masquerading as
# a change kind.
CHANGE_EMITTER_CONSTRUCTORS = {"dict"}


def _kinds_in_source(source: str) -> set[str]:
    """Kinds literal-emitted in ``source`` as a ``"kind"`` value.

    Both ``{"kind": "x"}`` dict literals and ``dict(kind="x")`` calls are
    recognised; other calls are ignored even when they happen to take a
    literal ``kind=`` keyword. The source is parsed rather than regex-scanned
    so a kind named only in a comment or docstring cannot satisfy the
    documented-kind guard. A kind built dynamically (held in a variable or
    computed at runtime) is not visible to this scan.
    """
    kinds: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "kind"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    kinds.add(value.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHANGE_EMITTER_CONSTRUCTORS
        ):
            for keyword in node.keywords:
                if (
                    keyword.arg == "kind"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    kinds.add(keyword.value.value)
    return kinds


def _kinds_emitted_in_source() -> set[str]:
    """All change kinds literal-emitted by the guards and the proxy."""
    sources = [REPO_ROOT / "source" / "app" / "proxy.py"]
    sources.extend(sorted((REPO_ROOT / "source" / "app" / "remediation").glob("*.py")))
    kinds: set[str] = set()
    for path in sources:
        kinds.update(_kinds_in_source(path.read_text(encoding="utf-8")))
    return kinds


def _assert_documented(changes: list[dict]) -> None:
    """Assert every change carries exactly the keys documented for its kind."""
    documented = _documented_change_keys()
    assert changes, "emitter produced no change records"
    for change in changes:
        kind = change["kind"]
        assert kind in documented, f"undocumented change kind: {kind}"
        required = {"kind", *documented[kind]}
        allowed = required | GENERIC_CHANGE_KEYS
        missing = required - set(change)
        extra = set(change) - allowed
        assert not missing, f"{kind} missing documented keys: {sorted(missing)}"
        assert not extra, f"{kind} emitted undocumented keys: {sorted(extra)}"


def test_documented_kinds_cover_every_emitter():
    assert _kinds_emitted_in_source() == set(_documented_change_keys())


def test_kind_source_scan_handles_quote_styles_and_digits():
    source = """records = [{"kind": "tool_overflow"}, {'kind': 'think2'}]"""
    assert _kinds_in_source(source) == {"tool_overflow", "think2"}


def test_kind_source_scan_handles_dict_constructor():
    source = 'record = dict(kind="flagged_malformed", reason="missing_name")'
    assert _kinds_in_source(source) == {"flagged_malformed"}


def test_kind_source_scan_ignores_unrelated_kind_keywords():
    # Only the documented emitter constructors count; a ``kind=`` keyword on
    # an unrelated call must not be mistaken for a change kind.
    source = 'verdict = Verdict(kind="ghost_emitter")\nstate = Thing(kind="other")'
    assert _kinds_in_source(source) == set()


def test_kind_source_scan_cannot_resolve_variable_kinds():
    # Documented limitation: the scan only sees literal kind values.
    source = 'kind = "tool_overflow"\nrecord = {"kind": kind}'
    assert _kinds_in_source(source) == set()


def test_kind_source_scan_ignores_comments_and_docstrings():
    source = '''\
"""Example: ``{"kind": "ghost_emitter"}`` in a docstring."""

records = [{"kind": "tool_overflow"}]
# {"kind": "commented_out"}
'''
    assert _kinds_in_source(source) == {"tool_overflow"}


def test_documented_change_keys_locates_fields_column_from_header(monkeypatch, tmp_path):
    schema = tmp_path / "record-schema.md"
    schema.write_text(
        "## Change records\n\n"
        "| `kind` | Fields besides `kind` | Meaning | Emitted by |\n"
        "| --- | --- | --- | --- |\n"
        "| `example_kind` | `size`, `flag` | what | emitter |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[__name__], "SCHEMA", schema)
    assert _documented_change_keys() == {"example_kind": {"size", "flag"}}


def test_assert_documented_rejects_undocumented_kind():
    with pytest.raises(AssertionError, match=r"undocumented change kind: ghost_kind"):
        _assert_documented([{"kind": "ghost_kind"}])


def test_assert_documented_rejects_undocumented_key():
    with pytest.raises(AssertionError, match=r"undocumented keys: \['chars'\]"):
        _assert_documented([{"kind": "repaired_arguments", "chars": 1}])


def test_assert_documented_rejects_missing_kind_specific_key():
    # `tool_overflow` also requires `truncated`, so `size` alone is not enough.
    with pytest.raises(AssertionError, match=r"missing documented keys: \['truncated'\]"):
        _assert_documented([{"kind": "tool_overflow", "size": 5}])


def test_generic_optional_keys_are_allowed_on_every_kind():
    # `index` and `role` are optional on every record, not just the kinds whose
    # table row happens to mention them.
    _assert_documented([{"kind": "empty_content_placeholder", "index": 0, "role": "assistant"}])


def test_overflow_emits_documented_keys():
    guard = MessageOverflowGuard(threshold=4, truncate=True, warning="warning")
    _, changes = guard.process({"messages": [{"role": "tool", "content": "12345"}]})
    assert [change["kind"] for change in changes] == ["tool_overflow"]
    _assert_documented(changes)


def test_think_emits_documented_keys():
    guard = ThinkContentGuard(tags=("think",), placeholder="placeholder")
    _, _, changes = guard.clean("<think>why</think>", "", None)
    assert [change["kind"] for change in changes] == [
        "relocated_think",
        "empty_content_placeholder",
    ]
    _assert_documented(changes)


def test_tool_call_emits_documented_keys():
    guard = ToolCallGuard()
    changes = guard.validate(
        [
            {"function": {"name": "f", "arguments": '{"a": 1'}},
            {"function": {"name": "f", "arguments": '{"a": }}'}},
        ]
    )
    assert [change["kind"] for change in changes] == [
        "repaired_arguments",
        "flagged_malformed",
    ]
    _assert_documented(changes)


def _result(content: str) -> _StreamResult:
    return _StreamResult(
        content=content,
        reasoning="",
        tool_calls=None,
        meta={"model": "test"},
        status=200,
        resp_headers={},
        attempt=1,
        max_attempts=1,
        duration=0.01,
    )


def _finalize(tmp_path, content: str, *, extract_outcome=None, empty_stop_outcome=None):
    """Drive the proxy finaliser with one exhausted rung and return the response."""
    record_path = tmp_path / "change_records.jsonl"
    proxy = SimpleNamespace(recorder=Recorder(record_path))

    async def run():
        return await Proxy._finalize_success(
            proxy,
            {},
            _result(content),
            None,
            False,
            1 if extract_outcome else 0,
            extract_outcome,
            0,
            None,
            1 if empty_stop_outcome else 0,
            empty_stop_outcome,
            0,
            0.0,
            None,
        )

    return asyncio.run(run())


@pytest.mark.parametrize(
    ("extract_outcome", "empty_stop_outcome"),
    [("exhausted", None), (None, "exhausted")],
)
def test_proxy_empty_placeholder_emits_documented_keys(
    tmp_path, read_records, extract_outcome, empty_stop_outcome
):
    """Both proxy-side empty-content places emit the shared documented kind."""
    _finalize(
        tmp_path,
        "",
        extract_outcome=extract_outcome,
        empty_stop_outcome=empty_stop_outcome,
    )
    changes = read_records(tmp_path / "change_records.jsonl")[-1]["think_cleanup"]
    assert [change["kind"] for change in changes] == ["empty_content_placeholder"]
    _assert_documented(changes)


@pytest.mark.parametrize(
    ("extract_outcome", "empty_stop_outcome"),
    [("exhausted", None), (None, "exhausted")],
)
def test_proxy_records_placeholder_only_when_it_substituted(
    tmp_path, read_records, extract_outcome, empty_stop_outcome
):
    """A non-empty final turn is delivered unchanged and gets no placeholder record."""
    response = _finalize(
        tmp_path,
        "kept",
        extract_outcome=extract_outcome,
        empty_stop_outcome=empty_stop_outcome,
    )
    assert response.status_code == 200
    message = json.loads(response.body)["choices"][0]["message"]
    assert message["content"] == "kept"
    assert read_records(tmp_path / "change_records.jsonl")[-1]["think_cleanup"] is None
