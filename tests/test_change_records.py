"""Change-record schema (#151).

Every silent mutation a guard or the proxy reports is a *change record*. This
file is the executable copy of `docs/record-schema.md`: it parses the per-kind
table and asserts each emitter produces `kind` plus only the keys documented
for that kind, so a new undocumented field fails the suite.
"""

import asyncio
import json
import re
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


def _documented_change_keys() -> dict[str, set[str]]:
    """Parse the per-kind table from the change-record schema section."""
    text = SCHEMA.read_text(encoding="utf-8")
    section = text.split("## Change records", 1)[1].split("\n## ", 1)[0]
    documented: dict[str, set[str]] = {}
    for line in section.splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        kind = cells[0].strip("`")
        if kind == "kind":
            continue
        documented[kind] = set(re.findall(r"`([a-z_]+)`", cells[2]))
    return documented


def _kinds_emitted_in_source() -> set[str]:
    """All change kinds literal-emitted by the guards and the proxy."""
    sources = [REPO_ROOT / "source" / "app" / "proxy.py"]
    sources.extend(sorted((REPO_ROOT / "source" / "app" / "remediation").glob("*.py")))
    kinds: set[str] = set()
    for path in sources:
        kinds.update(re.findall(r'"kind":\s*"([a-z_]+)"', path.read_text(encoding="utf-8")))
    return kinds


def _assert_documented(changes: list[dict]) -> None:
    """Assert every change carries exactly the keys documented for its kind."""
    documented = _documented_change_keys()
    assert changes, "emitter produced no change records"
    for change in changes:
        kind = change["kind"]
        expected = {"kind", *documented.get(kind, set())}
        assert kind in documented, f"undocumented change kind: {kind}"
        assert set(change) == expected, (
            f"{kind} emitted {sorted(change)}, documented {sorted(expected)}"
        )


def test_documented_kinds_cover_every_emitter():
    assert _kinds_emitted_in_source() == set(_documented_change_keys())


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


@pytest.mark.parametrize(
    ("extract_outcome", "empty_stop_outcome"),
    [("exhausted", None), (None, "exhausted")],
)
def test_proxy_empty_placeholder_emits_documented_keys(
    tmp_path, extract_outcome, empty_stop_outcome
):
    """Both proxy-side empty-content places emit the shared documented kind."""
    record_path = tmp_path / "change_records.jsonl"
    proxy = SimpleNamespace(recorder=Recorder(record_path))
    result = _StreamResult(
        content="",
        reasoning="",
        tool_calls=None,
        meta={"model": "test"},
        status=200,
        resp_headers={},
        attempt=1,
        max_attempts=1,
        duration=0.01,
    )

    async def run():
        await Proxy._finalize_success(
            proxy,
            {},
            result,
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

    asyncio.run(run())
    records = [
        json.loads(line)
        for line in record_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changes = records[-1]["think_cleanup"]
    assert [change["kind"] for change in changes] == ["empty_content_placeholder"]
    _assert_documented(changes)
