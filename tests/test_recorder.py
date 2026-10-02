"""Unit tests for the recorder's restart-rotation behaviour.

Runs with plain Python (stdlib only) inside the container:

    docker compose exec -T gateway python - < tests/test_recorder.py
"""

import json
import tempfile
import traceback
from pathlib import Path

try:
    from app.recorder import Recorder
except ImportError:  # pragma: no cover - run on host without the container
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
    from app.recorder import Recorder


def test_fresh_dir_starts_new_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        recorder = Recorder(path)
        recorder.record({"method": "POST", "path": "/x"})
        assert path.exists(), "record file should be created on first write"
        # No rotation happened: only the target file exists.
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["records.jsonl"]


def test_existing_file_is_rotated_on_restart():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"

        # First run writes a record.
        first = Recorder(path)
        first.record({"run": "one"})

        # Restart: the old file must be rotated aside (timestamp appended),
        # and a fresh file created on the next write.
        second = Recorder(path)
        second.record({"run": "two"})

        files = sorted(p.name for p in Path(tmp).iterdir())
        # One rotated file + the fresh target.
        rotated = [f for f in files if f != "records.jsonl"]
        assert len(rotated) == 1, files
        assert rotated[0].startswith("records-"), rotated

        # Rotated file holds run one; fresh file holds run two only.
        rotated_path = Path(tmp) / rotated[0]
        lines = [json.loads(line) for line in rotated_path.read_text().splitlines()]
        assert [r["run"] for r in lines] == ["one"]

        fresh_lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r["run"] for r in fresh_lines] == ["two"]


def test_no_rotation_when_file_absent_on_second_run():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        Recorder(path).record({"a": 1})
        # Remove the file entirely, then "restart" — nothing to rotate.
        path.unlink()
        Recorder(path).record({"b": 2})
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["records.jsonl"]


def _run_all() -> int:
    tests = [
        value
        for key, value in sorted(globals().items())
        if key.startswith("test_") and callable(value)
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    import sys

    sys.exit(1 if _run_all() else 0)
