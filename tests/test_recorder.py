"""Unit tests for the recorder's restart-rotation behaviour."""

import asyncio
import json
import tempfile
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
        asyncio.run(recorder.record({"method": "POST", "path": "/x"}))
        assert path.exists(), "record file should be created on first write"
        # No rotation happened: only the target file exists.
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["records.jsonl"]


def test_existing_file_is_rotated_on_restart():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"

        # First run writes a record.
        first = Recorder(path)
        asyncio.run(first.record({"run": "one"}))

        # Restart: the old file must be rotated aside (timestamp appended),
        # and a fresh file created on the next write.
        second = Recorder(path)
        asyncio.run(second.record({"run": "two"}))

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
        asyncio.run(Recorder(path).record({"a": 1}))
        # Remove the file entirely, then "restart" — nothing to rotate.
        path.unlink()
        asyncio.run(Recorder(path).record({"b": 2}))
        assert sorted(p.name for p in Path(tmp).iterdir()) == ["records.jsonl"]


def test_concurrent_records_preserve_call_order():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "records.jsonl"
        recorder = Recorder(path)

        async def record_all():
            await asyncio.gather(*(recorder.record({"seq": i}) for i in range(100)))

        asyncio.run(record_all())

        seqs = [json.loads(line)["seq"] for line in path.read_text().splitlines()]
        assert seqs == list(range(100))
