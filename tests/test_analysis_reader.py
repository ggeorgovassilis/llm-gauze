"""Tests for reading rotated + live recorder files."""

import json
from pathlib import Path

from app.analysis import discover_record_files, iter_records, read_records


def _record(identifier, timestamp):
    return {"id": identifier, "timestamp": timestamp}


def test_reads_rotated_and_live_files_in_chronological_order(tmp_path):
    rotated = tmp_path / "records-20260101-120000.jsonl"
    rotated.write_text(
        "".join(
            json.dumps(_record(name, timestamp)) + "\n"
            for name, timestamp in (("old-1", 1.0), ("old-2", 2.0))
        ),
        encoding="utf-8",
    )
    live = tmp_path / "records.jsonl"
    live.write_text(
        "".join(
            json.dumps(_record(name, timestamp)) + "\n"
            for name, timestamp in (("new-1", 3.0), ("new-2", 4.0))
        ),
        encoding="utf-8",
    )

    assert [record["id"] for record in read_records(live)] == [
        "old-1",
        "old-2",
        "new-1",
        "new-2",
    ]
    assert [record["id"] for record in read_records(tmp_path)] == [
        "old-1",
        "old-2",
        "new-1",
        "new-2",
    ]


def test_merge_by_timestamp_handles_clock_skew_between_files(tmp_path):
    # The rotated file carries a *later* timestamp than the live file; the
    # reader still yields the globally chronological order.
    (tmp_path / "records-20260101-120000.jsonl").write_text(
        json.dumps(_record("rotated", 20.0)) + "\n", encoding="utf-8"
    )
    live = tmp_path / "records.jsonl"
    live.write_text(json.dumps(_record("live", 10.0)) + "\n", encoding="utf-8")

    assert [record["id"] for record in read_records(tmp_path)] == ["live", "rotated"]


def test_rotated_files_are_sorted_by_timestamp_suffix(tmp_path):
    (tmp_path / "records-20260102-120000.jsonl").write_text(
        json.dumps(_record("second", 2.0)) + "\n", encoding="utf-8"
    )
    (tmp_path / "records-20260101-120000.jsonl").write_text(
        json.dumps(_record("first", 1.0)) + "\n", encoding="utf-8"
    )
    (tmp_path / "records-20260101-120000-1.jsonl").write_text(
        json.dumps(_record("first-collision", 1.5)) + "\n", encoding="utf-8"
    )
    (tmp_path / "records.jsonl").write_text(
        json.dumps(_record("live", 3.0)) + "\n", encoding="utf-8"
    )

    files = discover_record_files(tmp_path)
    assert [file.path.name for file in files] == [
        "records-20260101-120000.jsonl",
        "records-20260101-120000-1.jsonl",
        "records-20260102-120000.jsonl",
        "records.jsonl",
    ]
    assert files[-1].live is True


def test_hand_renamed_rotated_file_is_still_read(tmp_path):
    (tmp_path / "records-old.jsonl").write_text(
        json.dumps(_record("old", 1.0)) + "\n", encoding="utf-8"
    )
    (tmp_path / "records.jsonl").write_text(
        json.dumps(_record("live", 2.0)) + "\n", encoding="utf-8"
    )
    assert [record["id"] for record in read_records(tmp_path)] == ["old", "live"]


def test_corrupt_blank_and_non_object_lines_are_skipped(tmp_path):
    live = tmp_path / "records.jsonl"
    live.write_text(
        "\n".join(
            [
                json.dumps(_record("good-1", 1.0)),
                "{not json",
                "",
                json.dumps([1, 2, 3]),
                json.dumps(_record("good-2", 2.0)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert [record["id"] for record in read_records(live)] == ["good-1", "good-2"]


def test_zero_filled_and_truncated_tail_is_tolerated(tmp_path):
    live = tmp_path / "records.jsonl"
    good = json.dumps(_record("good", 1.0)).encode()
    torn = b'{"id": "torn", "timestamp": 2.0'
    live.write_bytes(good + b"\n" + torn + b"\n" + b"\x00" * 256 + b"\n")

    assert [record["id"] for record in read_records(live)] == ["good"]


def test_zero_bytes_inside_a_line_are_stripped(tmp_path):
    live = tmp_path / "records.jsonl"
    live.write_bytes(json.dumps(_record("good", 1.0)).encode() + b"\x00\x00\n")
    assert [record["id"] for record in read_records(live)] == ["good"]


def test_include_rotated_false_reads_only_the_live_file(tmp_path):
    (tmp_path / "records-20260101-120000.jsonl").write_text(
        json.dumps(_record("old", 1.0)) + "\n", encoding="utf-8"
    )
    live = tmp_path / "records.jsonl"
    live.write_text(json.dumps(_record("live", 2.0)) + "\n", encoding="utf-8")

    assert [record["id"] for record in iter_records(live, include_rotated=False)] == ["live"]


def test_missing_recording_yields_no_records(tmp_path):
    assert read_records(tmp_path / "records.jsonl") == []


def test_record_without_numeric_timestamp_sorts_first(tmp_path):
    live = tmp_path / "records.jsonl"
    live.write_text(
        json.dumps({"id": "no-ts"}) + "\n" + json.dumps(_record("with-ts", 1.0)) + "\n",
        encoding="utf-8",
    )
    assert [record["id"] for record in read_records(live)] == ["no-ts", "with-ts"]


def test_unreadable_file_does_not_abort_the_read(tmp_path, monkeypatch):
    live = tmp_path / "records.jsonl"
    live.write_text(json.dumps(_record("good", 1.0)) + "\n", encoding="utf-8")

    real_open = Path.open

    def broken_open(path, *args, **kwargs):
        if path == live:
            raise OSError("file vanished")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", broken_open)
    assert read_records(live) == []
