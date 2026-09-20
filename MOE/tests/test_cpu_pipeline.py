#!/usr/bin/env python3

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_pack_entries(pack: Path) -> list[dict[str, int]]:
    metadata = pack.read_bytes()
    entry_count = struct.unpack_from("<Q", metadata, 80)[0]
    entries = []
    for index in range(entry_count):
        offset = 256 + index * 80
        entries.append({
            "expert_id": struct.unpack_from("<I", metadata, offset)[0],
            "payload_offset": struct.unpack_from("<Q", metadata, offset + 48)[0],
            "payload_bytes": struct.unpack_from("<Q", metadata, offset + 56)[0],
            "source_offset": struct.unpack_from("<Q", metadata, offset + 72)[0],
        })
    return entries


def make_sparse_source_from_pack(pack: Path, output: Path, expert_id: int) -> list[dict[str, int]]:
    all_entries = read_pack_entries(pack)
    entries = [entry for entry in all_entries if entry["expert_id"] == expert_id]
    source_size = max(entry["source_offset"] + entry["payload_bytes"] for entry in all_entries)
    with output.open("wb") as target:
        target.truncate(source_size)
    with pack.open("rb") as source, output.open("r+b") as target:
        for entry in entries:
            source.seek(entry["payload_offset"])
            payload = source.read(entry["payload_bytes"])
            assert len(payload) == entry["payload_bytes"]
            target.seek(entry["source_offset"])
            target.write(payload)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="expert-miss-test-") as temporary:
        root = Path(temporary)
        pack = root / "experts.pack"
        results = root / "results.jsonl"
        run([
            args.generator,
            "--model", args.model,
            "--layer", "0",
            "--expert-count", "4",
            "--layout", "contiguous",
            "--output", str(pack),
        ])
        run([
            args.probe,
            "--dataset", str(pack),
            "--source-model", args.model,
            "--backend", "cpu",
            "--miss-counts", "1,2,3",
            "--max-miss-count", "3",
            "--repeat", "1",
            "--reserve-mib", "0",
            "--validate-compute", "true",
            "--output-jsonl", str(results),
        ])

        rows = read_jsonl(results)
        assert len(rows) == 3
        assert [row["miss_count"] for row in rows] == [1, 2, 3]
        for row in rows:
            assert row["status"] == "ok"
            assert row["tensor_types"] == ["q4_0", "q4_0", "q4_1"]
            assert row["payload_bytes"] == 6881280*row["miss_count"]
            assert row["schema_version"] == 4
            assert row["io_mode"] == "buffered-pread"
            assert row["source_ingress"] == "buffered-pread-to-destination"
            assert row["read_policy"] == "separate"
            assert row["read_policy_status"] == "pending-design"
            assert row["backend_buffer_semantics"] == "preallocated-expert-cache-slot"
            assert row["read_calls"] == 3*row["miss_count"]
            assert row["mapped_range_count"] == 0
            assert row["direct_destination_bytes"] == row["payload_bytes"]
            assert row["staging_copy_bytes"] == 0
            assert row["payload_validation_requested"] is False
            assert row["payload_validated"] is False
            assert row["checksum_ok"] is None
            assert row["payload_verify_us"] == 0
            assert row["slots_ready"] is True
            assert row["compute_validated"] is True
            assert row["validation_output_crc32"] != 0

        sequential_results = root / "sequential.jsonl"
        run([
            args.probe,
            "--dataset", str(pack),
            "--source-model", args.model,
            "--miss-count", "3",
            "--max-miss-count", "3",
            "--expert-selection", "consecutive",
            "--tensor-kinds", "up,down",
            "--repeat", "1",
            "--reserve-mib", "0",
            "--validate-compute", "false",
            "--output-jsonl", str(sequential_results),
        ])
        sequential = read_jsonl(sequential_results)[0]
        assert sequential["status"] == "ok"
        assert sequential["tensor_kinds"] == ["up", "down"]
        assert sequential["read_calls"] == 6
        assert sequential["minimum_request_bytes"] > 2 * 1024 * 1024

        random_results = root / "random.jsonl"
        run([
            args.probe,
            "--dataset", str(pack),
            "--source-model", args.model,
            "--miss-count", "3",
            "--max-miss-count", "3",
            "--expert-selection", "random",
            "--tensor-kinds", "up,down",
            "--repeat", "1",
            "--reserve-mib", "0",
            "--validate-compute", "false",
            "--output-jsonl", str(random_results),
        ])
        random_row = read_jsonl(random_results)[0]
        assert random_row["status"] == "ok"
        assert random_row["read_calls"] == 6
        assert random_row["minimum_request_bytes"] > 2 * 1024 * 1024

        corrupted_source = root / "corrupted-source.gguf"
        source_entries = make_sparse_source_from_pack(pack, corrupted_source, expert_id=0)
        with corrupted_source.open("r+b") as target:
            target.seek(source_entries[0]["source_offset"])
            first = target.read(1)
            target.seek(source_entries[0]["source_offset"])
            target.write(bytes([first[0] ^ 0x01]))

        corrupted_results = root / "corrupted.jsonl"
        run([
            args.probe,
            "--dataset", str(pack),
            "--source-model", str(corrupted_source),
            "--miss-count", "1",
            "--max-miss-count", "1",
            "--expert-selection", "fixed",
            "--repeat", "1",
            "--reserve-mib", "0",
            "--verify-payload", "true",
            "--validate-compute", "false",
            "--output-jsonl", str(corrupted_results),
        ])
        corrupted_rows = read_jsonl(corrupted_results)
        assert len(corrupted_rows) == 1
        assert corrupted_rows[0]["status"] == "checksum-mismatch"
        assert corrupted_rows[0]["payload_validation_requested"] is True
        assert corrupted_rows[0]["payload_validated"] is True
        assert corrupted_rows[0]["checksum_ok"] is False
        assert corrupted_rows[0]["payload_verify_us"] > 0
        assert corrupted_rows[0]["slots_ready"] is False


if __name__ == "__main__":
    main()
