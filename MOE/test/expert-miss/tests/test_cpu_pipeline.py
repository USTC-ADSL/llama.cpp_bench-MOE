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
            "--expert-count", "2",
            "--layout", "contiguous",
            "--output", str(pack),
        ])
        run([
            args.probe,
            "--dataset", str(pack),
            "--source-model", args.model,
            "--backend", "cpu",
            "--io-mode", "mmap-warm",
            "--miss-counts", "1,2",
            "--max-miss-count", "2",
            "--read-policy", "separate",
            "--repeat", "1",
            "--reserve-mib", "0",
            "--validate-compute", "true",
            "--output-jsonl", str(results),
        ])

        rows = read_jsonl(results)
        assert len(rows) == 2
        assert [row["miss_count"] for row in rows] == [1, 2]
        for row in rows:
            assert row["status"] == "ok"
            assert row["tensor_types"] == ["q4_0", "q4_0", "q4_1"]
            assert row["payload_bytes"] == 6881280*row["miss_count"]
            assert row["schema_version"] == 2
            assert row["io_mode"] == "mmap-warm"
            assert row["source_ingress"] == "mmap-shared-to-backend-copy"
            assert row["backend_buffer_semantics"] == "preallocated-expert-cache-slot"
            assert row["read_calls"] == 0
            assert row["mapped_range_count"] == 3*row["miss_count"]
            assert row["payload_validation_requested"] is False
            assert row["payload_validated"] is False
            assert row["checksum_ok"] is None
            assert row["payload_verify_us"] == 0
            assert row["slots_ready"] is True
            assert row["compute_validated"] is True
            assert row["validation_output_crc32"] != 0

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
            "--io-mode", "mmap-warm",
            "--miss-count", "1",
            "--max-miss-count", "1",
            "--expert-selection", "fixed",
            "--read-policy", "separate",
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
