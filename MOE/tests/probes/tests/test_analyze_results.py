#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_results import (
    attempt_rows,
    capacity_rows,
    descending_grid,
    error_rows,
    load_jsonl,
    synthesize_orphan_attempts,
    synthesize_partial_summaries,
)


class AnalyzeResultsTest(unittest.TestCase):
    def test_descending_grid_starts_near_upper_not_zero(self):
        self.assertEqual(descending_grid(3200, 2816, 64), [3136, 3072, 3008, 2944, 2880])

    def test_load_keeps_good_lines_and_reports_truncated_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.jsonl"
            path.write_text('{"record_type":"attempt","attempt_id":1}\n{"broken":\n', encoding="utf-8")
            records, errors = load_jsonl([path])
            self.assertEqual(len(records), 1)
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["line"], 2)

    def test_limit_error_is_joined_to_summary(self):
        attempt = {
            "record_type": "attempt",
            "attempt_id": 7,
            "mode": "htp",
            "variant": "raw-v79-aggregate-mapping",
            "ratio": "",
            "candidate_mib": 3200,
            "status": "mapping-failure",
            "errors": [{"stage": "htp.map", "api": "fastrpc_mmap", "code": 12, "symbol": "AEE_ERROR", "message": "limit"}],
            "created": {"htp": 3 * 1024**3},
            "host_touched": {"htp": 3 * 1024**3},
            "device_touched": {"htp": 0},
        }
        summary = {
            "record_type": "summary",
            "mode": "htp",
            "variant": "raw-v79-aggregate-mapping",
            "ratio": "",
            "first_unstable_attempt_id": 7,
            "first_unstable_mib": 3200,
            "last_stable_mib": 3072,
            "recommended_mib": 2608,
        }
        rows = capacity_rows([attempt], [summary])
        self.assertEqual(rows[0]["first_error_api"], "fastrpc_mmap")
        self.assertEqual(rows[0]["first_error_message"], "limit")
        errors = error_rows([attempt], "runtime.log", "logcat.log")
        self.assertEqual(errors[0]["code_hex"], "0x0000000c")
        self.assertEqual(errors[0]["runtime_log"], "runtime.log")

    def test_opencl_mmap_zero_copy_evidence_is_exported(self):
        attempt = {
            "record_type": "attempt",
            "attempt_id": 8,
            "mode": "opencl",
            "variant": "mmap-use-host-ptr-single",
            "candidate_mib": 64,
            "created": {"gpu": 64 * 1024**2},
            "host_touched": {"gpu": 64 * 1024**2},
            "host_verified": {"gpu": 64 * 1024**2},
            "device_touched": {"gpu": 64 * 1024**2},
            "checksums": {
                "gpu": 11,
                "gpu_host_input": 7,
                "gpu_device_input": 7,
                "gpu_host_output": 11,
                "gpu_mapped_output": 11,
            },
            "opencl": {
                "host_mmap": True,
                "map_sync": True,
                "direct_host_visible": True,
                "map_same_address": True,
                "zero_copy_passed": True,
            },
        }
        row = attempt_rows([attempt], "runtime.log", "logcat.log")[0]
        self.assertEqual(row["host_touched_gpu_mib"], 64.0)
        self.assertEqual(row["host_verified_gpu_mib"], 64.0)
        self.assertTrue(row["opencl_direct_host_visible"])
        self.assertTrue(row["opencl_map_same_address"])
        self.assertTrue(row["opencl_zero_copy_passed"])
        self.assertEqual(row["gpu_host_input_checksum"], row["gpu_device_input_checksum"])
        self.assertEqual(row["gpu_host_output_checksum"], row["gpu_device_output_checksum"])

    def test_opencl_copy_evidence_is_exported(self):
        attempt = {
            "record_type": "attempt",
            "attempt_id": 9,
            "mode": "opencl",
            "variant": "ordinary-single",
            "candidate_mib": 64,
            "created": {"gpu": 64 * 1024**2},
            "host_touched": {"gpu": 64 * 1024**2},
            "host_verified": {"gpu": 64 * 1024**2},
            "device_touched": {"gpu": 64 * 1024**2},
            "checksums": {
                "gpu": 13,
                "gpu_host_input": 7,
                "gpu_device_input": 7,
                "gpu_host_output": 13,
                "gpu_mapped_output": 0,
            },
            "opencl": {
                "host_mmap": False,
                "map_sync": False,
                "zero_copy_passed": False,
                "transfer": "copy",
                "copy": True,
                "copy_passed": True,
                "copy_host_mirror": True,
                "copy_host_mirror_bytes": 64 * 1024**2,
                "copy_write_calls": 8,
                "copy_read_calls": 8,
                "copy_write_bytes": 128 * 1024**2,
                "copy_read_bytes": 128 * 1024**2,
            },
        }
        row = attempt_rows([attempt], "runtime.log", "logcat.log")[0]
        self.assertEqual(row["opencl_transfer"], "copy")
        self.assertTrue(row["opencl_copy"])
        self.assertTrue(row["opencl_copy_passed"])
        self.assertTrue(row["opencl_copy_host_mirror"])
        self.assertEqual(row["opencl_copy_host_mirror_mib"], 64.0)
        self.assertFalse(row["opencl_zero_copy_passed"])
        self.assertEqual(row["opencl_copy_write_calls"], 8)
        self.assertEqual(row["opencl_copy_read_calls"], 8)
        self.assertEqual(row["opencl_copy_write_mib"], 128.0)
        self.assertEqual(row["opencl_copy_read_mib"], 128.0)
        self.assertEqual(row["gpu_host_input_checksum"], row["gpu_device_input_checksum"])
        self.assertEqual(row["gpu_host_output_checksum"], row["gpu_device_output_checksum"])

    def test_orphan_start_becomes_killed_attempt_and_partial_summary(self):
        records = [
            {
                "record_type": "attempt_start",
                "attempt_id": 9,
                "mode": "opencl",
                "variant": "ordinary-aggregate",
                "memory_policy": "swap-assisted",
                "swap_policy": "allow",
                "swap_floor_mib": 256,
                "ratio": "",
                "candidate_mib": 6400,
                "requested": {"gpu": 6400 * 1024**2},
            },
            {
                "record_type": "runner_event",
                "memory_policy": "swap-assisted",
                "swap_policy": "allow",
                "mode": "opencl",
                "event": "mode-exit",
                "exit_code": 137,
                "detail": "native probe exited",
            },
        ]
        attempts = synthesize_orphan_attempts(records)
        self.assertEqual(attempts[0]["status"], "killed")
        self.assertEqual(attempts[0]["memory_policy"], "swap-assisted")
        self.assertIn("no final fsynced", attempts[0]["errors"][0]["message"])
        summaries = synthesize_partial_summaries(attempts, [])
        self.assertFalse(summaries[0]["complete"])
        self.assertEqual(summaries[0]["first_unstable_attempt_id"], 9)
        self.assertEqual(summaries[0]["memory_policy"], "swap-assisted")

    def test_attempt_ids_from_different_mode_files_do_not_collide(self):
        records = [
            {"record_type": "attempt_start", "attempt_id": 1, "mode": "cpu", "_source_jsonl": "/tmp/cpu.jsonl"},
            {"record_type": "attempt", "attempt_id": 1, "mode": "cpu", "_source_jsonl": "/tmp/cpu.jsonl"},
            {"record_type": "attempt_start", "attempt_id": 1, "mode": "htp", "_source_jsonl": "/tmp/htp.jsonl"},
        ]
        attempts = synthesize_orphan_attempts(records)
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["mode"], "htp")

    def test_physical_and_swap_assisted_results_are_never_merged(self):
        attempts = [
            {
                "attempt_id": 1,
                "memory_policy": "physical-resident",
                "swap_policy": "reject",
                "mode": "cpu",
                "variant": "anonymous-resident",
                "ratio": "",
                "candidate_mib": 1024,
                "stable": True,
                "created": {"cpu": 1024 * 1024**2},
                "host_touched": {"cpu": 1024 * 1024**2},
                "device_touched": {},
                "baseline": {"swap_free": 4096 * 1024**2},
                "minimum": {"swap_free": 4064 * 1024**2},
            },
            {
                "attempt_id": 2,
                "memory_policy": "swap-assisted",
                "swap_policy": "allow",
                "mode": "cpu",
                "variant": "anonymous-resident",
                "ratio": "",
                "candidate_mib": 2048,
                "stable": True,
                "created": {"cpu": 2048 * 1024**2},
                "host_touched": {"cpu": 2048 * 1024**2},
                "device_touched": {},
                "baseline": {"swap_free": 4096 * 1024**2},
                "minimum": {"swap_free": 320 * 1024**2},
            },
        ]
        summaries = [
            {
                "memory_policy": "physical-resident",
                "mode": "cpu",
                "variant": "anonymous-resident",
                "ratio": "",
                "last_stable_mib": 1024,
            },
            {
                "memory_policy": "swap-assisted",
                "mode": "cpu",
                "variant": "anonymous-resident",
                "ratio": "",
                "last_stable_mib": 2048,
            },
        ]
        rows = capacity_rows(attempts, summaries)
        self.assertEqual(len(rows), 2)
        by_policy = {row["memory_policy"]: row for row in rows}
        self.assertEqual(by_policy["physical-resident"]["max_create_cpu_mib"], 1024.0)
        self.assertEqual(by_policy["swap-assisted"]["max_create_cpu_mib"], 2048.0)
        self.assertEqual(by_policy["swap-assisted"]["minimum_swap_free_mib"], 320.0)
        self.assertEqual(by_policy["swap-assisted"]["max_swap_free_delta_mib"], 3776.0)

    def test_swap_floor_error_keeps_policy_and_all_swap_fields(self):
        attempt = {
            "attempt_id": 11,
            "memory_policy": "swap-assisted",
            "swap_policy": "allow",
            "reserve_mib": 1024,
            "swap_tolerance_mib": 64,
            "swap_floor_mib": 256,
            "mode": "combined",
            "variant": "ordinary-pareto",
            "phase": "coarse-search",
            "baseline": {"swap_total": 4096 * 1024**2, "swap_free": 4000 * 1024**2},
            "minimum": {"swap_free": 240 * 1024**2},
            "errors": [
                {
                    "stage": "opencl.initial-touch",
                    "api": "SwapFree floor guard",
                    "symbol": "SWAPFREE_FLOOR",
                    "code": 240 * 1024**2,
                    "message": "SwapFree floor reached",
                    "swap_free": 240 * 1024**2,
                }
            ],
        }
        attempts_csv = attempt_rows([attempt], "runtime.log", "logcat.log")
        self.assertEqual(attempts_csv[0]["memory_policy"], "swap-assisted")
        self.assertEqual(attempts_csv[0]["baseline_swap_free_mib"], 4000.0)
        self.assertEqual(attempts_csv[0]["minimum_swap_free_mib"], 240.0)
        errors = error_rows([attempt], "runtime.log", "logcat.log")
        self.assertEqual(errors[0]["symbol"], "SWAPFREE_FLOOR")
        self.assertEqual(errors[0]["swap_floor_mib"], 256)
        self.assertEqual(errors[0]["swap_free_delta_mib"], 3760.0)


if __name__ == "__main__":
    unittest.main()
