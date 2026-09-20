#!/usr/bin/env python3
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "profiles"))
import analyze_expert_buffers as analysis


def session(sid=0, case="switch", source="gpu", target="htp"):
    directions = [("none", "gpu"), ("none", "htp")] if case == "prepare" else [
        (a, b) for a in ("gpu", "htp") for b in ("gpu", "htp") if case != "switch" or a != b]
    variants = ["shared_dma", "private_staging"] + (["private_map"] if case == "switch" else [])
    declared = [dict(key=f"{case}:{a}:{b}", case=case, source=a, target=b, variant=v, slot_count=0)
                for a, b in directions for v in variants]
    manifest = dict(event="manifest", benchmark="expert-buffer", schema_version=1, session=sid,
                    warmup=1, repeat=2, seed=20260914, slot_counts=[1], scenarios=declared,
                    slot_stride_bytes=analysis.STRIDE, repack_threads=1, pack_e3_crc=42, pack_e10_crc=43,
                    source_model="fixture", gpu_device="fixture", kernel="fixture", affinity=[0])
    result = [manifest]
    for scenario in declared:
        base = dict(scenario, slot_count=1)
        if case != "prepare":
            setup = dict(base, event="pool_setup", total_us=20, **{m: 0 for m in analysis.STAGES})
            setup["allocation_us"] = 10
            result.append(setup)
        result.append(dict(base, event="validation", status="success", phase="before"))
        for block in range(3):
            row = dict(base, event="sample", status="success", block=block, warmup=block == 0,
                       slot=0, active_slots=1, active_expert_bytes=analysis.STRIDE,
                       expert=3 if case in {"prepare", "switch"} else 10,
                       generation=2, old_generation=1, stale_ref_rejected=True,
                       payload_capacity_bytes=analysis.STRIDE * (1 if scenario["variant"] == "shared_dma" or case == "prepare" else 2),
                       repack_bytes=analysis.STRIDE, repack_scratch_bytes=0, repack_carry_bytes=128,
                       gpu_upload_bytes=0, gpu_download_bytes=0, cpu_copy_bytes=0, gpu_map_count=0, gpu_unmap_count=0,
                       gpu_events=0, profiled_gpu_events=0, resources_released=case == "prepare",
                       sync_ref_count=int(scenario["target"] == "htp"), sync_deref_count=int(scenario["target"] == "htp"),
                       read_bytes=analysis.STRIDE if case == "replace-storage" else 0,
                       physical_io_bytes=analysis.STRIDE if case == "replace-storage" else 0,
                       storage_io_verified=case == "replace-storage", page_cache_evict_ranges=3 if case == "replace-storage" else 0,
                       io_accounting_scope="thread" if case == "replace-storage" else "",
                       cold_ufs_guaranteed=False, **{m: 0.0 for m in analysis.METRICS})
            if scenario["variant"] == "private_map":
                row.update(cpu_copy_bytes=analysis.STRIDE, gpu_map_count=1, gpu_unmap_count=1)
            elif scenario["variant"] == "private_staging":
                if scenario["target"] == "gpu": row["gpu_upload_bytes"] = analysis.STRIDE
                elif case == "switch": row["gpu_download_bytes"] = analysis.STRIDE
            row.update(repack_us=10.0, state_publish_us=1.0, total_us=14.0 + sid)
            if case == "prepare": row["release_us"] = 1.0
            row["prepare_total_us"] = row["total_us"] - row["release_us"] if case == "prepare" else 0
            row["switch_total_us"] = row["total_us"] if case == "switch" else 0
            row["replacement_total_us"] = row["total_us"] if case.startswith("replace-") else 0
            row["accounted_us"] = sum(row[m] for m in analysis.STAGES)
            row["residual_us"] = row["total_us"] - row["accounted_us"]
            row["stage_status"] = {m: "measured" if row[m] else "not_applicable" for m in analysis.STAGES}
            result.append(row)
        result.append(dict(base, event="validation", status="success", phase="after"))
        result.append(dict(base, event="pool_teardown", release_us=2.0, resources_released=True))
    result.append(dict(event="run_summary", status="success"))
    return result


class AnalysisTests(unittest.TestCase):
    def test_all_workloads_and_map_pairs(self):
        for case in ("prepare", "switch", "replace-ram", "replace-storage"):
            _, samples, _ = analysis.validate_sessions([session(i, case) for i in range(3)])
            rows, ratios = analysis.build_summary(samples)
            self.assertTrue(rows and ratios)
            self.assertTrue(all(r["pairs"] == 6 for r in ratios))
            if case == "switch": self.assertEqual({r["private_variant"] for r in ratios}, {"private_map", "private_staging"})

    def test_bad_data_refuses_summary(self):
        for field, value in [("gpu_unmap_count", 99), ("sync_deref_count", 99), ("repack_bytes", 0),
                             ("total_us", 0), ("copy_us", float("nan")), ("generation", 1),
                             ("payload_capacity_bytes", 1), ("stale_ref_rejected", False),
                             ("resources_released", True)]:
            data = session()
            next(r for r in data if r["event"] == "sample")[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analysis.validate_sessions([data], 1)

    def test_storage_verification_required(self):
        data = session(case="replace-storage")
        next(r for r in data if r["event"] == "sample")["storage_io_verified"] = False
        with self.assertRaises(ValueError): analysis.validate_sessions([data], 1)
        data = session(case="replace-storage")
        next(r for r in data if r["event"] == "sample")["io_accounting_scope"] = "process"
        with self.assertRaises(ValueError): analysis.validate_sessions([data], 1)

    def test_missing_duplicate_and_failed_records(self):
        good = session()
        for event in ("sample", "validation", "pool_setup", "pool_teardown", "run_summary"):
            data = copy.deepcopy(good)
            data.remove(next(r for r in data if r["event"] == event))
            with self.subTest(event=event), self.assertRaises(ValueError): analysis.validate_sessions([data], 1)
        data = copy.deepcopy(good)
        data.insert(-1, next(r for r in data if r["event"] == "sample"))
        with self.assertRaises(ValueError): analysis.validate_sessions([data], 1)
        data = copy.deepcopy(good)
        data[-1]["status"] = "failure"
        with self.assertRaises(ValueError): analysis.validate_sessions([data], 1)

    def test_incompatible_sessions(self):
        data = [session(i) for i in range(3)]
        data[1][0]["pack_e10_crc"] = 9
        with self.assertRaises(ValueError): analysis.validate_sessions(data)
        with self.assertRaises(ValueError): analysis.validate_sessions([session(), session()], 2)

    def test_cli_and_stale_summary_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "expert-raw-session-0.jsonl"
            raw.write_text("\n".join(json.dumps(r) for r in session()), encoding="utf-8")
            out = root / "summary"
            argv = [str(root), "--output-dir", str(out), "--expected-sessions", "1"]
            analysis.main(argv)
            self.assertIn("private_map", (out / "summary.md").read_text())
            raw.write_text("broken", encoding="utf-8")
            with self.assertRaises(ValueError): analysis.main(argv)
            self.assertTrue(all(not (out / name).exists() for name in analysis.OUTPUTS))


if __name__ == "__main__":
    unittest.main()
