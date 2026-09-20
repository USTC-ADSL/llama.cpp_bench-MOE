import json
import math
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "profiles"))

import analyze_buffer_lifecycle as analysis


class BufferLifecycleAnalysisTests(unittest.TestCase):
    def test_rejects_mixed_schema_sessions(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        newer = self.valid_records(1, ["shared_dma"], "first-use")
        for record in newer:
            record["schema_version"] = 3
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "different schema"):
            analysis.validate_experiment(records + newer, [], 2)

    @staticmethod
    def sample(session, policy, case_name, pair, ready=5.0):
        populated = case_name != "alloc-only"
        cpu_private = policy in ("private_all_ready", "private_target_cpu")
        rpcmem = policy in ("shared_dma", "private_all_ready", "private_target_htp")
        gpu_upload = policy in ("private_all_ready", "private_target_gpu")
        record = {
            "event": "sample",
            "schema_version": 2,
            "status": "success",
            "measured": True,
            "policy": policy,
            "case": case_name,
            "slot_count": 1,
            "session": session,
            "pair": pair,
            "repeat_index": pair,
            "requested_payload_bytes": 6881280,
            "payload_allocation_count": 1 if policy == "shared_dma" else 3,
            "explicit_cross_allocation_transfer_bytes": (
                6881280 if policy == "shared_dma" else 3 * 6881280
            ),
            "htp_ref_count": 0,
            "htp_deref_count": 0,
            "allocation_identity_ok": True,
            "resources_released": case_name != "persistent-reuse",
            "population_breakdown_status": "measured" if populated else "not_applicable",
            "population_cpu_private_copy_us": 1.0 if populated and cpu_private else None,
            "population_rpcmem_copy_us": 1.0 if populated and rpcmem else None,
            "population_gpu_upload_us": 1.0 if populated and gpu_upload else None,
            "population_breakdown_residual_us": 0.1 if populated else None,
            "population_diagnostic_probe_us": 0.5 if populated else 0.0,
            "population_rusage_status": "measured" if populated else "not_applicable",
            "population_minor_faults": 2 if populated else None,
            "population_major_faults": 0 if populated else None,
            "population_user_cpu_us": 1.0 if populated else None,
            "population_system_cpu_us": 0.5 if populated else None,
            "population_cpu_us": 1.5 if populated else None,
        }
        for prefix, applicable in (("cpu_private", cpu_private), ("rpcmem", rpcmem)):
            measured = populated and applicable
            record[f"{prefix}_residency_status"] = "measured" if measured else "not_applicable"
            record[f"{prefix}_mincore_errno"] = 0
            for suffix, value in (
                ("page_size_bytes", 4096),
                ("pages_total", 1680),
                ("resident_pages_before", 0),
                ("resident_pages_after", 1680),
                ("newly_resident_pages", 1680),
            ):
                record[f"{prefix}_{suffix}"] = value if measured else None
        for metric in analysis.METRICS:
            record[metric] = ready if metric == "ready_total_us" else 1.0
        return record

    @staticmethod
    def validation(session, policy, case_name, phase):
        endpoints = set() if case_name == "alloc-only" else analysis.POLICY_ENDPOINTS[policy]
        expected_checksum = {"byte_sum": 1, "weighted_sum": 2, "nibble_sum": 3}
        order = "->".join(endpoint for endpoint in ("cpu", "gpu", "htp") if endpoint in endpoints)
        record = {
            "event": "validation",
            "schema_version": 2,
            "status": "success",
            "phase": phase,
            "session": session,
            "policy": policy,
            "case": case_name,
            "slot_count": 1,
            "checked_slots": 0 if case_name == "alloc-only" else 1,
            "verification_order": order,
            "checksums_match": True,
            "formal_sample_timing": False,
            "allocation_identity_ok": True,
            "htp_ref_count": 1 if "htp" in endpoints else 0,
            "htp_deref_count": 1 if "htp" in endpoints else 0,
            "resources_released": case_name != "persistent-reuse",
            "expected_checksum": expected_checksum,
        }
        for endpoint in ("cpu", "gpu", "htp"):
            record[f"{endpoint}_checksum"] = expected_checksum if endpoint in endpoints else None
        return record

    @classmethod
    def valid_records(cls, session, policies, case_name, repeat=1):
        records = [{
            "event": "manifest", "schema_version": 2, "status": "success", "session": session,
            "policies": policies, "cases": [case_name], "slot_counts": [1],
            "warmup": 0, "repeat": repeat,
            "checksum_in_formal_samples": False,
            "validation_phases": ["before", "after"],
            "population_diagnostics_in_formal_timing": False,
            "persistent_pool_setup_in_formal_samples": False,
            "persistent_pool_teardown_in_formal_samples": False,
        }]
        if case_name == "persistent-reuse":
            for policy in policies:
                records.append({
                    "event": "persistent_setup", "schema_version": 2, "status": "success",
                    "session": session, "policy": policy, "case": case_name, "slot_count": 1,
                    "resources_released": False, "htp_ref_count": 0, "htp_deref_count": 0,
                })
        records.extend(cls.validation(session, policy, case_name, "before") for policy in policies)
        for pair in range(repeat):
            records.extend(cls.sample(session, policy, case_name, pair, 5 + pair) for policy in policies)
        records.extend(cls.validation(session, policy, case_name, "after") for policy in policies)
        if case_name == "persistent-reuse":
            for policy in policies:
                records.append({
                    "event": "persistent_teardown", "schema_version": 2, "status": "success",
                    "session": session, "policy": policy, "case": case_name, "slot_count": 1,
                    "resources_released": True, "htp_ref_count": 2, "htp_deref_count": 2,
                })
        records.append({
            "event": "run_summary", "schema_version": 2, "status": "success", "session": session
        })
        return records

    def test_percentiles_and_summary(self):
        summary = analysis.summarize([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 2.5)
        self.assertAlmostEqual(summary["p50"], 2.5)
        self.assertAlmostEqual(summary["p95"], 3.85)
        self.assertAlmostEqual(summary["p99"], 3.97)

    def test_paired_ratio_is_deterministic(self):
        first = analysis.paired_ratio([1, 2, 3, 4], [2, 4, 6, 8], seed=7, bootstrap_iterations=200)
        second = analysis.paired_ratio([1, 2, 3, 4], [2, 4, 6, 8], seed=7, bootstrap_iterations=200)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.5)
        self.assertLessEqual(first[1], 0.5)
        self.assertGreaterEqual(first[2], 0.5)
        with self.assertRaises(ValueError):
            analysis.paired_ratio([1], [2], bootstrap_iterations=0)

    def test_linear_fit_and_break_even(self):
        intercept, slope, r_squared = analysis.linear_fit([(1, 3), (2, 5), (4, 9)])
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(r_squared, 1.0)
        self.assertAlmostEqual(analysis.break_even_uses(10, 4, 2, 5), 2.0)
        self.assertEqual(analysis.break_even_uses(4, 10, 2, 5), 0.0)
        self.assertIsNone(analysis.break_even_uses(10, 4, 5, 2))

    def test_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for session in range(3):
                records = self.valid_records(
                    session, ["shared_dma", "private_all_ready"], "first-use", repeat=3
                )
                for record in records:
                    if record.get("event") == "sample" and record.get("policy") == "private_all_ready":
                        record["ready_total_us"] = 10 + int(record["pair"])
                raw = root / f"raw-session-{session}.jsonl"
                raw.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            analysis.analyze(root, root, 11)
            self.assertTrue((root / "summary.csv").exists())
            self.assertIn("private_all_ready", (root / "summary.md").read_text(encoding="utf-8"))
            self.assertIn("better", (root / "analysis.md").read_text(encoding="utf-8"))

    def test_validation_rejects_failure_and_incomplete_session(self):
        records = [
            {
                "event": "manifest", "status": "success", "session": 0,
                "policies": ["shared_dma"], "cases": ["alloc-only"],
                "slot_counts": [1], "warmup": 0, "repeat": 1,
            },
            {"event": "sample_failure", "status": "failure", "session": 0, "error": "boom"},
            {"event": "run_summary", "status": "failure", "session": 0, "error": "boom"},
        ]
        with self.assertRaises(analysis.ExperimentValidationError):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_orphan_session_records(self):
        records = [
            {
                "event": "manifest", "status": "success", "session": 0,
                "policies": ["shared_dma"], "cases": ["alloc-only"],
                "slot_counts": [1], "warmup": 0, "repeat": 1,
            },
            {
                "event": "sample", "status": "success", "session": 0,
                "policy": "shared_dma", "case": "alloc-only", "slot_count": 1,
                "pair": 0, "repeat_index": 0, "measured": True,
                "htp_ref_count": 0, "htp_deref_count": 0,
                "allocation_identity_ok": True, "resources_released": True,
            },
            {"event": "run_summary", "status": "success", "session": 0},
            {
                "event": "persistent_setup", "status": "success", "session": 7,
                "policy": "shared_dma", "case": "persistent-reuse", "slot_count": 1,
            },
        ]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "without manifests"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_requires_standard_session_ids(self):
        records = [
            {
                "event": "manifest", "status": "success", "session": 7,
                "policies": ["shared_dma"], "cases": ["alloc-only"],
                "slot_counts": [1], "warmup": 0, "repeat": 1,
            },
            {
                "event": "sample", "status": "success", "session": 7,
                "policy": "shared_dma", "case": "alloc-only", "slot_count": 1,
                "pair": 0, "repeat_index": 0, "measured": True,
                "htp_ref_count": 0, "htp_deref_count": 0,
                "allocation_identity_ok": True, "resources_released": True,
            },
            {"event": "run_summary", "status": "success", "session": 7},
        ]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "expected session ids"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_requires_before_and_after_rounds(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        records = [
            record for record in records
            if not (record.get("event") == "validation" and record.get("phase") == "after")
        ]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "expected one after validation"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_round_inside_samples(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        before = next(i for i, record in enumerate(records) if record.get("phase") == "before")
        sample = next(i for i, record in enumerate(records) if record.get("event") == "sample")
        records[before], records[sample] = records[sample], records[before]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "before validation is not before samples"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_checksum_in_sample(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        sample = next(record for record in records if record.get("event") == "sample")
        sample["checksum_us"] = 10.0
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "contains validation work"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_legacy_schema(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        records[0]["schema_version"] = 1
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "requires schema_version 2"):
            analysis.validate_experiment(records, [], 1)

    def test_persistent_pool_setup_and_teardown_are_outside_samples(self):
        records = self.valid_records(0, ["shared_dma"], "persistent-reuse")
        analysis.validate_experiment(records, [], 1)
        setup_index = next(i for i, record in enumerate(records) if record.get("event") == "persistent_setup")
        setup = records.pop(setup_index)
        sample_index = next(i for i, record in enumerate(records) if record.get("event") == "sample")
        records.insert(sample_index + 1, setup)
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "persistent setup is inside"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_malformed_numeric_fields_without_crashing(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        sample = next(record for record in records if record.get("event") == "sample")
        sample["slot_count"] = "bad"
        with self.assertRaises(analysis.ExperimentValidationError):
            analysis.validate_experiment(records, [], 1)

    def test_validation_requires_resource_accounting_fields(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        sample = next(record for record in records if record.get("event") == "sample")
        del sample["payload_allocation_count"]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "invalid payload_allocation_count"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_requires_validation_contract_fields(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        validation = next(record for record in records if record.get("event") == "validation")
        del validation["allocation_identity_ok"]
        del validation["htp_ref_count"]
        del validation["htp_deref_count"]
        with self.assertRaises(analysis.ExperimentValidationError):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_checksum_mismatch(self):
        records = self.valid_records(0, ["private_target_cpu"], "first-use")
        validation = next(record for record in records if record.get("event") == "validation")
        validation["cpu_checksum"] = {"byte_sum": 9, "weighted_sum": 9, "nibble_sum": 9}
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "cpu checksum mismatch"):
            analysis.validate_experiment(records, [], 1)

    def test_validation_rejects_unrequested_persistent_events(self):
        records = self.valid_records(0, ["shared_dma"], "first-use")
        records.insert(-1, {
            "event": "persistent_setup", "schema_version": 2, "status": "success",
            "session": 0, "policy": "shared_dma", "case": "persistent-reuse", "slot_count": 1,
            "resources_released": False, "htp_ref_count": 0, "htp_deref_count": 0,
        })
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "outside its manifest cases"):
            analysis.validate_experiment(records, [], 1)

    def test_diagnostic_summary_reports_missing_samples(self):
        records = self.valid_records(0, ["shared_dma"], "first-use", repeat=2)
        samples = [record for record in records if record.get("event") == "sample"]
        samples[0]["population_rusage_status"] = "unavailable"
        for field in (
            "population_minor_faults", "population_major_faults", "population_user_cpu_us",
            "population_system_cpu_us", "population_cpu_us",
        ):
            samples[0][field] = None
        analysis.validate_experiment(records, [], 1)
        rows = analysis.build_summary_rows(analysis.measured_samples(records), 7)
        row = next(record for record in rows if record["metric"] == "population_minor_faults")
        self.assertEqual(row["available_count"], 1)
        self.assertEqual(row["missing_count"], 1)


if __name__ == "__main__":
    unittest.main()
