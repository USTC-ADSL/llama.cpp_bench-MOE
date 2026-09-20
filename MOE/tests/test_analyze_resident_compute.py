import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import analyze_resident_compute as analysis


def sample(mode, scope, token, session, measured, repeat_index):
    gpu, npu = analysis._expected_jobs(mode)
    serial = mode.endswith("_serial")
    return {
        "event": "resident_compute_sample",
        "status": "success",
        "mode": mode,
        "scope": scope,
        "token_num": token,
        "session": session,
        "repeat_index": repeat_index,
        "measured": measured,
        "compute_only_total_us": 100.0 if scope == "compute_only" else None,
        "resident_setup_compute_total_us": 110.0 if scope == "resident_setup_compute" else None,
        "total_time_us": 100.0 if scope == "compute_only" else 110.0,
        "compute_wall_us": 100.0,
        "gpu_explicit_sync_us": 2.0 if gpu and not serial else 0.0,
        "npu_explicit_sync_us": 0.5 if npu and not serial else 0.0,
        "gpu_job_count": gpu,
        "npu_job_count": npu,
        "gpu_completed_jobs": gpu,
        "npu_completed_jobs": npu,
        "gpu_blocking_compute_calls": gpu if serial else 0,
        "npu_blocking_compute_calls": npu if serial else 0,
        "gpu_async_compute_calls": 0 if serial else gpu,
        "npu_async_compute_calls": 0 if serial else npu,
        "gpu_explicit_sync_calls": 1 if gpu and not serial else 0,
        "npu_explicit_sync_calls": 1 if npu and not serial else 0,
        "gpu_compute_buffer_bytes": gpu * 1024,
        "npu_compute_buffer_bytes": npu * 2048,
    }


def session_records(session, warmup=2, repeat=10):
    modes = list(analysis.MODES)
    modes = modes[session % len(modes):] + modes[:session % len(modes)]
    records = [{
        "event": "manifest",
        "schema_version": 1,
        "session": session,
        "layer_count": 4,
        "experts_per_layer": 16,
        "resident_slot_count": analysis.SLOT_COUNT,
        "slot_stride_bytes": analysis.SLOT_STRIDE_BYTES,
        "arena_bytes": analysis.ARENA_BYTES,
        "token_nums": list(analysis.TOKENS),
        "modes": modes,
        "warmup": warmup,
        "repeat": repeat,
        "htp_graph_compute_async_flushes_and_waits": True,
    }]
    for mode in analysis.MODES:
        for token in analysis.TOKENS:
            records.append({
                "event": "resident_compute_validation",
                "status": "success",
                "session": session,
                "mode": mode,
                "token_num": token,
                "comparison_count": 64,
                "max_nmse": 1e-4,
                "nan_count": 0,
                "inf_count": 0,
                "slot_crc_unchanged": True,
            })
            for scope in analysis.SCOPES:
                for index in range(warmup):
                    records.append(sample(mode, scope, token, session, False, index))
                for index in range(repeat):
                    record = sample(mode, scope, token, session, True, index)
                    record["total_time_us"] += session + index
                    if scope == "compute_only":
                        record["compute_only_total_us"] = record["total_time_us"]
                        record["compute_wall_us"] = record["total_time_us"]
                    else:
                        record["resident_setup_compute_total_us"] = record["total_time_us"]
                    records.append(record)
    records.extend([
        {
            "event": "resident_compute_runtime",
            "status": "success",
            "session": session,
            "dsp_va": 0x1000,
            "htp_ref_count": 100,
            "htp_deref_count": 100,
            "htp_ref_deref_balanced": True,
            "slot_crc_unchanged": True,
        },
        {"event": "run_summary", "status": "success", "session": session},
    ])
    return records


class ResidentComputeAnalysisTests(unittest.TestCase):
    def test_explicit_smoke_token_matrix(self):
        records = [r for r in session_records(0, warmup=0, repeat=1) if r.get("token_num", 1) == 1]
        records[0]["token_nums"] = [1]
        analysis.validate_experiment(records, [], 1, 0, 1, (1,))
        rows = analysis.build_summary_rows(analysis.measured_samples(records))
        self.assertEqual(len(rows), 10)
        self.assertEqual(len(analysis.speedup_rows(rows, (1,))), 8)

    def test_rejects_mixed_schema_sessions(self):
        records = session_records(0) + session_records(1)
        for record in records:
            if record.get("event") == "manifest" and record["session"] == 1:
                record["schema_version"] = 2
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "different schema"):
            analysis.validate_experiment(records, [], 2)

    def setUp(self):
        self.records = [record for session in range(3) for record in session_records(session)]

    def test_complete_three_session_aggregate(self):
        self.assertEqual(analysis.validate_experiment(self.records, []), [0, 1, 2])
        rows = analysis.build_summary_rows(analysis.measured_samples(self.records))
        self.assertEqual(len(rows), 30)
        self.assertTrue(all(row["sample_count"] == 30 for row in rows))
        self.assertEqual(len(analysis.speedup_rows(rows)), 24)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for session in range(3):
                path = root / f"raw-session-{session}.jsonl"
                path.write_text(
                    "".join(json.dumps(record) + "\n" for record in session_records(session)),
                    encoding="utf-8",
                )
            analysis.analyze(root, root)
            with (root / "summary.csv").open(newline="", encoding="utf-8") as source:
                summary_rows = list(csv.DictReader(source))
            self.assertEqual(len(summary_rows), 30)
            self.assertIn("total_time_p99_us", summary_rows[0])
            self.assertIn("npu_async_batch", (root / "summary.md").read_text(encoding="utf-8"))

    def test_rejects_incomplete_or_duplicate_samples(self):
        records = copy.deepcopy(self.records)
        records.pop(next(
            index for index, record in enumerate(records)
            if record.get("event") == "resident_compute_sample"
            and record.get("session") == 1 and record.get("measured") is True
        ))
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "incomplete sample indices"):
            analysis.validate_experiment(records, [])

        records = copy.deepcopy(self.records)
        duplicate = next(
            record for record in records
            if record.get("event") == "resident_compute_sample"
            and record.get("session") == 0 and record.get("measured") is True
        )
        records.append(copy.deepcopy(duplicate))
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "incomplete sample indices"):
            analysis.validate_experiment(records, [])

    def test_rejects_validation_runtime_and_failure_records(self):
        mutations = []
        records = copy.deepcopy(self.records)
        next(record for record in records if record.get("event") == "resident_compute_validation")["comparison_count"] = 63
        mutations.append(records)
        records = copy.deepcopy(self.records)
        next(record for record in records if record.get("event") == "resident_compute_runtime")["htp_deref_count"] = 99
        mutations.append(records)
        records = copy.deepcopy(self.records)
        records.append({"event": "resident_compute_failure", "status": "failure", "session": 0, "error": "boom"})
        mutations.append(records)
        for records in mutations:
            with self.subTest():
                with self.assertRaises(analysis.ExperimentValidationError):
                    analysis.validate_experiment(records, [])

    def test_rejects_scope_specific_total_mismatch(self):
        records = copy.deepcopy(self.records)
        target = next(
            record for record in records
            if record.get("event") == "resident_compute_sample"
            and record.get("scope") == "compute_only"
        )
        target["compute_only_total_us"] = target["total_time_us"] + 1.0
        with self.assertRaisesRegex(
            analysis.ExperimentValidationError,
            "compute_only_total_us does not match total_time_us",
        ):
            analysis.validate_experiment(records, [])

    def test_rejects_session_or_manifest_mismatch(self):
        records = [record for record in self.records if record.get("session") != 2]
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "expected session ids"):
            analysis.validate_experiment(records, [])
        records = copy.deepcopy(self.records)
        next(record for record in records if record.get("event") == "manifest")["warmup"] = 1
        with self.assertRaisesRegex(analysis.ExperimentValidationError, "manifest warmup"):
            analysis.validate_experiment(records, [])


if __name__ == "__main__":
    unittest.main()
