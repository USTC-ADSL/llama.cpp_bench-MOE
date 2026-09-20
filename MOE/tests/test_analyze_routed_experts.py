import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).resolve().parents[1] / "experiments" / "analyze_routed_experts.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def session():
    rows = [{"event": "manifest", "schema_version": 1, "arena_bytes": 440401920,
             "tokens": [1], "repeat": 1, "warmup": 1, "session": "1", "environment": {}}]
    rows.append({"event": "model", "source_model": "phi", "packs": ["0", "1", "2", "3"]})
    for l in range(4):
        rows.append({"event": "fixture", "tokens": 1, "layer": l, "activation_crc": l,
                     "routes": [{"ids": [2*s, 2*s+1], "weights": [.6, .4]} for s in range(8)]})
    for mode in analysis.MODES:
        slices, end = [], 0
        groups = 2 if "hetero" in mode else 1
        for l in range(4):
            for k in range(3):
                for side in range(groups):
                    size = (16//groups) * (2457600 if k == 2 else 2211840)
                    backend = ("gpu" if side == 0 else "htp") if groups == 2 else ("gpu" if mode == analysis.MODES[0] else "htp")
                    slices.append({"layer": l, "kind": k, "offset": end, "bytes": size,
                                   "backend": backend, "canonical_crc": 123, "experts": list(range(side, 16, groups))})
                    end += size
        rows.append({"event": "layout", "mode": mode, "arena_crc": 123, "slices": slices})
        rows.append({"event": "integrity", "mode": mode, "arena_crc_unchanged": True})
        for stage in ("preflight", "post_core", "post_total"):
            for s in range(8 if stage == "preflight" else 3):
                for l in range(4):
                    rows.append({"event": "validation", "mode": mode, "tokens": 1, "fixture": s,
                                 "layer": l, "stage": stage, "passed": True, "nmse": 0.0001,
                                 "max_abs_error": .01,
                                 "nan_count": 0, "inf_count": 0})
        for s in range(3):
            for p in ("core", "total"):
                hetero = "hetero" in mode
                rows.append({"event": "sample", "mode": mode, "tokens": 1, "fixture": s, "pass": p,
                             "iteration": 0, "expert_core_us": 4., "four_layer_wall_us": 8.,
                             "moe_dispatch_total_us": 8. if p == "total" else None,
                             "expert_graphs": 8 if hetero else 4, "aggregate_graphs": 4 if hetero else 0,
                             "gpu_pairs_per_layer": 1 if hetero else 2 if mode == analysis.MODES[0] else 0,
                             "htp_pairs_per_layer": 1 if hetero else 2 if mode == analysis.MODES[1] else 0,
                             "layers": [{"layer": l, "expert_core_us": 1., "wall_us": 2.,
                                         "input_handoff_us": .5, "combine_us": .5} for l in range(4)]})
    rows.append({"event": "complete"})
    return rows


def detailed_session(enabled=True):
    rows = session()
    rows[0].update(schema_version=2, breakdown=enabled, htp_profile=False)
    for row in rows:
        if row["event"] != "sample":
            continue
        for layer in row["layers"]:
            layer["breakdown"] = None
            if not enabled:
                continue
            b = dict.fromkeys(analysis.BREAKDOWN, 0.)
            gpu = row["mode"] != analysis.MODES[1]
            htp = row["mode"] != analysis.MODES[0]
            b.update(gpu_compute_us=.4 if gpu else 0., htp_compute_us=.6 if htp else 0.,
                     gpu_graph_call_us=.2 if gpu else 0., gpu_graph_sync_us=.1 if gpu else 0.,
                     htp_graph_call_us=.5 if htp else 0., htp_graph_sync_us=.01 if htp else 0.)
            b["host_overlap_us"] = .3 if row["mode"] == analysis.MODES[3] else 0.
            b["dispatch_join_overhead_us"] = 1. - b["gpu_compute_us"] - b["htp_compute_us"] + b["host_overlap_us"]
            b.update(htp_output_get_us=.05, gpu_output_set_us=.15, gpu_aggregate_call_us=.1, gpu_aggregate_sync_us=.1)
            layer["breakdown"] = b
    return rows


class AnalysisTests(unittest.TestCase):
    def test_runtime_loader_schema(self):
        rows = detailed_session()
        rows[0]["schema_version"] = 3
        for row in rows:
            if row["event"] == "layout":
                for item in row["slices"]:
                    del item["canonical_crc"]
        analysis.validate(rows, formal=False)
        rows[0]["schema_version"] = 2
        with self.assertRaisesRegex(ValueError, "canonical fingerprint"):
            analysis.validate(rows, formal=False)

    def test_custom_token_matrix(self):
        rows = session()
        rows[0]["tokens"] = [2, 7, 64]
        expanded = []
        for row in rows:
            if "tokens" not in row or row["event"] == "manifest":
                expanded.append(row)
                continue
            for count in rows[0]["tokens"]:
                item = copy.deepcopy(row)
                item["tokens"] = count
                if item["event"] == "fixture":
                    for route in item["routes"]:
                        route["ids"] *= count
                        route["weights"] *= count
                elif item["event"] == "sample":
                    item["gpu_pairs_per_layer"] *= count
                    item["htp_pairs_per_layer"] *= count
                expanded.append(item)
        _, samples, _ = analysis.validate(expanded, formal=False)
        self.assertEqual(len(samples), 72)
        for tokens in ([], [0], [2, 2], [100001], [True]):
            broken = copy.deepcopy(expanded)
            broken[0]["tokens"] = tokens
            with self.assertRaises(ValueError):
                analysis.validate(broken, formal=False)

    def test_breakdown_accounting_and_disabled(self):
        for enabled in (False, True):
            analysis.validate(detailed_session(enabled), formal=False)
        for key, value in (("gpu_compute_us", float("nan")), ("gpu_graph_sync_us", 5.),
                           ("gpu_output_set_us", 5.), ("host_overlap_us", 5.),
                           ("dispatch_join_overhead_us", 1.)):
            rows = detailed_session()
            next(r for r in rows if r["event"] == "sample")["layers"][0]["breakdown"][key] = value
            with self.assertRaises(ValueError):
                analysis.validate(rows, formal=False)

    def test_writes_breakdown_and_rejects_mixed_profiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = detailed_session(), detailed_session(False)
            second[0]["session"] = "2"
            paths = [root / "one.jsonl", root / "two.jsonl"]
            for path, rows in zip(paths, (first, second)):
                path.write_text("\n".join(json.dumps(row) for row in rows))
            analysis.analyze(paths[:1], root / "summary", formal=False)
            self.assertIn("gpu_graph_sync_us", (root / "summary/breakdown.csv").read_text())
            self.assertIn("Heterogeneous breakdown", (root / "summary/summary.md").read_text())
            with self.assertRaises(ValueError):
                analysis.analyze(paths, root / "mixed", formal=False)

    def test_complete_smoke(self):
        _, samples, _ = analysis.validate(session(), formal=False)
        self.assertEqual(len(samples), 24)

    def test_rejects_missing_duplicate_nonfinite_and_single_sided(self):
        base = session()
        sample_index = next(i for i, r in enumerate(base) if r["event"] == "sample" and "hetero" in r["mode"])
        variants = [base[:-1], base[:sample_index] + base[sample_index+1:], base[:-1] + [base[sample_index], base[-1]]]
        for key, value in (("expert_core_us", float("nan")), ("gpu_pairs_per_layer", 0), ("expert_graphs", 64)):
            rows = copy.deepcopy(base)
            rows[sample_index][key] = value
            variants.append(rows)
        for rows in variants:
            with self.assertRaises(ValueError):
                analysis.validate(rows, formal=False)

    def test_rejects_failed_validation_and_crc(self):
        for event, key in (("validation", "passed"), ("integrity", "arena_crc_unchanged")):
            rows = session()
            next(r for r in rows if r["event"] == event)[key] = False
            with self.assertRaises(ValueError):
                analysis.validate(rows, formal=False)

    def test_smoke_is_not_formal(self):
        with self.assertRaises(ValueError):
            analysis.validate(session())

    def test_rejects_route_with_two_gpu_experts(self):
        rows = session()
        next(r for r in rows if r["event"] == "fixture")["routes"][0]["ids"] = [0, 2]
        with self.assertRaises(ValueError):
            analysis.validate(rows, formal=False)

    def test_rejects_missing_fingerprints(self):
        for event in ("model", "layout"):
            rows = [r for r in session() if r["event"] != event]
            with self.assertRaises(ValueError):
                analysis.validate(rows, formal=False)

    def test_rejects_bad_outer_and_layer_timing(self):
        for value in (None, float("nan"), -1., 1.):
            rows = session()
            next(r for r in rows if r["event"] == "sample")["four_layer_wall_us"] = value
            with self.assertRaises(ValueError):
                analysis.validate(rows, formal=False)
        rows = session()
        sample = next(r for r in rows if r["event"] == "sample")
        sample["layers"][0]["expert_core_us"] = -1.
        sample["layers"][1]["expert_core_us"] = 3.
        with self.assertRaises(ValueError):
            analysis.validate(rows, formal=False)

    def test_percentile_interpolates(self):
        self.assertEqual(analysis.percentile([10., 40., 20., 30.], .5), 25.)
        self.assertAlmostEqual(analysis.percentile([10., 20.], .95), 19.5)

    def test_writes_summary_and_rejects_mixed_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = session(), session()
            second[0]["session"] = "2"
            paths = [root / "one.jsonl", root / "two.jsonl"]
            for path, rows in zip(paths, (first, second)):
                path.write_text("\n".join(json.dumps(row) for row in rows))
            summary = analysis.analyze(paths, root / "summary", formal=False)
            self.assertEqual(len(summary), 120)
            self.assertTrue(all(row["samples"] == 2 for row in summary))
            self.assertEqual(json.loads((root / "summary/validation.json").read_text())["sample_count"], 48)
            next(r for r in second if r["event"] == "fixture")["activation_crc"] += 1
            paths[1].write_text("\n".join(json.dumps(row) for row in second))
            with self.assertRaises(ValueError):
                analysis.analyze(paths, root / "mixed", formal=False)


if __name__ == "__main__":
    unittest.main()
