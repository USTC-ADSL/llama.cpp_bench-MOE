#!/usr/bin/env python3
"""Audit the fixed final cohort without selecting, fitting or changing any plan."""
import importlib.util
import json
import pathlib
import statistics

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location("study", ROOT / "tests/run-moe-calibrate64-study.py")
supplement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supplement)
s = supplement.s


def main():
    p = s.prepare()
    e = s.e
    model = e.read(s.MODEL)
    original = e.read(supplement.PARENT)
    unchanged = set(original) - {"nodes", "calibration_guard"}
    assert all(model[k] == original[k] for k in unchanged)
    assert set(model) == set(original) | {"supplement64_scope"}
    assert {k:v for k,v in model["calibration_guard"].items() if k not in ("native","profiles","sources")} == {k:v for k,v in original["calibration_guard"].items() if k not in ("native","profiles","sources")}
    assert model["nodes"][1:] == original["nodes"] and model["base"] == original["base"]
    for key in ("native", "profiles"):
        assert [x for x in model["calibration_guard"][key] if x["tokens"] != 64] == original["calibration_guard"][key]
    assert not any(x["tokens"] == 64 and x["family"] == "gpu" for x in model["calibration_guard"]["profiles"])
    assert model["nodes"][0]["families"]["gpu"] == original["nodes"][0]["families"]["gpu"]
    for path, digest in model["calibration_guard"]["sources"].items():
        assert s.sha(ROOT / path) == digest, path
        assert "reservoir" not in path and "textile" not in path
    assert s.sha(s.MODEL) == e.read(TASK / "freeze.json")["model_sha256"]
    records = []

    def check(name, nt, tag, mode, plan=None, alternate=False, cost=None):
        d = s.verify_run(name, nt, tag, mode, plan, alternate, cost)
        m = e.read(s.EV / f"{tag}.run.json")
        row = {"tag": tag, "mode": mode, "resolved_mode": d.get("mode"), "start_ns": m["host_start_unix_ns"], "end_ns": m["host_end_unix_ns"],
               "report_sha256": s.sha(s.EV / f"{tag}.json"), "manifest_sha256": s.sha(s.EV / f"{tag}.run.json")}
        if mode not in ("capture", "predict"):
            assert len(d["token_ids"]) == nt
            times = [v["total_us"] for v in d["samples"]]
            assert abs(statistics.median(times) - d["timing"]["median_us"]) < 1e-6
            row.update(median_us=statistics.median(times), samples_us=times, logits_sha256=s.sha(s.EV / f"{tag}.json.logits.bin"),
                       major_faults=sum(v["major_faults"] for v in d["samples"]), process_status=d.get("process_status"))
            if d["mode"] == "custom":
                assert all(len(v["layers"]) == 24 for v in d["samples"])
                baseline_allocations = [v["allocations"] for v in d["samples"][0]["layers"]]
                phase_sums = []
                last = [0, 0, 0]
                for sample in d["samples"]:
                    assert [v["allocations"] for v in sample["layers"]] == baseline_allocations
                    phase = [0., 0., 0.]
                    for il, layer in enumerate(sample["layers"]):
                        assert layer["layer"] == il and layer["tokens"] == (1 if il == 23 else nt)
                        counts = layer["expert_token_counts"]
                        assert len(counts) == 60 and sum(counts) == 4 * layer["tokens"]
                        placement = d["placements"][il]
                        assert len(placement["owner"]) == 60 and all(b in (0, 1, 2) for b in placement["owner"])
                        pairs = [sum(c for c, b in zip(counts, placement["owner"]) if b == backend) for backend in range(3)]
                        assert pairs == layer["pairs"]
                        for b in range(3):
                            phase[b] += layer["backend_phases_us"][b][1]
                        if il < 23:
                            last[max(range(3), key=lambda b: layer["intervals_us"][b][1])] += 1
                    phase_sums.append(phase)
                row.update(activation_pool_mib=sum(d["arena_capacity_bytes"])/2**20,
                           phase_sum_median_ms=[statistics.median(v[b] for v in phase_sums)/1000 for b in range(3)],
                           last_backend_counts=last, allocations_stable=True,
                           all_round_logits_bitwise_equal=d["all_round_logits_bitwise_equal"])
                row["policy_means_ms"] = {}
                for policy, overlap in (("barrier", False), ("overlap", True)):
                    samples = [v for v in d["samples"] if v["overlap_gather"] == overlap]
                    if not samples:
                        continue
                    def mean(fn):
                        return statistics.mean(fn(v) for v in samples)/1000
                    row["policy_means_ms"][policy] = {
                        "total": mean(lambda v: v["total_us"]),
                        "gather": mean(lambda v: sum(l["gather_us"] for l in v["layers"])),
                        "expert_wall": mean(lambda v: sum(l["expert_wall_us"] for l in v["layers"])),
                        "combine": mean(lambda v: sum(l["combine_us"] for l in v["layers"])),
                        "outside_ffn": mean(lambda v: v["total_us"]-sum(l["gather_us"]+l["expert_wall_us"]+l["combine_us"] for l in v["layers"])),
                        "sum_layer_max_compute": mean(lambda v: sum(max(b[1] for b in l["backend_phases_us"]) for l in v["layers"]))}
        records.append(row)
        return d

    for name in p["calibration_prompts"]:
        for nt in p["calibration_lengths"]:
            if (name, nt) not in s.REUSE:
                check(name, nt, f"{name}-{nt}-capture", "capture")
            routes = s.routes(name, nt)
            native = []
            for repeat in ("a", "b"):
                tag = f"{name}-{nt}-native-{repeat}"
                d = check(name, nt, tag, "npu")
                assert d["token_ids"] == routes["token_ids"]
                native.append(e.raw(s.EV / f"{tag}.json.logits.bin"))
            assert native[0] == native[1]
            for t in (t for t in p["templates"] if t["tokens"] == nt):
                tag = f'{name}-{t["id"]}'
                check(name, nt, tag, "custom", TASK / f"{tag}-plan.json", alternate=True)
    report = e.read(TASK / "confirmation-report.json")
    assert report["model_sha256"] == s.sha(s.MODEL) and len(report["cases"]) == 2
    for name in p["confirmation_prompts"]:
        for nt in p["confirmation_lengths"]:
            check(name, nt, f"{name}-{nt}-capture", "capture")
            plan_path = s.EV / f"{name}-{nt}-plan.json"
            plan = check(name, nt, f"{name}-{nt}-plan", "predict", TASK / f"{name}-{nt}-routes.json", cost=s.MODEL)
            s.parity(s.routes(name, nt), plan)
            case = next(c for c in report["cases"] if (c["prompt"], c["tokens"]) == (name, nt))
            assert case["execution_mode"] == plan["execution_mode"]
            assert case["decision_us"] == plan["decision_device_total_us"]
            assert case["path_decision"] == plan["path_decision"]
            assert case["custom_candidates"] == plan["custom_candidates"]
            assert case["placements"] == [l["selected"] for l in plan["layers"]]
            assert case["execution_policy"] == plan["execution_policy"] and case["selected_profile"] == plan.get("selected_profile")
            for repeat in ("a", "b"):
                n = check(name, nt, f"{name}-{nt}-npu-{repeat}", "npu")
                a = check(name, nt, f"{name}-{nt}-auto-{repeat}", "auto", plan_path)
                ni = statistics.median(v["total_us"] for v in n["samples"])
                ai = statistics.median(v["total_us"] for v in a["samples"]) + plan["decision_device_total_us"]
                cell = next(r for r in case["repeats"] if r["repeat"] == repeat)
                assert abs(cell["native_ms"] - ni/1000) < 1e-9
                assert abs(cell["auto_inclusive_ms"] - ai/1000) < 1e-9
                assert abs(cell["throughput_gain_percent"] - 100*(ni/ai-1)) < 1e-9
                compute = statistics.median(v["total_us"] for v in a["samples"])
                for key, expected in (("auto_compute_ms", compute/1000), ("native_tokens_s", nt*1e6/ni),
                                      ("auto_tokens_s", nt*1e6/ai), ("activation_pool_mib", sum(a.get("arena_capacity_bytes", []))/2**20)):
                    assert abs(cell[key] - expected) < 1e-9, key
                for key, prediction, actual in (("native_prediction_error_percent", plan["path_decision"]["native_predicted_us"], ni),
                        ("selected_prediction_error_percent", plan["path_decision"]["custom_predicted_us"] if a["mode"] == "custom" else None, compute)):
                    expected = None if prediction is None else 100*(prediction/actual-1)
                    assert cell[key] is None if expected is None else abs(cell[key]-expected) < 1e-9
                outputs = {}
                for mode in ("npu", "auto"):
                    raw = e.raw(s.EV / f"{name}-{nt}-{mode}-{repeat}.json.logits.bin")
                    assert raw == e.raw(s.EV / f"{name}-{nt}-{mode}-a.json.logits.bin")
                    outputs[mode] = e.np.frombuffer(raw, dtype=e.np.float32).astype(e.np.float64)
                x, y = outputs["auto"], outputs["npu"]
                same = bool(e.np.array_equal(x, y))
                assert cell["same_native_logits"] == same and cell["same_top1"] == bool(x.argmax() == y.argmax())
                assert abs(cell["logit_nrmse_vs_native"] - float(e.np.linalg.norm(x-y)/e.np.linalg.norm(y))) < 1e-12
                cpu = e.np.frombuffer(e.raw(s.EV / f"{name}-{nt}-capture.json.logits.bin"), dtype=e.np.float32).astype(e.np.float64)
                assert cpu.shape == x.shape and e.np.isfinite(cpu).all()
                records[-1]["cpu_reference_comparison"] = {
                    "auto_top1": int(x.argmax()), "native_top1": int(y.argmax()), "cpu_top1": int(cpu.argmax()),
                    "auto_nrmse_vs_cpu": float(e.np.linalg.norm(x-cpu)/e.np.linalg.norm(cpu)),
                    "native_nrmse_vs_cpu": float(e.np.linalg.norm(y-cpu)/e.np.linalg.norm(cpu))}
                if a["mode"] == "npu":
                    assert same
    forced = 0
    for name in p["confirmation_prompts"]:
        plan = s.EV / f"{name}-64-plan.json"
        if e.read(plan)["execution_mode"] == "npu":
            for repeat in ("a", "b"):
                check(name,64,f"{name}-64-forced-{repeat}","custom",plan)
                forced += 1
    assert len(records) == 24 + forced
    manifests = list(s.EV.glob("*.run.json"))
    assert len(manifests) == 25 + forced
    failed = e.read(s.EV / "technical-a-64-gpu-tail24.run.json")
    assert failed["exit"] == 1
    assert b"ERROR: alternating callback logits differ" in e.raw(s.EV / "technical-a-64-gpu-tail24.log")
    legacy = []
    for path in sorted(supplement.PARENT.parent.glob("*-routes.json*")):
        c = e.read(path)
        if c["tokens"] == 64: continue
        a = e.make_plan(original,c,c["layers"],2000.)
        b = e.make_plan(model,c,c["layers"],2000.)
        assert a == b, str(path)
        legacy.append(str(path.relative_to(ROOT)))
    assert legacy
    ordered = sorted(records, key=lambda r: r["start_ns"])
    assert all(a["end_ns"] <= b["start_ns"] for a, b in zip(ordered, ordered[1:]))
    s.write(TASK / "validation-audit.json", {"successful_processes": len(records), "model_sha256": s.sha(s.MODEL),
        "base_coefficients_unchanged": True, "all_legacy_nodes_and_guards_unchanged": True, "legacy_identical_plans": legacy, "excluded_failed_processes": 1, "no_confirmation_fitting_sources": True,
        "processes": ordered})
    print("PASS", len(records), "successful processes; 1 retained correctness failure;", len(legacy), "unchanged legacy plans")


if __name__ == "__main__":
    main()
