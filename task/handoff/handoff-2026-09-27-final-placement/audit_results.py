#!/usr/bin/env python3
"""Audit the fixed final cohort without selecting, fitting or changing any plan."""
import importlib.util
import json
import pathlib
import statistics

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
spec = importlib.util.spec_from_file_location("study", ROOT / "tests/run-moe-final-placement-study.py")
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def main():
    p = s.prepare()
    e = s.e
    model = e.read(s.MODEL)
    original = e.read(s.GUARD / "guarded-model.json")
    assert {k: v for k, v in model.items() if k != "calibration_guard"} == {k: v for k, v in original.items() if k != "calibration_guard"}
    for key in ("native", "profiles"):
        assert [x for x in model["calibration_guard"][key] if x["tokens"] in (128, 384)] == [x for x in original["calibration_guard"][key] if x["tokens"] in (128, 384)]
    for path, digest in model["calibration_guard"]["sources"].items():
        assert s.sha(ROOT / path) == digest, path
        assert "railway" not in path and "kitchen" not in path
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
    assert report["model_sha256"] == s.sha(s.MODEL) and len(report["cases"]) == 10
    for name in p["confirmation_prompts"]:
        for nt in p["confirmation_lengths"]:
            check(name, nt, f"{name}-{nt}-capture", "capture")
            plan_path = s.EV / f"{name}-{nt}-plan.json"
            plan = check(name, nt, f"{name}-{nt}-plan", "predict", TASK / f"{name}-{nt}-routes.json", cost=s.MODEL)
            s.parity(s.routes(name, nt), plan)
            case = next(c for c in report["cases"] if (c["prompt"], c["tokens"]) == (name, nt))
            assert case["execution_mode"] == plan["execution_mode"]
            for repeat in ("a", "b"):
                n = check(name, nt, f"{name}-{nt}-npu-{repeat}", "npu")
                a = check(name, nt, f"{name}-{nt}-auto-{repeat}", "auto", plan_path)
                ni = statistics.median(v["total_us"] for v in n["samples"])
                ai = statistics.median(v["total_us"] for v in a["samples"]) + plan["decision_device_total_us"]
                cell = next(r for r in case["repeats"] if r["repeat"] == repeat)
                assert abs(cell["native_ms"] - ni/1000) < 1e-9
                assert abs(cell["auto_inclusive_ms"] - ai/1000) < 1e-9
                assert abs(cell["throughput_gain_percent"] - 100*(ni/ai-1)) < 1e-9
    assert len(records) == 86
    assert len(list(s.EV.glob("*.run.json"))) == 86
    ordered = sorted(records, key=lambda r: r["start_ns"])
    assert all(a["end_ns"] <= b["start_ns"] for a, b in zip(ordered, ordered[1:]))
    s.write(TASK / "validation-audit.json", {"successful_processes": len(records), "model_sha256": s.sha(s.MODEL),
        "base_coefficients_unchanged": True, "legacy_128_384_unchanged": True, "no_confirmation_fitting_sources": True,
        "processes": ordered})
    print("PASS 86 processes: provenance, frozen model, decision parity, raw medians/costs, logits, pair accounting, allocation stability")


if __name__ == "__main__":
    main()
