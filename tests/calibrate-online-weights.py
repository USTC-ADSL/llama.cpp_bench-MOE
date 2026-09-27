#!/usr/bin/env python3
"""Freeze branch corrections from development first-pass runs, never held-out routes."""
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-27-online-weights"
PARENT = ROOT / "task/handoff/handoff-2026-09-27-length-intervals/interval-model.json"
TAGS = ["full-online-dev-tail28", "full-online-dev-tail32"]


def main():
    output = OUT / "online-cost-model.json"
    assert not output.exists(), "calibration is frozen; use a new protocol for revisions"
    cpu, npu, overhead_cpu, overhead_npu, gpu = [], [], [], [], []
    evidence = {}
    for tag in TAGS:
        path = OUT / (tag + ".json")
        evidence[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        assert len(data["tokens"]) == 128 and data["mode"] == "online"
        # First two layers have cold setup/CPU ramp effects. Do not treat these
        # as stable layout slopes; their actual costs remain inside request time.
        for layer in data["layers"][2:23]:
            predicted = layer["decision"]["candidates"][0]["backend_us"]
            branches = layer["branches"]
            active_npu = sum(count > 0 and owner == 1 for count, owner in
                             zip(layer["counts"], layer["placement"]["owner"]))
            cpu.append(branches[0]["compute_us"] / predicted[0])
            npu.append((branches[1]["compute_us"] - (predicted[1] - 300 * active_npu)) / active_npu)
            overhead_cpu.append(branches[0]["total_us"] - branches[0]["compute_us"])
            overhead_npu.append(branches[1]["total_us"] - branches[1]["compute_us"])
            gpu.append(branches[2]["total_us"] - predicted[2])
    correction = {"tokens": 128, "candidate_tails": [0, 16, 24, 28, 32, 36],
                  "cpu_scale": statistics.median(cpu),
                  "cpu_overhead_us": statistics.median(overhead_cpu),
                  "npu_conversion_us_per_expert": max(0, statistics.median(npu)),
                  "npu_overhead_us": statistics.median(overhead_npu),
                  "gpu_preparation_us": max(0, statistics.median(gpu)),
                  "scope": "development-only robust branch correction; no held-out route capture; no speedup guarantee",
                  "evidence": evidence, "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest()}
    model = json.loads(PARENT.read_text())
    model["online_weight_calibration"] = correction
    output.write_text(json.dumps(model, separators=(",", ":")) + "\n")
    (OUT / "online-calibration.json").write_text(json.dumps(correction, indent=2) + "\n")
    print(json.dumps(correction, indent=2))


if __name__ == "__main__":
    main()
