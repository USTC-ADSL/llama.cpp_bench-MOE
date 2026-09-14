#!/usr/bin/env python3
"""Validate and summarize the requested 6/5/4 GiB ubatch128 speed sweep."""
import argparse
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "evidence"


def read(name):
    return json.loads((OUT / name).read_text())


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--opencl", action="store_true", help="include the OpenCL supplement")
args = parser.parse_args()
assert read("g5-deploy-hashes.json") == read("deploy-hashes.json")
if args.opencl:
    assert read("gpu-deploy-hashes.json") == read("deploy-hashes.json")
rows = []
allocations = {}
conditions = {}
for gib in (6, 5, 4):
    prefix = "" if gib == 6 else f"g{gib}-"
    plans = read(f"{prefix}allocation-plan.json")
    assert plans["cpu"]["pinned"] == plans["htp"]["pinned"]
    assert plans["cpu"]["slots"] == plans["htp"]["slots"] >= 128
    assert plans["cpu"]["predicted_weight_bytes"] == plans["htp"]["predicted_weight_bytes"]
    planned_bytes = plans["cpu"]["predicted_weight_bytes"]
    assert 0 <= gib * 2**30 - planned_bytes < 3244032
    allocations[gib] = {"fixed_expert_banks": len(plans["cpu"]["pinned"]),
                        "slots_per_pool": plans["cpu"]["slots"],
                        "planned_weight_bytes": planned_bytes,
                        "planned_weight_gib": planned_bytes / 2**30,
                        "capacity_evidence": "device report" if gib == 6 else "same-layout allocation calculation"}
    if args.opencl:
        gpu_plan = read(f"{prefix}gpu-allocation-plan.json")["gpu"]
        assert gpu_plan["pinned"] == plans["cpu"]["pinned"]
        assert gpu_plan["slots"] == plans["cpu"]["slots"]
        assert 0 <= gib * 2**30 - gpu_plan["predicted_weight_bytes"] < 3244064
        allocations[gib]["opencl_planned_weight_bytes"] = gpu_plan["predicted_weight_bytes"]
        allocations[gib]["opencl_planned_weight_gib"] = gpu_plan["predicted_weight_bytes"] / 2**30
    for backend in (("cpu", "htp", "gpu") if args.opencl else ("cpu", "htp")):
        name = f"{prefix}{backend}-ub128"
        data = read(f"{name}.json")
        config = read(f"{name}.config.json")
        assert config == read(f"{prefix}{backend}.config.json")
        assert config["weights"] == read(f"{prefix}cpu.config.json")["weights"]
        assert "report" not in config
        assert data["exit"] == 0 and not data["diagnostic"] and not data["profile"]
        assert data.get("weight_gib", 6) == gib
        conditions[name] = {
            "battery_c_before": int(re.search(r"temperature: (\d+)", data["before"]["battery"])[1]) / 10,
            "battery_c_after": int(re.search(r"temperature: (\d+)", data["after"]["battery"])[1]) / 10,
            "frequency_caps_before": data["before"]["cpu_frequency_caps"],
            "frequency_caps_after": data["after"]["cpu_frequency_caps"],
            "max_sampled_swap_kib": max(int(sample.get("VmSwap", "0 kB").split()[0]) for sample in data["process_samples"]),
        }
        for phase in ("before", "after"):
            caps = conditions[name][f"frequency_caps_{phase}"]
            assert "3532800" in caps and "4320000" in caps, f"Check frequency caps: {name} {phase}"
        expected = {(threads, prompt) for threads in ((4, 8) if backend == "cpu" else (4,))
                    for prompt in (128, 512)}
        assert {(item["n_threads"], item["n_prompt"]) for item in data["bench"]} == expected
        for item in data["bench"]:
            assert item["n_batch"] == 512 and item["n_ubatch"] == 128 and item["n_gen"] == 0
            assert item["flash_attn"] == 1 and item["load_mode"] == "mmap" and item["lazy_mode"] == "off"
            assert not item["no_repack"]
            assert item["n_gpu_layers"] == (0 if backend == "cpu" else 99)
            if backend == "gpu":
                assert item["devices"] == "GPUOpenCL"
            samples = [item["n_prompt"] * 1e9 / ns for ns in item["samples_ns"]]
            assert len(samples) == 3 and all(value > 0 for value in samples)
            assert abs(statistics.mean(samples) - item["avg_ts"]) < 0.001
            assert abs(statistics.stdev(samples) - item["stddev_ts"]) < 0.001
            rows.append({"weight_gib": gib, "backend": backend, "threads": item["n_threads"],
                         "prompt": item["n_prompt"], "mean_ts": item["avg_ts"],
                         "sample_sd_ts": item["stddev_ts"], "samples_ts": samples,
                         "source": f"{name}.json", "reused": backend != "gpu" if args.opencl else gib == 6})
assert len(rows) == (24 if args.opencl else 18)
result = {"batch": 512, "ubatch": 128, "repetitions": 3, "configurations": len(rows),
          "new_samples": 18 if args.opencl else 36, "reused_samples": 54 if args.opencl else 18, "rows": rows,
          "allocations": allocations, "conditions": conditions}
output = "sweep-opencl-summary.json" if args.opencl else "sweep-summary.json"
(OUT / output).write_text(json.dumps(result, indent=2) + "\n")
print("GiB  Prompt  CPU4     CPU8     HTP4" + ("  OpenCL4" if args.opencl else "") + "  (tokens/s)")
for gib in (6, 5, 4):
    for prompt in (128, 512):
        group = {(row["backend"], row["threads"]): row["mean_ts"] for row in rows
                 if row["weight_gib"] == gib and row["prompt"] == prompt}
        gpu_column = f"  {group['gpu', 4]:7.3f}" if args.opencl else ""
        print(f"{gib:3d}  {prompt:6d}  {group['cpu', 4]:7.3f}  {group['cpu', 8]:7.3f}  {group['htp', 4]:7.3f}" + gpu_column)
