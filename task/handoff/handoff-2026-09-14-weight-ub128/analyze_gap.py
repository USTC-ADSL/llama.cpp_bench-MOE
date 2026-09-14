#!/usr/bin/env python3
"""Validate the routing-cache regression and corrected OpenCL capacity results."""
import gzip
import json
import math
import pathlib
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "evidence"


def read(name):
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as stream:
        return json.load(stream)


old = read("gap-deploy-hashes.json")
new = read("gap-fixed-deploy-hashes.json")
assert {name for name in old if old[name] != new[name]} == {"libggml-opencl.so"}
assert read("gap-routing-before.json")["exit"] != 0
assert read("gap-routing-after.json")["exit"] == 0
assert "original IDs after remapped IDs nonfinite=0 max_abs=0" in read("gap-routing-after.json")["stdout"]
final_regression = read("gap-routing-final.json")
assert final_regression["exit"] == 0
assert final_regression["stdout"].count("original IDs after remapped IDs nonfinite=0 max_abs=0") == 3
for tag in ("gap-route-512", "gap-nocb-512"):
    data = read(f"{tag}.json")
    assert read(f"{tag}.run.json")["exit"] == 0
    assert all(item["nonfinite_logits"] == len(item["logits"]) == 151936 for item in data["results"])
bad = read("gap-route-512.json")["results"]
before = bad[0]["weight_devices"]["GPUOpenCL"]
after = bad[1]["weight_devices"]["GPUOpenCL"]
collapse = {key: after[key] - before[key] for key in ("hits", "misses", "read_bytes", "operations")}
assert collapse["hits"] == 515 and collapse["misses"] == 4 and collapse["operations"] == 79

validation = []
for gib in (6, 5, 4):
    for prompt in (128, 512):
        tag = f"gap-fixed-{'g' + str(gib) + '-' if gib != 6 else ''}{prompt}"
        data = read(f"{tag}.json")
        run = read(f"{tag}.run.json")
        assert run["exit"] == 0 and data["prompt"] == prompt and not data["trace"]
        for name in ("libllama.so", "libggml-opencl.so"):
            assert new[name] in run["library_hashes"]
        reference_tag = "gap-inputs-full-128" if prompt == 128 else "gap-route-full-512"
        reference = read(f"{reference_tag}.json")
        for actual, expected in zip(data["results"], reference["results"]):
            assert actual["stage"] == expected["stage"] and actual["tokens"] == expected["tokens"]
            a, e = actual["logits"], expected["logits"]
            assert len(a) == len(e) == 151936 and actual["nonfinite_logits"] == 0
            assert all(value is not None and math.isfinite(value) for value in a)
            error = max(abs(x - y) for x, y in zip(a, e))
            rrmse = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, e)) / sum(y * y for y in e))
            assert error == 0 and rrmse == 0, (tag, actual["stage"], error, rrmse)
            validation.append({"weight_gib": gib, "prompt": prompt, "stage": actual["stage"],
                               "nonfinite": 0, "max_abs": error, "rrmse": rrmse,
                               "source": f"{tag}.json", "reference": f"{reference_tag}.json"})
assert len(validation) == 12

rows = []
conditions = {}
for gib in (None, 6, 5, 4):
    tag = "gap-fixed-speed" + ("-full" if gib is None else str(gib))
    data = read(f"{tag}.json")
    assert data["exit"] == 0 and not data["diagnostic"] and not data["profile"]
    assert data["weight_gib"] == gib and data["full_weight"] == (gib is None)
    conditions[tag] = {phase: data[phase] for phase in ("before", "after")}
    for phase in ("before", "after"):
        caps = data[phase]["cpu_frequency_caps"]
        assert "3532800" in caps and "4320000" in caps
    if gib is not None:
        prefix = "" if gib == 6 else f"g{gib}-"
        assert read(f"{tag}.config.json") == read(f"{prefix}gpu.config.json")
    assert {(item["n_prompt"], item["n_gen"]) for item in data["bench"]} == {(128, 0), (512, 0)}
    for item in data["bench"]:
        assert item["n_batch"] == 512 and item["n_ubatch"] == 128 and item["n_threads"] == 4
        assert item["devices"] == "GPUOpenCL" and item["n_gpu_layers"] == 99
        assert item["flash_attn"] == 1 and item["load_mode"] == "mmap" and item["lazy_mode"] == "off"
        samples = [item["n_prompt"] * 1e9 / ns for ns in item["samples_ns"]]
        assert len(samples) == 3 and all(value > 0 for value in samples)
        mean, sd = statistics.mean(samples), statistics.stdev(samples)
        assert abs(mean - item["avg_ts"]) < 0.001 and abs(sd - item["stddev_ts"]) < 0.001
        rows.append({"weight_gib": gib, "prompt": item["n_prompt"], "mean_ts": mean,
                     "sample_sd_ts": sd, "samples_ts": samples, "source": f"{tag}.json"})
        print(f"{str(gib or 'Full'):>4} PP{item['n_prompt']}: {mean:.6f} +/- {sd:.6f}")
result = {"device": "fd8657d6", "batch": 512, "ubatch": 128, "threads": 4,
          "numerical_validation": validation, "invalid_run_post_warmup_deltas": collapse,
          "corrected_speed_samples": 24, "rows": rows, "conditions": conditions,
          "original_hashes": old, "fixed_hashes": new}
(OUT / "gap-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print("PASS: 12 exact logit comparisons, 24 speed samples, old failure and fixed routing regression.")
