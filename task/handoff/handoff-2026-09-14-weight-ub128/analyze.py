#!/usr/bin/env python3
"""Check recorded benchmark samples, matched allocation and baseline comparisons."""
import collections
import gzip
import json
import pathlib
import re
import statistics

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "evidence"
BASELINE = HERE.parent / "handoff-2026-09-14-k80-prefill/evidence"


def read(path):
    return json.loads(path.read_text())


baseline_hashes = {pathlib.Path(path).name: digest
                   for digest, path in (line.split(maxsplit=1)
                                        for line in (BASELINE / "host.sha256").read_text().splitlines())}
deployed = read(OUT / "deploy-hashes.json")
assert len(deployed) == 10
assert all(baseline_hashes[name] == digest for name, digest in deployed.items())
assert (OUT / "model.sha256").read_text().split()[0] == "8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473"

rows = []
conditions = {}
for name in ("cpu-ub128", "htp-ub128", "htp-ub4", "cpu-ub4", "cpu-ub4-cool"):
    data = read(OUT / f"{name}.json")
    config = read(OUT / f"{name}.config.json")
    assert data["exit"] == 0 and not data["diagnostic"] and not data["profile"]
    assert "report" not in config
    conditions[name] = {
        "battery_c_before": int(re.search(r"temperature: (\d+)", data["before"]["battery"])[1]) / 10,
        "battery_c_after": int(re.search(r"temperature: (\d+)", data["after"]["battery"])[1]) / 10,
        "max_sampled_rss_kib": max(int(sample.get("VmRSS", "0 kB").split()[0]) for sample in data["process_samples"]),
        "max_sampled_swap_kib": max(int(sample.get("VmSwap", "0 kB").split()[0]) for sample in data["process_samples"]),
        "frequency_caps_after": data["after"]["cpu_frequency_caps"],
    }
    for item in data["bench"]:
        assert item["n_batch"] == 512 and item["n_gen"] == 0
        assert item["n_ubatch"] == (128 if name.endswith("128") else 4)
        assert item["flash_attn"] == 1 and item["lazy_mode"] == "off"
        assert item["load_mode"] == "mmap" and not item["no_repack"]
        assert item["n_gpu_layers"] == (0 if name.startswith("cpu") else 99)
        samples = [item["n_prompt"] * 1e9 / ns for ns in item["samples_ns"]]
        assert len(samples) == 3 and all(sample > 0 for sample in samples)
        assert abs(statistics.mean(samples) - item["avg_ts"]) < 0.001
        assert abs(statistics.stdev(samples) - item["stddev_ts"]) < 0.001
        rows.append({"run": name, "backend": name.split("-")[0],
                     "threads": item["n_threads"], "prompt": item["n_prompt"],
                     "ubatch": item["n_ubatch"], "mean_ts": item["avg_ts"],
                     "sample_sd_ts": item["stddev_ts"], "samples_ts": samples})
assert len(rows) == 12
excluded = [row for row in rows if row["run"] == "cpu-ub4"]
rows = [row for row in rows if row["run"] != "cpu-ub4"]
assert len(rows) == 10 and len(excluded) == 2
for name, condition in conditions.items():
    if name != "cpu-ub4":
        assert "3532800" in condition["frequency_caps_after"] and "4320000" in condition["frequency_caps_after"]

configs = [read(OUT / f"{backend}.config.json") for backend in ("cpu", "htp")]
assert configs[0]["weights"] == configs[1]["weights"]
assert all(budget["expert_slots"] == 135 for config in configs for budget in config["budgets"].values())
diagnostics = {}
for backend in ("cpu", "htp"):
    data = read(OUT / f"diag-{backend}-ub128.json")
    assert data["exit"] == 0 and data["weight_bytes"] == 6440794112
    diagnostics[backend] = data["devices"]

comparisons = []
for row in rows:
    if row["ubatch"] != 128:
        continue
    candidates = read(BASELINE / f"{row['backend']}-ub128.json")
    baseline = next(item for item in candidates if item["n_threads"] == row["threads"]
                    and item["n_prompt"] == row["prompt"] and item["n_gen"] == 0)
    assert baseline["n_batch"] == 512 and baseline["n_ubatch"] == 128
    comparisons.append({"backend": row["backend"], "threads": row["threads"],
                        "prompt": row["prompt"], "full_mean_ts": baseline["avg_ts"],
                        "six_gib_mean_ts": row["mean_ts"],
                        "retained_throughput": row["mean_ts"] / baseline["avg_ts"]})

microbatch = []
for row in rows:
    if row["ubatch"] != 4:
        continue
    large = next(item for item in rows if item["backend"] == row["backend"]
                 and item["threads"] == row["threads"] and item["prompt"] == row["prompt"]
                 and item["ubatch"] == 128)
    microbatch.append({"backend": row["backend"], "threads": row["threads"],
                       "prompt": row["prompt"], "ub128_over_ub4": large["mean_ts"] / row["mean_ts"]})

counts = collections.Counter()
cache_counts = collections.Counter()
layers = collections.defaultdict(collections.Counter)
excerpts = []
log = OUT / "profile-htp-ub128.log"
opener = open if log.exists() else gzip.open
if not log.exists():
    log = log.with_suffix(".log.gz")
with opener(log, "rt") as source:
    for number, line in enumerate(source, 1):
        if "profile-op MUL_MAT_ID" not in line:
            continue
        kernel = line.split("|")[5].split()[0]
        layer = re.search(r" -> ffn_moe_(?:up|gate|down)-(\d+)\|", line)[1]
        counts[kernel] += 1
        layers[layer][kernel] += 1
        if "ffn_exps_cache_" in line:
            cache_counts[kernel] += 1
        if layer in ("0", "16", "17", "23"):
            excerpts.append(f"{number}:{line}")
assert counts == {"hmx-tiled": 53, "hvx-tiled": 3}
assert cache_counts == {"hmx-tiled": 19, "hvx-tiled": 3}
assert layers["23"] == {"hvx-tiled": 3}
probe = read(OUT / "slice-results.json")
assert len(probe["cases"]) == 2
assert all(case["exit"] == 0 and "max_abs=0" in case["stdout"] for case in probe["cases"])
(OUT / "profile-excerpts.txt").write_text("".join(excerpts))

result = {"formal_configurations": len(rows), "formal_samples": len(rows) * 3,
          "excluded_throttled_rows": excluded, "total_validated_samples": 36,
          "rows": rows, "baseline_comparisons": comparisons, "microbatch_comparisons": microbatch,
          "diagnostic_devices": diagnostics, "conditions": conditions,
          "profile_expert_counts": counts, "profile_cache_expert_counts": cache_counts,
          "profile_layer_counts": layers,
          "diagnostic_note": "Counters aggregate no-warmup pp128 and pp512; excluded from formal timing."}
(OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "formal_configurations", "formal_samples", "total_validated_samples",
    "microbatch_comparisons", "profile_expert_counts", "profile_cache_expert_counts")}, indent=2))
