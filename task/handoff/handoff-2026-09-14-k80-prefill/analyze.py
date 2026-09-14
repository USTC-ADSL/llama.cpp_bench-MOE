#!/usr/bin/env python3
import collections
import gzip
import json
import pathlib
import re
import statistics

root = pathlib.Path(__file__).resolve().parent
evidence = root / "evidence"
rows = []
for name in ("cpu-ub128", "htp-ub128", "htp-ub512", "cpu-ub512"):
    data = json.loads((evidence / (name + ".json")).read_text())
    for entry in data:
        samples = entry["samples_ts"]
        assert len(samples) == 3 and all(value > 0 for value in samples)
        assert abs(statistics.mean(samples) - entry["avg_ts"]) < 0.001
        assert abs(statistics.stdev(samples) - entry["stddev_ts"]) < 0.001
        assert entry["n_gpu_layers"] == (0 if name.startswith("cpu") else 99)
        rows.append({"run": name, "backend": name.split("-")[0],
                     "threads": entry["n_threads"], "batch": entry["n_batch"],
                     "ubatch": entry["n_ubatch"], "prompt": entry["n_prompt"],
                     "generation": entry["n_gen"], "mean_ts": entry["avg_ts"],
                     "sample_sd_ts": entry["stddev_ts"], "samples_ts": samples})
assert len(rows) == 18

comparisons = []
for htp in rows:
    if htp["backend"] != "htp" or not htp["prompt"]:
        continue
    cpu = [row for row in rows if row["backend"] == "cpu"
           and row["ubatch"] == htp["ubatch"] and row["prompt"] == htp["prompt"]]
    assert len(cpu) == 2
    comparisons.append({"prompt": htp["prompt"], "ubatch": htp["ubatch"],
                        "htp_over_cpu": {str(row["threads"]): htp["mean_ts"] / row["mean_ts"]
                                         for row in cpu}})

labels = ("pp128", "pp512", "tg4")
phase = 0
counts = {label: collections.Counter() for label in labels}
layers = {label: collections.defaultdict(collections.Counter) for label in labels}
pattern = re.compile(r"profile-op (MUL_MAT_ID(?:_NX)?)\|blk\.(\d+)\.")
log = evidence / "htp-profile.log"
opener = open if log.exists() else gzip.open
if not log.exists():
    log = log.with_suffix(".log.gz")
with opener(log, "rt") as source:
    for line in source:
        match = pattern.search(line)
        if match:
            assert phase < len(labels)
            kernel = line.split("|")[5].split()[0]
            counts[labels[phase]][kernel] += 1
            layers[labels[phase]][match.group(2)][kernel] += 1
        if line.startswith("llama_perf_context_print:") and "load time" in line:
            phase += 1
assert phase == 3
for label in labels[:2]:
    assert counts[label] == {"hmx-tiled": 46, "hvx-tiled": 2}
    for layer in range(23):
        assert layers[label][str(layer)] == {"hmx-tiled": 2}
    assert layers[label]["23"] == {"hvx-tiled": 2}
assert counts["tg4"] == {"hvx-tiled": 192}
assert all(layers["tg4"][str(layer)] == {"hvx-tiled": 8} for layer in range(24))

result = {"formal_configurations": len(rows), "formal_samples": len(rows) * 3,
          "rows": rows, "comparisons": comparisons,
          "profile_excluded_from_performance": True,
          "expert_profile_counts": counts, "expert_profile_layers": layers,
          "profile_note": "Completed PROFILE=1 kernel records, not independent hardware PMU trace. "
                          "Last-layer prefill FFN keeps one output token and uses HVX."}
(evidence / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"configurations": len(rows), "samples": len(rows) * 3,
                  "comparisons": comparisons, "expert_profile_counts": counts}, indent=2))
