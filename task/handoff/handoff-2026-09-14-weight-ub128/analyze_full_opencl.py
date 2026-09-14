#!/usr/bin/env python3
"""Validate the full OpenCL baseline against the recorded capacity protocol."""
import json
import pathlib
import re
import shlex
import statistics

OUT = pathlib.Path(__file__).resolve().parent / "evidence"


def read(name):
    return json.loads((OUT / name).read_text())


def command_tokens(command):
    tokens = shlex.split(command)
    if "--weight-config" in tokens:
        index = tokens.index("--weight-config")
        del tokens[index:index + 2]
    return tokens


assert read("full-gpu-deploy-hashes.json") == read("gpu-deploy-hashes.json")
reference = read("gpu-ub128.json")
rows = []
conditions = {}
for tag, expected in (("full-gpu-ub128", {(128, 0), (512, 0)}),
                      ("full-gpu-tg128", {(0, 128)})):
    data = read(f"{tag}.json")
    assert data["exit"] == 0 and data["full_weight"] and data["weight_gib"] is None
    assert not data["diagnostic"] and not data["profile"]
    assert "--weight-config" not in shlex.split(data["command"])
    actual_command = command_tokens(data["command"])
    if tag == "full-gpu-tg128":
        actual_command[actual_command.index("-p") + 1] = "128,512"
        actual_command[actual_command.index("-n") + 1] = "0"
    assert actual_command == command_tokens(reference["command"])
    assert data["bench"] == read(f"{tag}.stdout")
    assert {(item["n_prompt"], item["n_gen"]) for item in data["bench"]} == expected
    conditions[tag] = {}
    for phase in ("before", "after"):
        snapshot = data[phase]
        caps = snapshot["cpu_frequency_caps"]
        assert "3532800" in caps and "4320000" in caps
        conditions[tag][phase] = {
            "battery_c": int(re.search(r"temperature: (\d+)", snapshot["battery"])[1]) / 10,
            "cpu_frequency_caps": caps,
        }
    for item in data["bench"]:
        for key in ("build_commit", "build_number", "model_filename", "model_type", "model_size",
                    "model_n_params", "gpu_info", "devices", "n_batch", "n_ubatch", "n_threads",
                    "n_gpu_layers", "n_cpu_moe", "flash_attn", "load_mode", "lazy_mode", "no_repack",
                    "type_k", "type_v", "no_op_offload", "cpu_mask", "cpu_strict", "poll",
                    "split_mode", "main_gpu", "no_kv_offload", "tensor_split", "tensor_buft_overrides",
                    "embeddings", "no_host", "fit_target", "fit_min_ctx", "n_depth"):
            assert item[key] == reference["bench"][0][key], key
        assert item["weight_config"] in ("", "none")
        samples = [(item["n_prompt"] + item["n_gen"]) * 1e9 / ns for ns in item["samples_ns"]]
        assert len(samples) == 3 and all(value > 0 for value in samples)
        mean, sd = statistics.mean(samples), statistics.stdev(samples)
        assert abs(mean - item["avg_ts"]) < 0.001
        assert abs(sd - item["stddev_ts"]) < 0.001
        row = {"prompt": item["n_prompt"], "gen": item["n_gen"], "mean_ts": mean,
               "sample_sd_ts": sd, "samples_ts": samples, "source": f"{tag}.json"}
        if item["n_prompt"]:
            row["capacity_comparison"] = []
            for gib, prefix in ((6, ""), (5, "g5-"), (4, "g4-")):
                limited = next(b for b in read(f"{prefix}gpu-ub128.json")["bench"]
                               if b["n_prompt"] == item["n_prompt"])
                row["capacity_comparison"].append({"weight_gib": gib, "mean_ts": limited["avg_ts"],
                    "sample_sd_ts": limited["stddev_ts"], "retained_fraction": limited["avg_ts"] / mean})
        rows.append(row)
        print(f"pp{item['n_prompt']} tg{item['n_gen']}: {mean:.6f} +/- {sd:.6f} tokens/s; samples={samples}")
result = {"device": "fd8657d6", "backend": "GPUOpenCL", "batch": 512, "ubatch": 128,
          "threads": 4, "repetitions": 3, "new_samples": 9, "weight_manager": False,
          "rows": rows, "conditions": conditions}
(OUT / "full-opencl-summary.json").write_text(json.dumps(result, indent=2) + "\n")
print("Validated 9 full-weight samples and matching capacity-run options and deployment hashes.")
