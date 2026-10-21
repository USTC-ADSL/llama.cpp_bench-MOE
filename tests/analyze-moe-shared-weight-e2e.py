#!/usr/bin/env python3
"""Audit frozen-input ABBA records and summarize E2E without summing overlaps."""
import hashlib
import gzip
import json
import pathlib
import statistics
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-10-21-shared-weight-moe-e2e"
EVIDENCE = TASK / "evidence"


def digest_lines(record):
    return {pathlib.PurePosixPath(line.split()[-1]).name: line.split()[0]
            for line in record["hashes"].strip().splitlines()}


def main():
    report = {"statistic": "mean of two process medians; each process has 2 warmups + 6 measured requests",
              "scope": "resident 24-layer MoE prefill including last-token logits; 768 tokens; c8-g15; fixed real-routing inputs",
              "limitations": ["2 prompts, 2 processes per condition; no decode or general quality claim",
                              "new path uses grouped 2D expert graphs; improvement is not isolated layout/kernel causality",
                              "backend cumulative durations overlap; never sum them to claim E2E",
                              "RSS excludes or differently charges DMA mappings; owner bytes exclude driver metadata and kernel scratch"],
              "prompts": {}}
    binary_hashes = None
    for prompt in ("wetland", "foundry"):
        groups = {}
        raw_reports = {}
        logits = {}
        tokens = None
        for variant in ("native", "shared"):
            processes = []
            for suffix in ("a", "b"):
                tag = f"{prompt}-{variant}-{suffix}"
                data = json.loads((EVIDENCE / (tag + ".json")).read_text())
                run = json.loads((EVIDENCE / (tag + ".run.json")).read_text())
                assert run["exit"] == 0 and len(data["samples"]) == 6 and data["tokens"] == 768
                assert data["current_route_scheduler"] is False
                assert data["activation_arena"] and data["activation_lifetimes"] and data["overlap_gather"]
                assert data["direct_callback_io"] and data["fused_host_reduce"] and data["host_reduce_threads"] == 1
                assert not data["npu_device_gather_effective"]
                if tokens is None:
                    tokens = data["token_ids"]
                assert data["token_ids"] == tokens
                hashes = digest_lines(run)
                hashes.pop("prompt.txt")
                if binary_hashes is None:
                    binary_hashes = hashes
                assert hashes == binary_hashes, "binaries changed within cohort"
                expected = hashlib.sha256(gzip.decompress((TASK / f"inputs/{prompt}-c8-g15.json.gz").read_bytes())).hexdigest()
                assert run["reference_hash"].split()[0] == expected
                values = np.fromfile(EVIDENCE / (tag + ".json.logits.bin"), dtype=np.float32).astype(np.float64)
                assert np.isfinite(values).all() and len(values) == 151936
                assert int(values.argmax()) == data["top1"]
                logits[tag] = values
                raw_reports[tag] = data
                if variant == "shared":
                    assert all(p["alias_checks_passed"] and p["weight_owner_allocations"] == 7 for p in data["placements"])
                    payload = sum(p["unique_weight_payload_bytes"] for p in data["placements"])
                    allocated = sum(p["unique_weight_allocation_bytes"] for p in data["placements"])
                else:
                    payload = sum(sum(p["resident_weight_bytes"]) for p in data["placements"])
                    allocated = sum(sum(p["resident_buffer_bytes"]) for p in data["placements"])
                stages = []
                for sample in data["samples"]:
                    layers = sample["layers"]
                    assert len(layers) == 24
                    assert sum(l["tokens"] for l in layers) == 23 * 768 + 1
                    ffn = sum(l["gather_us"] + l["expert_wall_us"] + l["combine_us"] for l in layers)
                    stages.append({"ffn_ms": ffn / 1000, "outside_ffn_ms": (sample["total_us"] - ffn) / 1000,
                                   "expert_wall_ms": sum(l["expert_wall_us"] for l in layers) / 1000,
                                   "combine_ms": sum(l["combine_us"] for l in layers) / 1000,
                                   **{f"{b}_cumulative_compute_ms": sum(l["backend_phases_us"][i][1] for l in layers) / 1000
                                      for i, b in enumerate(("cpu", "htp", "gpu"))}})
                process = {"tag": tag, "median_ms": data["timing"]["median_us"] / 1000,
                           "samples_ms": [s["total_us"] / 1000 for s in data["samples"]],
                           "resident_preparation_seconds": data["resident_preparation_us"] / 1e6,
                           "weight_payload_bytes": payload, "weight_allocation_bytes": allocated,
                           "activation_arena_bytes": sum(data["activation_arena_bytes"]),
                           "process_memory": data["process_memory"], "top1": data["top1"],
                           "stages_ms": {key: statistics.median(s[key] for s in stages) for key in stages[0]}}
                processes.append(process)
            assert np.array_equal(logits[f"{prompt}-{variant}-a"], logits[f"{prompt}-{variant}-b"]), "cross-process output differs"
            groups[variant] = {"processes": processes, "median_ms": statistics.mean(p["median_ms"] for p in processes),
                               "stages_ms": {key: statistics.mean(p["stages_ms"][key] for p in processes) for key in processes[0]["stages_ms"]}}
        native, shared = groups["native"]["median_ms"], groups["shared"]["median_ms"]
        a, b = logits[f"{prompt}-native-a"], logits[f"{prompt}-shared-a"]
        groups["latency_change_percent"] = (shared / native - 1) * 100
        groups["speedup"] = native / shared
        groups["prefill_tokens_per_second"] = {v: 768000 / groups[v]["median_ms"] for v in ("native", "shared")}
        groups["logits"] = {"nrmse": float(np.linalg.norm(a-b) / np.linalg.norm(a)),
                            "max_abs": float(np.max(np.abs(a-b))),
                            "cosine": float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)),
                            "top1_equal": int(a.argmax()) == int(b.argmax()),
                            "top10_native": np.argsort(a)[-10:][::-1].tolist(),
                            "top10_shared": np.argsort(b)[-10:][::-1].tolist()}
        groups["route_count_l1"] = sum(sum(abs(a-b) for a, b in zip(x["expert_token_counts"], y["expert_token_counts"]))
                                           for x, y in zip(raw_reports[f"{prompt}-native-a"]["samples"][0]["layers"],
                                                           raw_reports[f"{prompt}-shared-a"]["samples"][0]["layers"]))
        report["prompts"][prompt] = groups
    report["binary_hashes"] = binary_hashes
    (TASK / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    for prompt, values in report["prompts"].items():
        print(prompt, values["native"]["median_ms"], values["shared"]["median_ms"], values["latency_change_percent"], values["speedup"])


if __name__ == "__main__":
    main()
