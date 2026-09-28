#!/usr/bin/env python3
"""Summarize frozen CPU first-prefill cohorts; never pool validation timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    folder = args.directory
    plan = json.loads((folder / "confirmation-plan.json").read_text())
    result = {"plan_sha256": hashlib.sha256((folder / "confirmation-plan.json").read_bytes()).hexdigest(),
              "protocol": plan["protocol"], "prompts": {}, "numerics": {}}
    common_sources = None
    common_artifacts = None
    for prompt in plan["prompts"]:
        modes = {}
        for mode in plan["modes"]:
            rows = []
            for repetition in range(1, plan["rounds"] + 1):
                tag = f"cpu-confirm-{prompt}-{mode}-{repetition}"
                path = folder / f"{tag}.json"
                r = json.loads(path.read_text())
                manifest = json.loads((folder / f"{tag}.run.json").read_text())
                assert manifest["exit"] == 0 and manifest["serial"] == plan["device"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["output_sha256"]
                assert hashlib.sha256((folder / f"{tag}.logits.bin").read_bytes()).hexdigest() == manifest["logits_sha256"]
                assert not any(r.get(k, False) for k in ["cpu_reference_validation_only", "reference_placement_validation_only", "reference_packed_validation_only"])
                assert len(r["tokens"]) == plan["tokens"]
                if common_sources is None:
                    common_sources, common_artifacts = manifest["sources"], manifest["artifacts"]
                assert manifest["sources"] == common_sources and manifest["artifacts"] == common_artifacts
                row = {"tag": tag, "ms": r["total_us"] / 1000, "top1": r["top1"],
                       "load_ms": r["load_us"] / 1000, "maxrss_kib": r["maxrss_kib"]}
                if mode != "native":
                    assert r["cpu_experts"]["initial_prepared_experts"] == 0
                    assert r["prefetch"]["initial_prepared_experts"] == 0
                    assert r["cpu_experts"]["mode"] == ("canonical-mm" if mode == "grouped" else "canonical-id")
                    row["moe_ms"] = sum(l["total_us"] for l in r["layers"]) / 1000
                    row["non_moe_ms"] = row["ms"] - row["moe_ms"]
                    row["cpu_ms"] = sum(l["branches"][0]["total_us"] for l in r["layers"]) / 1000
                    row["layerwise_max_ms"] = sum(max(b["total_us"] for b in l["branches"]) for l in r["layers"]) / 1000
                    row["scoped_extra_mib"] = r["prefetch"]["allowance_minus_known_source_alias_bytes"] / 2**20
                    row["cpu_weight_workspace_bytes"] = r["cpu_experts"]["workspace_bytes"]
                    row["tails"] = [l["placement"]["name"] for l in r["layers"]]
                rows.append(row)
            modes[mode] = {"runs": rows, "median_ms": statistics.median(r["ms"] for r in rows),
                           "range_ms": [min(r["ms"] for r in rows), max(r["ms"] for r in rows)]}
            if mode != "native":
                for key in ["moe_ms", "non_moe_ms", "cpu_ms", "layerwise_max_ms", "scoped_extra_mib"]:
                    modes[mode]["median_" + key] = statistics.median(r[key] for r in rows)
        for mode in plan["modes"]:
            modes[mode]["throughput_vs_native_pct"] = (modes["native"]["median_ms"] / modes[mode]["median_ms"] - 1) * 100
        for baseline in ["original", "refit"]:
            modes["grouped"][f"throughput_vs_{baseline}_pct"] = (modes[baseline]["median_ms"] / modes["grouped"]["median_ms"] - 1) * 100
        result["prompts"][prompt] = modes
        reference = np.fromfile(folder / f"cpu-confirm-{prompt}-native-1.logits.bin", dtype=np.float32).astype(np.float64)
        errors = {}
        for mode in ["original", "refit", "grouped"]:
            actual = np.fromfile(folder / f"cpu-confirm-{prompt}-{mode}-1.logits.bin", dtype=np.float32).astype(np.float64)
            assert actual.shape == reference.shape and np.isfinite(actual).all()
            errors[mode] = {"relative_l2_vs_native": float(np.linalg.norm(actual-reference) / np.linalg.norm(reference)),
                            "max_abs_vs_native": float(np.max(np.abs(actual-reference))),
                            "top1_matches_native": bool(np.argmax(actual) == np.argmax(reference))}
        result["numerics"][prompt] = errors
    (folder / "confirmation-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for prompt, modes in result["prompts"].items():
        print(prompt, {mode: round(r["median_ms"], 3) for mode, r in modes.items()})
        print("grouped throughput vs native/original/refit:",
              [round(modes["grouped"][f"throughput_vs_{key}_pct"], 3) for key in ["native", "original", "refit"]])
    print("numerics:", result["numerics"])


if __name__ == "__main__":
    main()
