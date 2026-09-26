#!/usr/bin/env python3
"""Audit transferred placement screening; report numeric drift, never hide losses."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


def read(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + ".gz")
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()
    evidence = args.task_dir / "evidence"
    cases = []
    binary_hashes = set()
    for prompt in ("technical", "narrative"):
        for nt in (64, 128, 256, 512):
            rows = {}
            logits = {}
            inputs = None
            for mode, cohort in (("cpu", "a"), ("gpu", "a"), ("npu", "a"),
                                 ("custom", "a"), ("custom", "b"), ("npu", "b")):
                tag = f"{prompt}-{nt}-{mode}-{cohort}"
                try:
                    manifest = json.loads(read(evidence / f"{tag}.run.json"))
                except FileNotFoundError:
                    continue
                if "exit" not in manifest:
                    continue
                if manifest["exit"] != 0:
                    rows[f"{mode}-{cohort}"] = {"failed": True, "exit": manifest.get("exit")}
                    continue
                d = json.loads(read(evidence / f"{tag}.json"))
                assert d["tokens"] == nt and d["mode"] == mode
                assert len(d["token_ids"]) == nt
                if inputs is None:
                    inputs = d["token_ids"]
                assert inputs == d["token_ids"], f"input mismatch {tag}"
                binary_hashes.add(tuple(line for line in manifest["hashes"].splitlines() if not line.endswith("/prompt.txt")))
                values = np.frombuffer(read(evidence / f"{tag}.json.logits.bin"), dtype="<f4").astype(np.float64)
                assert len(values) and np.isfinite(values).all(), f"invalid logits {tag}"
                logits[f"{mode}-{cohort}"] = values
                samples = d["samples"]
                assert len(samples) == 4 and all(s["round"] == i for i, s in enumerate(samples))
                row = {"median_ms": statistics.median(s["total_us"] for s in samples)/1000,
                       "mean_ms": statistics.mean(s["total_us"] for s in samples)/1000,
                       "minor_faults": sum(s["minor_faults"] for s in samples),
                       "major_faults": sum(s["major_faults"] for s in samples),
                       "top1": int(np.argmax(values)), "activation_mib": d["expert_activation_buffer_bytes"]/2**20}
                row["tokens_s"] = nt * 1000 / row["median_ms"]
                if mode == "custom":
                    assert d["transfer_placement"] and d["activation_arena"] and d["fused_host_reduce"]
                    assert d["all_round_logits_bitwise_equal"]
                    for s in samples:
                        assert len(s["layers"]) == 24
                        assert [l["allocations"] for l in s["layers"]] == [l["allocations"] for l in samples[0]["layers"]]
                        for il, l in enumerate(s["layers"]):
                            assert l["tokens"] == (1 if il == 23 else nt)
                            assert not l["route_reference_available"]
                            counts = l["expert_token_counts"]
                            assert len(counts) == 60 and sum(counts) == 4*l["tokens"]
                            owner = d["placements"][il]["owner"]
                            assert l["pairs"] == [sum(c for c, b in zip(counts, owner) if b == backend) for backend in range(3)]
                    row["phase_mean_ms"] = {key.removesuffix("_us")+"_ms": statistics.mean(sum(l[key] for l in s["layers"]) for s in samples)/1000
                                             for key in ("gather_us", "expert_wall_us", "combine_us")}
                    row["layers"] = []
                    for il, placement in enumerate(d["placements"]):
                        layer_samples = [s["layers"][il] for s in samples]
                        row["layers"].append({"layer": il, "shared": placement["shared"],
                            "owner": placement["owner"], "counts": layer_samples[0]["expert_token_counts"],
                            "pairs": layer_samples[0]["pairs"],
                            "expert_wall_ms": statistics.mean(l["expert_wall_us"] for l in layer_samples)/1000,
                            "backend_compute_ms": [statistics.mean(l["backend_phases_us"][b][1] for l in layer_samples)/1000 for b in range(3)]})
                rows[f"{mode}-{cohort}"] = row
            if "cpu-a" in logits:
                reference = logits["cpu-a"]
                for key, values in logits.items():
                    assert values.shape == reference.shape
                    rows[key]["cpu_logit_nrmse"] = float(np.linalg.norm(values-reference)/max(np.linalg.norm(reference), 1e-30))
            for mode in ("npu", "custom"):
                if f"{mode}-a" in logits and f"{mode}-b" in logits:
                    assert np.array_equal(logits[f"{mode}-a"], logits[f"{mode}-b"]), f"repeat logits differ {prompt}/{nt}/{mode}"
            case = {"prompt": prompt, "tokens": nt, "runs": rows, "comparisons": []}
            for cohort in ("a", "b"):
                native, custom = rows.get(f"npu-{cohort}"), rows.get(f"custom-{cohort}")
                if native and custom and not native.get("failed") and not custom.get("failed"):
                    ratio = native["median_ms"]/custom["median_ms"]
                    case["comparisons"].append({"cohort": cohort, "throughput_change_percent": (ratio-1)*100,
                                                "latency_reduction_percent": (1-1/ratio)*100})
            cases.append(case)
    assert len(binary_hashes) <= 1, "mixed executable/library cohort"
    result = {"scope": "authored prompt screen; frozen placement transfer, no timed migration; four samples/process; no general quality guarantee",
              "cases": cases}
    (args.task_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for case in cases:
        compact = {key: round(row["median_ms"], 2) if "median_ms" in row else "FAILED" for key, row in case["runs"].items()}
        print(case["prompt"], case["tokens"], compact, case["comparisons"])


if __name__ == "__main__":
    main()
