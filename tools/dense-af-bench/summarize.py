#!/usr/bin/env python3
"""Aggregate two complete 36-layer Android runs; emit raw samples and summary CSVs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path)
    ap.add_argument("--prompts", nargs="+", type=int, default=[64, 256])
    args = ap.parse_args()
    groups = defaultdict(list)
    raw = []
    for rnd in (1, 2):
        files = sorted(args.results.glob(f"r{rnd}-*.json"))
        expected_runs = len(args.prompts) * 6
        if len(files) != expected_runs:
            raise SystemExit(f"round {rnd}: expected {expected_runs} runs, found {len(files)}")
        for path in files:
            run = json.loads(path.read_text())
            if run["returncode"] != 0:
                raise SystemExit(f"failed run: {path}")
            expected_blocks = ["attention", "ffn"] if path.stem.endswith("-both") else ["attention"]
            rows = run["results"]
            if len(rows) != 36 * len(expected_blocks):
                raise SystemExit(f"incomplete run: {path}")
            for block in expected_blocks:
                if sorted(r["layer"] for r in rows if r["block"] == block) != list(range(36)):
                    raise SystemExit(f"duplicate/missing layers: {path}, {block}")
            for row in rows:
                key = (row["prompt"], row["block"], row["flash"], row["device"])
                groups[key].append(dict(row, round=rnd))
                for sample, ms in enumerate(row["compute_ms"]):
                    raw.append(dict(round=rnd, prompt=key[0], block=key[1], flash=key[2], device=key[3],
                                    layer=row["layer"], sample=sample, compute_ms=ms,
                                    upload_ms=row["upload_ms"][sample], download_ms=row["download_ms"][sample]))
    summary, layer_rows = [], []
    times = {}
    for key, rows in sorted(groups.items()):
        layers = defaultdict(list)
        for r in rows:
            layers[r["layer"]].extend(r["compute_ms"])
        means = [statistics.mean(statistics.median(r["compute_ms"]) for r in rows if r["round"] == rnd) for rnd in (1, 2)]
        layer_medians = {layer: statistics.median(values) for layer, values in layers.items()}
        mean = statistics.mean(layer_medians.values())
        times[key] = mean
        summary.append(dict(prompt=key[0], block=key[1], flash=key[2], device=key[3],
                            mean_layer_ms=mean, sum_36_layers_ms=sum(layer_medians.values()),
                            round1_mean_layer_ms=means[0], round2_mean_layer_ms=means[1],
                            min_layer_ms=min(layer_medians.values()), max_layer_ms=max(layer_medians.values()),
                            mean_upload_ms=statistics.mean(statistics.median(r["upload_ms"]) for r in rows),
                            mean_download_ms=statistics.mean(statistics.median(r["download_ms"]) for r in rows),
                            max_relative_rms=max(r["relative_rms"] for r in rows)))
        for layer, ms in sorted(layer_medians.items()):
            layer_rows.append(dict(prompt=key[0], block=key[1], flash=key[2], device=key[3], layer=layer, median_ms=ms))
    comparisons = []
    for prompt in args.prompts:
        for block, flash in (("attention", 0), ("attention", 1), ("ffn", 0)):
            cpu, gpu, npu = [times[prompt, block, flash, dev] for dev in ("CPU", "GPUOpenCL", "HTP0")]
            npu_layers = {r["layer"]: r["median_ms"] for r in layer_rows
                          if (r["prompt"], r["block"], r["flash"], r["device"]) == (prompt, block, flash, "HTP0")}
            win_counts = {}
            for dev in ("CPU", "GPUOpenCL"):
                peers = [r for r in layer_rows if (r["prompt"], r["block"], r["flash"], r["device"]) == (prompt, block, flash, dev)]
                win_counts[dev] = sum(npu_layers[r["layer"]] < r["median_ms"] for r in peers)
            comparisons.append(dict(prompt=prompt, block=block, flash=flash, cpu_ms=cpu, gpu_ms=gpu, npu_ms=npu,
                                    npu_speedup_vs_cpu=cpu/npu, npu_speedup_vs_gpu=gpu/npu,
                                    npu_latency_reduction_vs_cpu_pct=100*(1-npu/cpu),
                                    npu_latency_reduction_vs_gpu_pct=100*(1-npu/gpu),
                                    npu_wins_vs_cpu_layers=win_counts["CPU"], npu_wins_vs_gpu_layers=win_counts["GPUOpenCL"]))
    write_csv(args.results / "samples.csv", raw)
    write_csv(args.results / "layers.csv", layer_rows)
    write_csv(args.results / "summary.csv", summary)
    write_csv(args.results / "comparison.csv", comparisons)
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
