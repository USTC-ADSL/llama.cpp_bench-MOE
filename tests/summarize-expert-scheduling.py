#!/usr/bin/env python3
"""Pool paired scheduling trials and plot measured migration savings."""
import gzip
import json
import math
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-13-expert-scheduling/evidence"
TOKENS = (1, 8, 16, 32, 64, 128)
DEVICES = ("CPU", "GPUOpenCL", "HTP0")


def load(path):
    if path.exists():
        return json.loads(path.read_text())
    with gzip.open(str(path) + ".gz", "rt") as source:
        return json.load(source)


def stats(samples):
    ordered = sorted(samples)
    assert samples and all(math.isfinite(v) for v in samples)
    return {"count": len(samples), "median_us": statistics.median(samples),
            "p95_us": ordered[math.ceil(0.95 * len(samples)) - 1], "mean_us": statistics.mean(samples)}


def key(case):
    return (case["source"], case["target"], case["kind"])


def path_evidence():
    path = OUT / "profile01.log"
    if path.exists():
        lines = path.read_text().splitlines()
    else:
        with gzip.open(str(path) + ".gz", "rt") as source:
            lines = source.read().splitlines()
    paths = {}
    current = None
    for line in lines:
        match = re.search(r"SCHEDULE_CASE tokens=(\d+)", line)
        if match:
            current = int(match[1])
            paths[current] = {"kernels": set(), "events": set(), "examples": []}
        if current is None:
            continue
        data = paths[current]
        if "profile-op MUL_MAT_ID" in line:
            kernel = re.search(r"\|(hmx-tiled|hvx-tiled|hvx-flat) vtcm", line)
            assert kernel, line
            if kernel[1] not in data["kernels"]:
                data["examples"].append(line)
            data["kernels"].add(kernel[1])
        if "trace-evt" in line:
            event = re.search(r"trace-evt (HMX_COMP|HVX_COMP):", line)
            if event:
                if event[1] not in data["events"]:
                    data["examples"].append(line)
                data["events"].add(event[1])
    assert set(paths) == set(TOKENS)
    for tokens, data in paths.items():
        assert data["kernels"] == {"hvx-tiled" if tokens == 1 else "hmx-tiled"}, (tokens, data)
        assert ("HVX_COMP" if tokens == 1 else "HMX_COMP") in data["events"]
        data["kernels"] = sorted(data["kernels"]); data["events"] = sorted(data["events"])
    return paths


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    routes = [("CPU", "HTP0", "gguf_reread"), ("CPU", "GPUOpenCL", "gguf_reread"),
              ("GPUOpenCL", "HTP0", "packed_only")]
    names = {"CPU": "CPU_REPACK", "GPUOpenCL": "OpenCL", "HTP0": "HTP (HMX preferred)"}
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.0), layout="constrained")
    for col, (source, target, alternative) in enumerate(routes):
        for row, reuse in enumerate((1, 8)):
            ax = axes[row, col]
            for kind, color, marker in (("retained_raw", "#0072B2", "o"), (alternative, "#D55E00", "s")):
                selected = [next(c for c in summary["cases"] if c["tokens"] == t and c["reuse"] == reuse
                                 and c["source"] == source and c["target"] == target and c["kind"] == kind) for t in TOKENS]
                values = [c["savings"]["median_us"] / reuse / 1000 for c in selected]
                lower = [max(0, v-min(c["run_savings_median_us"])/reuse/1000) for v, c in zip(values, selected)]
                upper = [max(0, max(c["run_savings_median_us"])/reuse/1000-v) for v, c in zip(values, selected)]
                label = {"retained_raw": "Raw Q4_0 retained", "gguf_reread": "GGUF reread (warm)",
                         "packed_only": "Restore source packed weights"}[kind]
                ax.errorbar(TOKENS, values, yerr=[lower, upper], color=color, marker=marker,
                            capsize=3, linewidth=1.5, markersize=4, label=label)
            ax.axhline(0, color="#333333", linewidth=1)
            ax.set_xscale("log", base=2); ax.set_xticks(TOKENS, [str(t) for t in TOKENS])
            ax.grid(True, alpha=0.22); ax.set_axisbelow(True)
            ax.set_title(f"{names[source]} to {names[target]}\nR = {reuse} invocation(s)", fontsize=10)
            ax.set_xlabel("Tokens assigned to this expert")
            if col == 0:
                ax.set_ylabel("Time saved per invocation (ms)")
            ax.legend(fontsize=8, loc="best")
    fig.suptitle("Measured migration benefit: positive values favor moving the expert", fontsize=13)
    fig.supxlabel("Paired medians; bars span the three run medians. Preallocated 8-slot banks; idle backends; no overlap.", fontsize=9)
    fig.savefig(OUT / "migration-benefit.png", dpi=180)
    fig.savefig(OUT / "migration-benefit.pdf")
    plt.close(fig)


def main():
    runs = [load(OUT / f"run{i:02d}.json") for i in (1, 2, 3)]
    records = [load(OUT / f"run{i:02d}.run.json") for i in (1, 2, 3)]
    identities = {r["hashes"] for r in records}
    models = {r["model_sha256"].split()[0] for r in records}
    assert len(identities) == len(models) == 1
    for run, record in zip(runs, records):
        assert record["exit"] == 0 and record["serial"] == "fd8657d6"
        assert run["reps"] == 20 and run["warmup_pairs"] == 2 and run["slots"] == 8
        assert run["expert_id"] == 7 and run["cache_slot"] == 0 and run["cpu_persistent_threadpool"]
        assert not run["diagnostic"] and {c["tokens"] for c in run["cases"]} == set(TOKENS)
    profile_record = load(OUT / "profile01.run.json")
    assert profile_record["exit"] == 0 and profile_record["hashes"] in identities
    assert profile_record["model_sha256"].split()[0] in models
    summary = {"runs": ["run01", "run02", "run03"], "samples_per_comparison": 60,
               "probe_and_library_hashes": identities.pop(), "model_sha256": models.pop(),
               "paths": path_evidence(), "cases": [], "target_ready": [],
               "decision_rule": "move/stay requires same sign of paired median in all 3 runs and >=80% samples agreeing; otherwise mixed",
               "scope": "direct paired totals, including weight migration and input/output; allocations and queueing excluded",
               "reuse_caveat": "R is known and the target copy remains available; no online prediction or model speedup is measured"}
    for tokens in TOKENS:
        token_cases = [next(c for c in run["cases"] if c["tokens"] == tokens) for run in runs]
        for case in token_cases:
            for validation in [*case["initial_validation"].values(), *case["final_validation"].values()]:
                assert validation["worst_token_nrmse"] < 0.04
        for reuse in ([1, 8, 32] if tokens <= 8 else [1, 8]):
            reuse_cases = [next(c for c in case["reuse_cases"] if c["reuse"] == reuse) for case in token_cases]
            for device in DEVICES:
                samples = [v for c in reuse_cases for v in c["target_ready"][device]["samples_us"]]
                summary["target_ready"].append({"tokens": tokens, "reuse": reuse, "device": device, **stats(samples)})
            keys = {key(c) for c in reuse_cases[0]["comparisons"]}
            assert len(keys) == 12 and all({key(c) for c in rc["comparisons"]} == keys for rc in reuse_cases)
            for source, target, kind in sorted(keys):
                comparisons = [next(c for c in rc["comparisons"] if key(c) == (source, target, kind)) for rc in reuse_cases]
                stay = []; move = []; savings = []; per_run = []
                for c in comparisons:
                    assert c["zero_target_restore_validated"] and c["validation"]["worst_token_nrmse"] < 0.04
                    s = c["stay"]["samples_us"]; m = c["move"]["samples_us"]
                    assert len(s) == len(m) == len(c["savings_us"]) == 20
                    assert all(abs(a-b-d) < 1e-6 for a, b, d in zip(s, m, c["savings_us"]))
                    assert all(a > 0 for a in s+m)
                    phases = [c[name] for name in ("restore_or_read_us", "prepare_us", "target_input_us",
                                                   "target_compute_us", "target_output_us")]
                    assert all(sum(p[r] for p in phases) <= m[r]+1e-6 for r in range(20)), "phase overlap or timing inconsistency"
                    stay.extend(s); move.extend(m); savings.extend(c["savings_us"])
                    per_run.append(statistics.median(c["savings_us"]))
                win_fraction = sum(s > 0 for s in savings)/len(savings)
                decision = "mixed"
                if min(per_run) > 0 and win_fraction >= 0.8:
                    decision = "move"
                elif max(per_run) < 0 and win_fraction <= 0.2:
                    decision = "stay"
                entry = {"tokens": tokens, "reuse": reuse, "source": source, "target": target, "kind": kind,
                         "stay": stats(stay), "move": stats(move), "savings": stats(savings),
                         "run_savings_median_us": per_run, "move_win_fraction": win_fraction,
                         "decision": decision, "ratio_of_medians": statistics.median(stay)/statistics.median(move)}
                for phase in ("restore_or_read_us", "prepare_us", "target_input_us", "target_compute_us", "target_output_us"):
                    values = [v for c in comparisons for v in c[phase]]
                    assert len(values) == 60 and all(v >= 0 for v in values)
                    entry[phase] = stats(values)
                summary["cases"].append(entry)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(summary)
    print("All values ms; stay/move are directly measured medians, delta is paired savings median.")
    for c in summary["cases"]:
        if c["reuse"] == 1 or (c["source"] == "GPUOpenCL" and c["target"] == "HTP0"):
            print(f"T={c['tokens']:3} R={c['reuse']:2} {c['source']:9}->{c['target']:9} {c['kind']:12} "
                  f"stay={c['stay']['median_us']/1000:8.3f} move={c['move']['median_us']/1000:8.3f} "
                  f"delta={c['savings']['median_us']/1000:8.3f} win={c['move_win_fraction']:.2f} {c['decision']}")


if __name__ == "__main__":
    main()
