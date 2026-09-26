#!/usr/bin/env python3
"""Fixed-owner, within-process ABBA experiment for host token workers."""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-host-parallel"
PRIOR = TASK.parent / "handoff-2026-09-26-model-extension"


def run(prompt, nt, threads, tag, deploy=False, technical=False, mode="custom", quality=False):
    output = TASK / "evidence"
    path, manifest = output / f"{tag}.json", output / f"{tag}.run.json"
    if path.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
        print("REUSE", tag, flush=True)
        return json.loads(path.read_text())
    args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", mode,
            "--tokens", str(nt), "--tag", tag, "--reps", "1" if quality else "12" if mode == "custom" else "6",
            "--output-dir", str(output), "--prompt-file", str(PRIOR / f"prompts/{prompt}.txt")]
    if deploy:
        args += ["--deploy"]
    if mode == "custom":
        plan = PRIOR / ("technical-1024-cpu-plan.json.gz" if technical else f"evidence/{prompt}-{nt}-plan.json.gz")
        args += ["--plan-file", str(plan), "--shared-activations", "--direct-callback-io", "--activation-arena",
                 "--fused-host-reduce", "--activation-lifetimes"]
        if technical:
            args += ["--transfer-placement"]
        if quality:
            args += ["--quality-steps", "16", "--quality-tokens", str(TASK / f"{prompt}-{nt}-teacher-tokens.json")]
        else:
            args += ["--host-reduce-threads", str(threads), "--alternate-reduce-threads"]
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("tune", "confirm"))
    a = p.parse_args()
    if a.stage == "tune":
        rows = []
        for threads in (2, 4):
            d = run("technical", 768, threads, f"tune-technical-768-t{threads}", deploy=threads == 2, technical=True)
            groups = [[s["total_us"] for s in d["samples"] if s["host_reduce_threads"] == n] for n in (1, threads)]
            rows.append({"threads": threads, "serial_mean_us": statistics.mean(groups[0]),
                         "parallel_mean_us": statistics.mean(groups[1]), "gain": statistics.mean(groups[0]) / statistics.mean(groups[1]) - 1})
        selected = max(rows, key=lambda r: r["gain"])["threads"]
        freeze = {"threads": selected, "scope": "worker count selected using technical768 only; placement model/ownership unchanged", "tuning": rows}
        (TASK / "host-worker-choice.json").write_text(json.dumps(freeze, indent=2) + "\n")
        print(freeze, flush=True)
    else:
        threads = json.loads((TASK / "host-worker-choice.json").read_text())["threads"]
        for prompt, nt in (("museum", 128), ("museum", 384), ("museum", 768), ("programming", 768)):
            for repeat in ("a", "b"):
                # Bracket independent custom processes with native runs at768.
                if nt == 768 and repeat == "a":
                    run(prompt, nt, 1, f"parallel-{prompt}-{nt}-npu-a", mode="npu")
                run(prompt, nt, threads, f"parallel-{prompt}-{nt}-{repeat}")
                if nt == 768 and repeat == "b":
                    run(prompt, nt, 1, f"parallel-{prompt}-{nt}-npu-b", mode="npu")
        run("museum", 128, 1, "quality-lifetime-museum-128", quality=True)


if __name__ == "__main__":
    main()
