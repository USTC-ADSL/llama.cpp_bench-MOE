#!/usr/bin/env python3
"""Sequential, resumable resident-model study on fd8657d6 (no device switching)."""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-placement-model"
PROMPTS = ROOT / "task/handoff/handoff-2026-09-26-route-generalization/prompts"


def command(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def run(mode, tokens, tag, *extra):
    path = TASK / "evidence" / (tag + ".json")
    manifest = TASK / "evidence" / (tag + ".run.json")
    if path.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
        print("REUSE", tag, flush=True)
        return
    command(sys.executable, "tests/run-fullmodel-oracle.py", "--mode", mode, "--tokens", tokens,
            "--tag", tag, "--output-dir", TASK / "evidence", *extra)


def main():
    global TASK
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["calibrate", "validate", "fullmodel"])
    p.add_argument("--task-dir", type=pathlib.Path, default=TASK, help="Use a fresh directory for a new cohort; resume only before archiving")
    p.add_argument("--deploy", action="store_true", help="Deploy binaries with the first calibration capture")
    args = p.parse_args()
    TASK = args.task_dir.resolve()
    if args.deploy and args.stage != "calibrate":
        p.error("--deploy applies to the initial calibrate stage")
    ev = TASK / "evidence"
    if args.stage == "calibrate":
        for n in (128, 256, 512):
            run("capture", n, f"technical-{n}-capture", "--reps", 1, "--prompt-file", PROMPTS / "technical.txt",
                *(["--deploy"] if args.deploy and n == 128 else []))
            run("profile", n, f"technical-{n}-profile-v2", "--reference", f"technical-{n}-capture",
                "--profile-layers", "0,3,11,19,23", "--reps", 3)
        command(sys.executable, "tests/moe-placement-model.py", "train",
                *(ev / f"technical-{n}-profile-v2.json" for n in (128, 256, 512)), "--output", TASK / "model.json")
    elif args.stage == "validate":
        assert (TASK / "model.json").exists()
        for n in (128, 512):
            run("capture", n, f"narrative-{n}-capture", "--reps", 1, "--prompt-file", PROMPTS / "narrative.txt")
            # Both implementations decide BEFORE any held-out candidate timing.
            command(sys.executable, "tests/moe-placement-model.py", "plan", "--model", TASK / "model.json",
                    "--capture", ev / f"narrative-{n}-capture.json", "--output", TASK / f"plan-{n}-host.json")
            run("predict", n, f"narrative-{n}-predicted", "--reference", f"narrative-{n}-capture", "--cost-model", TASK / "model.json")
            host = json.loads((TASK / f"plan-{n}-host.json").read_text())
            device = json.loads((ev / f"narrative-{n}-predicted.json").read_text())
            for a, b in zip(host["layers"], device["layers"]):
                assert a["selected"] == b["selected"]
                assert abs(a["prediction"]["total_us"] - b["prediction"]["total_us"]) < 1e-6
            run("profile", n, f"narrative-{n}-profile", "--reference", f"narrative-{n}-capture",
                "--profile-layers", "1,7,15,21", "--reps", 3)
        command(sys.executable, "tests/moe-placement-model.py", "evaluate", "--model", TASK / "model.json",
                *(ev / f"narrative-{n}-profile.json" for n in (128, 512)), "--output", TASK / "heldout.json")
    else:
        for n in (128, 512):
            for policy, repeat in (("npu", "a"), ("predicted", "a"), ("frozen", "a"),
                                   ("frozen", "b"), ("predicted", "b"), ("npu", "b")):
                options = ["--reps", 6, "--prompt-file", PROMPTS / "narrative.txt"]
                if policy != "npu":
                    options += ["--shared-activations", "--direct-callback-io", "--activation-arena", "--fused-host-reduce"]
                    if policy == "predicted":
                        options += ["--plan-file", ev / f"narrative-{n}-predicted.json"]
                    else:
                        options += ["--transfer-placement", "--plan-file", ROOT / "task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence" / f"search{128 if n == 128 else 256}.json.gz"]
                run("npu" if policy == "npu" else "custom", n, f"narrative-{n}-{policy}-{repeat}", *options)


if __name__ == "__main__":
    main()
