#!/usr/bin/env python3
"""Calibrate fullmodel context on technical; validate frozen model on new ecology prompt."""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-context-model"
PREVIOUS = ROOT / "task/handoff/handoff-2026-09-26-placement-model"
TECHNICAL = ROOT / "task/handoff/handoff-2026-09-26-route-generalization/prompts/technical.txt"


def command(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def run(mode, nt, tag, *extra):
    path = TASK / "evidence" / f"{tag}.json"
    manifest = TASK / "evidence" / f"{tag}.run.json"
    if path.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
        print("REUSE", tag, flush=True)
        return
    command(sys.executable, "tests/run-fullmodel-oracle.py", "--mode", mode, "--tokens", nt, "--tag", tag,
            "--output-dir", TASK / "evidence", *extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "prepare-test", "confirm"))
    args = p.parse_args()
    TASK.mkdir(parents=True, exist_ok=True)
    if args.stage == "calibrate":
        for nt in (128, 256, 512):
            for f in ("cpu", "gpu", "branch"):
                command(sys.executable, "tests/moe-context-model.py", "calibration-plan",
                        "--capture", PREVIOUS / "evidence" / f"technical-{nt}-capture.json.gz",
                        "--base-model", PREVIOUS / "model.json", "--family", f,
                        "--output", TASK / f"technical-{nt}-{f}-plan.json")
            for f, repeat in (("npu", "a"), ("cpu", "a"), ("gpu", "a"), ("branch", "a"),
                              ("branch", "b"), ("gpu", "b"), ("cpu", "b"), ("npu", "b")):
                extra = ["--reps", 4, "--prompt-file", TECHNICAL]
                if f != "npu":
                    extra += ["--shared-activations", "--direct-callback-io", "--activation-arena",
                              "--fused-host-reduce", "--plan-file", TASK / f"technical-{nt}-{f}-plan.json"]
                if nt == 128 and f == "npu" and repeat == "a":
                    extra += ["--deploy"]
                run("npu" if f == "npu" else "custom", nt, f"technical-{nt}-{f}-{repeat}", *extra)
        command(sys.executable, "tests/moe-context-model.py", "fit", "--task-dir", TASK,
                "--base-model", PREVIOUS / "model.json", "--output", TASK / "context-model.json")
    elif args.stage == "prepare-test":
        for nt in (128, 256, 512):
            run("capture", nt, f"ecology-{nt}-capture", "--reps", 1, "--prompt-file", TASK / "prompts/ecology.txt")
            command(sys.executable, "tests/moe-context-model.py", "plan", "--model", TASK / "context-model.json",
                    "--capture", TASK / "evidence" / f"ecology-{nt}-capture.json", "--output", TASK / f"ecology-{nt}-host-plan.json")
            run("predict", nt, f"ecology-{nt}-plan", "--reference", f"ecology-{nt}-capture", "--cost-model", TASK / "context-model.json")
            host = json.loads((TASK / f"ecology-{nt}-host-plan.json").read_text())
            device = json.loads((TASK / "evidence" / f"ecology-{nt}-plan.json").read_text())
            decision = host["path_decision"]
            adjusted_upper = decision["custom_upper_us"] + device["path_decision"]["decision_budget_us"] - decision["decision_budget_us"]
            assert device["execution_mode"] == ("custom" if adjusted_upper < decision["native_lower_us"] else "npu")
            for a, b in zip(host["layers"], device["layers"]):
                assert a["selected"] == b["selected"]
                assert abs(a["prediction"]["score_us"] - b["prediction"]["score_us"]) < 1e-6
            command(sys.executable, "tests/moe-placement-model.py", "plan", "--model", PREVIOUS / "model.json",
                    "--capture", TASK / "evidence" / f"ecology-{nt}-capture.json", "--output", TASK / f"ecology-{nt}-v1-plan.json")
    else:
        for nt in (128, 256, 512):
            for f, repeat in (("npu", "a"), ("selected", "a"), ("v1", "a"), ("custom", "a"),
                              ("custom", "b"), ("v1", "b"), ("selected", "b"), ("npu", "b")):
                extra = ["--reps", 6, "--prompt-file", TASK / "prompts/ecology.txt"]
                mode = "npu"
                if f != "npu":
                    mode = "auto" if f == "selected" else "custom"
                    path = TASK / f"ecology-{nt}-v1-plan.json" if f == "v1" else TASK / "evidence" / f"ecology-{nt}-plan.json"
                    extra += ["--plan-file", path]
                    if f != "selected":
                        extra += ["--shared-activations", "--direct-callback-io", "--activation-arena", "--fused-host-reduce"]
                    if f == "custom":
                        extra += ["--feedback-model", TASK / "context-model.json"]
                run(mode, nt, f"ecology-{nt}-{f}-{repeat}", *extra)
        plans = [json.loads((TASK / "evidence" / f"ecology-{nt}-plan.json").read_text()) for nt in (128, 256, 512)]
        if all(p["execution_mode"] != "npu" for p in plans):
            # Functional loader check only; never present this as a model choice.
            forced = plans[0]
            forced["execution_mode"] = "npu"
            forced["path_decision"] = {"execution_mode": "npu", "scope": "forced native loader contract test"}
            forced["contract_test_forced_mode"] = True
            path = TASK / "native-contract-plan.json"
            path.write_text(json.dumps(forced, indent=2) + "\n")
            run("auto", 128, "native-loader-contract", "--reps", 2, "--prompt-file", TASK / "prompts/ecology.txt", "--plan-file", path)


if __name__ == "__main__":
    main()
