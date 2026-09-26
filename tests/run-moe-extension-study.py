#!/usr/bin/env python3
"""Append technical calibration; evaluate frozen placement on two new prompts."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-model-extension"
PREVIOUS = ROOT / "task/handoff/handoff-2026-09-26-context-model"
BASE = ROOT / "task/handoff/handoff-2026-09-26-placement-model/model.json"


def command(*args):
    print("RUN", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def run(mode, nt, tag, *extra):
    path, manifest = (TASK / "evidence" / f"{tag}{suffix}" for suffix in (".json", ".run.json"))
    if path.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
        print("REUSE", tag, flush=True)
        return
    command(sys.executable, "tests/run-fullmodel-oracle.py", "--mode", mode, "--tokens", nt,
            "--tag", tag, "--output-dir", TASK / "evidence", *extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "prepare", "confirm"))
    a = p.parse_args()
    common = ["--shared-activations", "--direct-callback-io", "--activation-arena", "--fused-host-reduce"]
    if a.stage == "calibrate":
        run("capture", 1024, "technical-1024-capture", "--reps", 1, "--prompt-file", TASK / "prompts/technical.txt")
        for family in ("cpu", "gpu", "branch"):
            command(sys.executable, "tests/moe-context-model.py", "calibration-plan", "--capture",
                    TASK / "evidence/technical-1024-capture.json", "--base-model", BASE, "--family", family,
                    "--shared-backend", 2 if family in ("cpu", "branch") else 1,
                    "--output", TASK / f"technical-1024-{family}-plan.json")
        for family, repeat in (("npu", "a"), ("cpu", "a"), ("gpu", "a"), ("branch", "a"),
                               ("branch", "b"), ("gpu", "b"), ("cpu", "b"), ("npu", "b")):
            extra = ["--reps", 4, "--prompt-file", TASK / "prompts/technical.txt"]
            if family != "npu":
                extra += common + ["--plan-file", TASK / f"technical-1024-{family}-plan.json"]
            run("npu" if family == "npu" else "custom", 1024, f"technical-1024-{family}-{repeat}", *extra)
        command(sys.executable, "tests/moe-context-model.py", "fit", "--task-dir", TASK, "--base-model", BASE,
                "--lengths", 1024, "--extend-model", PREVIOUS / "context-model.json", "--shared-family-policy", "--output", TASK / "context-model.json")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        frozen = {"model_sha256": digest(TASK / "context-model.json"), "previous_sha256": digest(PREVIOUS / "context-model.json"),
                  "prompt_sha256": {p.name: digest(p) for p in sorted((TASK / "prompts").glob("*.txt"))},
                  "scope": "1024 technical node appended before museum/programming capture or timing; lifetime reuse excluded from fitting"}
        (TASK / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
    elif a.stage == "prepare":
        for prompt in ("museum", "programming"):
            for nt in (128, 384, 768):
                stem = f"{prompt}-{nt}"
                run("capture", nt, stem + "-capture", "--reps", 1, "--prompt-file", TASK / f"prompts/{prompt}.txt")
                command(sys.executable, "tests/moe-context-model.py", "plan", "--model", TASK / "context-model.json",
                        "--capture", TASK / "evidence" / f"{stem}-capture.json", "--output", TASK / f"{stem}-host-plan.json")
                run("predict", nt, stem + "-plan", "--reference", stem + "-capture", "--cost-model", TASK / "context-model.json")
                host = json.loads((TASK / f"{stem}-host-plan.json").read_text())
                device = json.loads((TASK / "evidence" / f"{stem}-plan.json").read_text())
                assert len(host["layers"]) == len(device["layers"]) == 24
                for x, y in zip(host["layers"], device["layers"]):
                    assert x["selected"] == y["selected"] and abs(x["prediction"]["score_us"] - y["prediction"]["score_us"]) < 1e-6
                d = host["path_decision"]
                upper = d["custom_upper_us"] + device["path_decision"]["decision_budget_us"] - d["decision_budget_us"]
                assert device["execution_mode"] == ("custom" if upper < d["native_lower_us"] else "npu")
    else:
        for prompt in ("museum", "programming"):
            for nt in (128, 384, 768):
                stem = f"{prompt}-{nt}"
                path = TASK / "evidence" / f"{stem}-plan.json"
                plan = json.loads(path.read_text())
                optimize = nt == 768 or (prompt == "museum" and nt == 128)
                policies = ["npu", "selected"]
                if optimize:
                    if plan["execution_mode"] == "npu":
                        policies += ["forced"]
                    policies += ["lifetime"]
                for policy, repeat in [(f, "a") for f in policies] + [(f, "b") for f in reversed(policies)]:
                    extra = ["--reps", 6, "--prompt-file", TASK / f"prompts/{prompt}.txt"]
                    mode = "npu" if policy == "npu" else "auto" if policy == "selected" else "custom"
                    if policy != "npu":
                        extra += ["--plan-file", path]
                    if policy in ("lifetime", "forced"):
                        extra += common
                    if policy == "lifetime":
                        extra += ["--activation-lifetimes"]
                    run(mode, nt, f"{stem}-{policy}-{repeat}", *extra)


if __name__ == "__main__":
    main()
