#!/usr/bin/env python3
"""Bounded resident decode screening; not prefill placement calibration."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-27-decode-parallel"
EV = TASK / "evidence"
PROMPTS = ROOT / "task/handoff/handoff-2026-09-26-model-extension/prompts"
TEACHERS = ROOT / "task/handoff/handoff-2026-09-26-host-parallel"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def run(case, nt, policy, repeat, deploy=False, persistent=False):
    tag = f"{case}-{nt}-{policy}-{repeat}"
    path = EV / f"{tag}.json"
    if path.exists():
        assert json.loads((EV / f"{tag}.run.json").read_text())["exit"] == 0
        print("REUSE", tag, flush=True)
        return
    mode = policy if policy in ("cpu", "gpu") else "custom"
    command = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"),
               "--mode", mode, "--tokens", str(nt), "--tag", tag, "--reps", "1",
               "--quality-steps", "16", "--decode-reps", "4", "--output-dir", str(EV),
               "--prompt-file", str(PROMPTS / f"{case}.txt"),
               "--quality-tokens", str(TEACHERS / f"{case}-{nt}-teacher-tokens.json")]
    if deploy:
        command.append("--deploy")
    if mode == "custom":
        shared = {"custom-cpu": 0, "shared-gpu": 2, "shared-npu": 1, "routed-gpu": 0, "routed-npu": 0}[policy]
        owner = {"routed-gpu": 2, "routed-npu": 1}.get(policy, 0)
        plan = TASK / f"{policy}-plan.json"
        if not plan.exists():
            write(plan, {"tokens": nt, "layers": [
                {"layer": il, "tokens": 1 if il == 23 else nt,
                 "selected": {"name": policy, "owner": [owner]*60, "shared": shared}}
                for il in range(24)]})
        command += ["--plan-file", str(plan), "--transfer-placement", "--dense-backend", "cpu",
                    "--cpu-mapped-experts", "--shared-activations", "--direct-callback-io",
                    "--activation-arena", "--activation-lifetimes", "--fused-host-reduce"]
        if persistent:
            command.append("--native-decode-graph")
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pilot", "graph", "reverse", "confirm"])
    args = parser.parse_args()
    if args.stage == "pilot":
        protocol = TASK / "protocol.json"
        if not protocol.exists():
            files = ["tests/fullmodel-oracle.cpp", "tests/expert-routing-probe.cpp", "tests/run-fullmodel-oracle.py", "tests/run-moe-decode-study.py"]
            write(protocol, {"scope": "resident full-model single-request decode; no migration speedup claim",
                             "pilot": "museum128 CPU, GPU, customCPU, sharedGPU, sharedNPU, CPU bracket",
                             "sequence": "16 fixed CPU teacher tokens, two sequence warmups, four measured repeats",
                             "confirmation": "freeze candidate and policy before programming768; native brackets and reverse order; negative candidates retained",
                             "memory": "CPU routed references original immutable file pages; full shared replicas are a screening upper bound (~446MiB), not a 200MB design",
                             "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}})
        for i, (policy, repeat) in enumerate([(x, "v3") for x in ["cpu", "custom-cpu", "shared-gpu", "shared-npu"]] + [("cpu", "v3b")]):
            run("museum", 128, policy, repeat, deploy=i == 0)
    elif args.stage == "graph":
        for i, (policy, repeat) in enumerate([(x, "v4") for x in ["cpu", "custom-cpu", "shared-gpu", "shared-npu"]] + [("cpu", "v4b")]):
            run("museum", 128, policy, repeat, deploy=i == 0, persistent=True)
    elif args.stage == "reverse":
        for policy in ("routed-gpu", "routed-npu", "cpu"):
            run("museum", 128, policy, "reverse")
    else:
        freeze=TASK/"confirmation-freeze.json"
        declaration={"selected_candidate":"shared-gpu with native CPU graph", "selection":"fastest screened heterogeneous path on museum128, still slower than nativeCPU; confirmation may remain negative",
                     "cases":[["museum",128],["programming",768]], "steps":16,"sequence_repetitions":4,
                     "order":"museum CPU/custom/CPU; programming CPU/GPU/custom/custom/GPU/CPU",
                     "binary_sha256":hashlib.sha256((ROOT/"build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes()).hexdigest(),
                     "stop":"No further fitting or candidate changes based on confirmation. Retain CPU if no reliable heterogeneous advantage."}
        if freeze.exists():
            assert json.loads(freeze.read_text())==declaration
        else:
            write(freeze,declaration)
        for policy,repeat in (("cpu","confirm-a"),("shared-gpu","confirm-a"),("cpu","confirm-b")):
            run("museum",128,policy,repeat,persistent=True)
        for policy,repeat in (("cpu","v2"),("gpu","confirm-a"),("shared-gpu","confirm-a"),
                              ("shared-gpu","confirm-b"),("gpu","confirm-b"),("cpu","confirm-b")):
            run("programming",768,policy,repeat,persistent=True)


if __name__ == "__main__":
    main()
