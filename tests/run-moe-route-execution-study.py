#!/usr/bin/env python3
"""Controlled unit-cost audit and opt-in NPU graph gather experiments."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-28-route-execution"
OLD = ROOT / "task/handoff/handoff-2026-09-28-route-scheduler"
EV = TASK / "evidence"
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
BUILD = ROOT / "build-android-opencl-hexagon"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
FLAGS = ["--shared-activations", "--direct-callback-io", "--activation-arena", "--activation-lifetimes", "--fused-host-reduce", "--overlap-gather"]


def adb(*args):
    return subprocess.check_output(["adb", "-s", "fd8657d6", *map(str, args)], text=True)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def deploy():
    assert adb("get-state").strip() == "device"
    assert not adb("shell", "pidof fullmodel-oracle moe-route-unit-probe route-resident-probe || true").strip()
    EV.mkdir(parents=True, exist_ok=True)
    adb("shell", f"mkdir -p {REMOTE}")
    paths = [BUILD / "bin" / name for name in ("fullmodel-oracle", "moe-route-unit-probe", "route-resident-probe", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so")]
    paths += [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so", BUILD / "ggml/src/ggml-hexagon/ship/libc++_shared.so"]
    capture = OLD / "evidence/dev-128-capture.json"
    paths += [capture, pathlib.Path(str(capture)+".layer0.bin")]
    hashes = {}
    for p in paths:
        adb("push", p, f"{REMOTE}/{p.name}")
        hashes[p.name] = sha(p)
        assert adb("shell", f"sha256sum {REMOTE}/{p.name}").split()[0] == hashes[p.name]
    (TASK / "deployment.json").write_text(json.dumps(hashes, indent=2)+"\n")


def probe(kind):
    EV.mkdir(parents=True, exist_ok=True)
    result = EV / f"{kind}.json"
    assert not result.exists()
    binary = "moe-route-unit-probe" if kind.startswith("unit") else "route-resident-probe"
    capture = REMOTE + "/dev-128-capture.json" + (".layer0.bin" if kind.startswith("unit") else "")
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0", "GGML_HEXAGON_TIMING=1", "MOE_NPU_DEVICE_GATHER=1"]
    extra = ["CPU"] if kind=="unit-cpu" else ["GPUOpenCL"] if kind=="unit-gpu" else []
    cmd = shlex.join(env + [REMOTE+"/"+binary, MODEL, capture, REMOTE+f"/{kind}.json"] + extra)
    with (EV / f"{kind}.log").open("w") as log:
        status = subprocess.run(["adb", "-s", "fd8657d6", "shell", cmd], stdout=log, stderr=subprocess.STDOUT).returncode
    (EV / f"{kind}.run.json").write_text(json.dumps({"command": cmd, "exit": status, "deployment": json.loads((TASK/"deployment.json").read_text())}, indent=2)+"\n")
    assert status == 0, f"{kind} failed; inspect log"
    adb("pull", REMOTE+f"/{kind}.json", result)
    print(kind, "complete", flush=True)


def full(nt):
    name = "technical-extended" if nt == 1536 else "museum"
    route = OLD / ("dev-1536-resident-v2.json" if nt == 1536 else f"plans/confirm-museum-{nt}-route.json")
    for repeat, order in zip("ab", [("native", "base", "gather"), ("gather", "base", "native")]):
        for policy in order:
            tag = f"execution-{nt}-{policy}-{repeat}"
            path = EV / f"{tag}.json"
            if path.exists():
                assert json.loads(path.with_suffix(".run.json").read_text())["exit"] == 0
                continue
            extra = [] if policy == "native" else FLAGS + ["--plan-file", str(route), "--route-model", str(OLD/"model-frozen.json")]
            if policy == "gather": extra += ["--npu-device-gather"]
            subprocess.run([sys.executable, str(ROOT/"tests/run-fullmodel-oracle.py"), "--mode", "npu" if policy == "native" else "custom",
                            "--tokens", str(nt), "--tag", tag, "--reps", "6", "--hexagon-timing",
                            "--prompt-file", str(OLD/"prompts"/f"{name}.txt"), "--output-dir", str(EV), *extra], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["deploy", "unit", "unit-cpu", "unit-gpu", "resident", "full"])
    p.add_argument("--tokens", type=int, choices=[128,768,1024,1536,2048], default=768)
    args = p.parse_args()
    deploy() if args.command == "deploy" else full(args.tokens) if args.command == "full" else probe(args.command)
