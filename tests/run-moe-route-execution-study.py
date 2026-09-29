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
    if kind.endswith("-fused"): env += ["GGML_HEXAGON_FUSE_MOE_GATHER=1"]
    cmd = shlex.join(env + [REMOTE+"/"+binary, MODEL, capture, REMOTE+f"/{kind}.json"] + extra)
    with (EV / f"{kind}.log").open("w") as log:
        status = subprocess.run(["adb", "-s", "fd8657d6", "shell", cmd], stdout=log, stderr=subprocess.STDOUT).returncode
    (EV / f"{kind}.run.json").write_text(json.dumps({"command": cmd, "exit": status, "deployment": json.loads((TASK/"deployment.json").read_text())}, indent=2)+"\n")
    assert status == 0, f"{kind} failed; inspect log"
    adb("pull", REMOTE+f"/{kind}.json", result)
    print(kind, "complete", flush=True)


def full(nt, fused=False):
    name = "technical-extended" if nt == 1536 else "museum"
    route = OLD / ("dev-1536-resident-v2.json" if nt == 1536 else f"plans/confirm-museum-{nt}-route.json")
    for repeat, order in zip("ab", [("native", "base", "gather"), ("gather", "base", "native")]):
        for policy in order:
            tag = f"{'fusion' if fused else 'execution'}-{nt}-{policy}-{repeat}"
            path = EV / f"{tag}.json"
            if path.exists():
                assert json.loads(path.with_suffix(".run.json").read_text())["exit"] == 0
                continue
            extra = [] if policy == "native" else FLAGS + ["--plan-file", str(route), "--route-model", str(OLD/"model-frozen.json")]
            if policy == "gather": extra += ["--npu-device-gather"]
            if policy == "gather" and fused: extra += ["--npu-fused-gather"]
            subprocess.run([sys.executable, str(ROOT/"tests/run-fullmodel-oracle.py"), "--mode", "npu" if policy == "native" else "custom",
                            "--tokens", str(nt), "--tag", tag, "--reps", "6", "--hexagon-timing",
                            "--prompt-file", str(OLD/"prompts"/f"{name}.txt"), "--output-dir", str(EV), *extra], check=True)


def regression(kind):
    EV.mkdir(parents=True, exist_ok=True)
    assert not adb("shell", "pidof fullmodel-oracle moe-route-unit-probe route-resident-probe htp-indexed-gather-probe || true").strip()
    path = BUILD / "bin/htp-indexed-gather-probe"
    adb("push", path, REMOTE+"/"+path.name)
    assert adb("shell", f"sha256sum {REMOTE}/{path.name}").split()[0] == sha(path)
    result = EV / f"{kind}.json"
    assert not result.exists()
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_VERBOSE=1",
           "GGML_HEXAGON_MM_SELECT="+("1" if kind.endswith("hvx") else "2"), "GGML_HEXAGON_OPFUSION="+("0" if kind.endswith("no-nx") else "1")]
    cmd = shlex.join(env+[REMOTE+"/"+path.name,MODEL,REMOTE+f"/{kind}.json"])
    with (EV/f"{kind}.log").open("w") as log:
        status = subprocess.run(["adb","-s","fd8657d6","shell",cmd],stdout=log,stderr=subprocess.STDOUT).returncode
    (EV/f"{kind}.run.json").write_text(json.dumps({"command":cmd,"exit":status,"probe_sha256":sha(path),"deployment":json.loads((TASK/"deployment.json").read_text())},indent=2)+"\n")
    assert status==0, f"{kind} failed"
    indexed = [line for line in (EV/f"{kind}.log").read_text().splitlines() if "execute-op MUL_MAT_ID+INDEXED_ROWS" in line]
    assert bool(indexed) == (not kind.endswith("hvx")), "unexpected fusion eligibility"
    assert all("-> fused-projection" in line for line in indexed), "unsafe case fused"
    adb("pull",REMOTE+f"/{kind}.json",result)
    print(kind,"complete",flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["deploy", "unit", "unit-cpu", "unit-gpu", "unit-fused", "resident", "resident-fused", "full", "full-fused", "regression", "regression-no-nx", "regression-hvx"])
    p.add_argument("--tokens", type=int, choices=[128,768,1024,1536,2048], default=768)
    p.add_argument("--cohort", default="", help="Separate immutable evidence directory for a new build")
    args = p.parse_args()
    assert not args.cohort or args.cohort.replace("-","").isalnum()
    if args.cohort: EV = EV / args.cohort
    if args.command=="deploy": deploy()
    elif args.command.startswith("full"): full(args.tokens,args.command=="full-fused")
    elif args.command.startswith("regression"): regression(args.command)
    else: probe(args.command)
