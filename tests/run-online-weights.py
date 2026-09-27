#!/usr/bin/env python3
"""Run bounded-weight prototypes on fd8657d6, preserving failed evidence too."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-27-online-weights"
BUILD = ROOT / "build-android-opencl-hexagon"
REMOTE = "/data/local/tmp/llama-online-weights"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--mode", choices=["probe","native","online"], default="probe")
    p.add_argument("--prompt", type=pathlib.Path)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--tail", type=int, default=-1)
    p.add_argument("--fused-gpu", action="store_true")
    p.add_argument("--lazy-scratch", action="store_true")
    p.add_argument("--profile-prepare", action="store_true")
    p.add_argument("--memory", action="store_true")
    p.add_argument("--packed-head", action="store_true")
    p.add_argument("--kernel-cache-off", action="store_true")
    p.add_argument("--reference-packed", action="store_true")
    p.add_argument("--bounded-source", action="store_true")
    p.add_argument("--resident-shared-source", action="store_true")
    p.add_argument("--resident-output", action="store_true")
    p.add_argument("--cost-model", type=pathlib.Path, default=ROOT / "task/handoff/handoff-2026-09-27-length-intervals/interval-model.json")
    args = p.parse_args()
    assert re.fullmatch(r"[A-Za-z0-9_-]+", args.tag)
    assert 4 <= args.reps <= 1000
    prefix = OUT / args.tag
    program = "online-weight-probe" if args.mode == "probe" else "online-moe"
    assert not list(OUT.glob(args.tag + ".*")), "use a fresh evidence tag"
    def adb(*parts, check=True):
        return subprocess.run(["adb", "-s", "fd8657d6", *parts], check=check, capture_output=True, text=True)
    def shell(command, check=True):
        return adb("shell", command, check=check)
    print(subprocess.run(["adb", "devices", "-l"], check=True, capture_output=True, text=True).stdout, flush=True)
    assert adb("get-state").stdout.strip() == "device"
    assert not shell("pidof online-weight-probe online-moe dynamic-moe fullmodel-oracle", check=False).stdout.strip(), "inference already active"
    if args.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            assert shell(f"test -f {REMOTE}/.online-weights-owner", check=False).returncode == 0, "unowned directory"
        else:
            shell(f"mkdir {REMOTE}")
            shell(f"touch {REMOTE}/.online-weights-owner")
        paths = [BUILD / "bin" / name for name in [program, "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    arguments = [f"./{program}", MODEL, f"{REMOTE}/{args.tag}.json", str(args.reps)]
    inputs = {}
    if args.mode != "probe":
        assert args.prompt and args.prompt.is_file(), "provide a prompt"
        for path, name in [(args.prompt,"prompt.txt"),(args.cost_model,"cost-model.json")]:
            destination = f"{REMOTE}/{args.tag}-{name}"
            adb("push", str(path), destination)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert shell(f"sha256sum {destination}").stdout.split()[0] == digest
            inputs[name] = {"source": str(path), "sha256": digest}
        arguments = [f"./{program}", MODEL, f"{REMOTE}/{args.tag}-prompt.txt", str(args.tokens),
            f"{REMOTE}/{args.tag}-cost-model.json", f"{REMOTE}/{args.tag}.json", args.mode, str(args.tail)]
    environment = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
        "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0"]
    if args.fused_gpu:
        environment.append("GGML_OPENCL_FUSED_Q4_PREPARE=1")
    if args.lazy_scratch:
        environment.append("GGML_OPENCL_LAZY_SCRATCH=1")
    if args.profile_prepare:
        environment.append("GGML_OPENCL_PROFILE_EXPERT_PREPARE=1")
    if args.memory:
        environment += ["GGML_ACCEL_BUDGET_MODE=observe", "GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra", "GGML_ACCEL_BUDGET_RPCMEM_COVERAGE=charged"]
    if args.packed_head:
        environment.append("MOE_ONLINE_PACKED_HEAD=1")
    if args.kernel_cache_off:
        environment.append("GGML_OPENCL_KERNEL_CACHE_DIR=off")
    if args.reference_packed:
        environment.append("MOE_ONLINE_REFERENCE_PACKED=1")
    if args.bounded_source:
        environment.append("MOE_BOUNDED_SOURCE=1")
    if args.resident_shared_source:
        assert args.bounded_source, "resident canonical shared source requires bounded source mappings"
        environment.append("MOE_RAW_SHARED_SOURCE=1")
    if args.resident_output:
        environment.append("MOE_RESIDENT_OUTPUT=1")
    command = f"cd {REMOTE} && exec " + shlex.join(environment + arguments)
    record = {"serial": "fd8657d6", "command": command, "inputs": inputs,
        "artifacts": shell(f"sha256sum {REMOTE}/{program} {REMOTE}/*.so").stdout,
        "model_stat": shell(f"stat -c '%n %s %Y' {MODEL}").stdout,
        "before_battery": shell("dumpsys battery").stdout}
    record["sources"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [
        ROOT / "tests/online-weight-probe.cpp", ROOT / "tests/online-moe.cpp", ROOT / "tests/online-moe-runtime.h",
        ROOT / "tests/expert-routing-probe.cpp", ROOT / "ggml/src/ggml-hexagon/ggml-hexagon.cpp",
        ROOT / "src/llama-model.cpp", ROOT / "include/llama.h",
        ROOT / "ggml/src/ggml-opencl/ggml-opencl.cpp", ROOT / "ggml/src/ggml-opencl/kernels/cvt.cl",
        ROOT / "ggml/src/ggml-hexagon/htp/htp-ops.h", ROOT / "ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h", ROOT / "ggml/src/ggml-hexagon/htp/matmul-ops.c"]}
    start = time.monotonic()
    with prefix.with_suffix(".log").open("x") as log:
        proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"{args.tag}: running {time.monotonic()-start:.0f}s", flush=True)
    record["exit"] = proc.returncode
    record["elapsed_seconds"] = time.monotonic()-start
    record["after_battery"] = shell("dumpsys battery").stdout
    if shell(f"test -f {REMOTE}/{args.tag}.json", check=False).returncode == 0:
        adb("pull", f"{REMOTE}/{args.tag}.json", str(prefix.with_suffix(".json")))
        expected = shell(f"sha256sum {REMOTE}/{args.tag}.json").stdout.split()[0]
        assert hashlib.sha256(prefix.with_suffix(".json").read_bytes()).hexdigest() == expected
        record["output_sha256"] = expected
    if args.mode != "probe" and shell(f"test -f {REMOTE}/{args.tag}.json.logits.bin", check=False).returncode == 0:
        path = pathlib.Path(str(prefix)+".logits.bin")
        adb("pull", f"{REMOTE}/{args.tag}.json.logits.bin", str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert shell(f"sha256sum {REMOTE}/{args.tag}.json.logits.bin").stdout.split()[0] == digest
        record["logits_sha256"] = digest
    prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2)+"\n")
    assert proc.returncode == 0, f"failed; see {prefix}.log"
    print(f"saved {prefix}.json", flush=True)


if __name__ == "__main__":
    main()
