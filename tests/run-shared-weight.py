#!/usr/bin/env python3
"""Sequential Android operator measurements, with deployment identity and raw samples."""
import argparse
import hashlib
import json
import pathlib
import shlex
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-android-opencl-hexagon"
OUT = ROOT / "task/handoff/handoff-2026-10-16-shared-weight-layout/evidence"
REMOTE = "/data/local/tmp/llama-shared-weight-layout"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--backend", choices=["CPU", "GPUOpenCL", "HTP0"], required=True)
    p.add_argument("--reps", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--cpu-mask", default="f0", help="CPU compute affinity on K80 Pro; helpers do no weight conversion")
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--shared", action="store_true")
    p.add_argument("--edges", action="store_true", help="small shapes and tail batches; correctness only")
    p.add_argument("--profile", action="store_true", help="enable DSP counters; GPU counters require profiling build")
    p.add_argument("--model", action="store_true", help="read expert 7 weights from the designated device GGUF")
    a = p.parse_args()
    if not a.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.tag):
        p.error("invalid tag")
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / a.tag
    if prefix.with_suffix(".run.json").exists():
        p.error("tag already exists")

    def adb(*args, check=True):
        return subprocess.run(["adb", "-s", "fd8657d6", *args], check=check, capture_output=True, text=True)

    def shell(cmd, check=True):
        return adb("shell", cmd, check=check)

    devices = subprocess.run(["adb", "devices", "-l"], check=True, capture_output=True, text=True).stdout
    print(devices, flush=True)
    if adb("get-state").stdout.strip() != "device":
        raise RuntimeError("default device unavailable")
    if shell("pidof shared-weight-probe", check=False).stdout.strip():
        raise RuntimeError("probe already running")
    if a.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            if shell(f"test -f {REMOTE}/.shared-weight-owned", check=False).returncode:
                raise RuntimeError("refusing unowned remote directory")
        else:
            shell(f"mkdir {REMOTE} && touch {REMOTE}/.shared-weight-owned")
        names = ["shared-weight-probe", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths += [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so"]
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            if shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError("deployment hash mismatch")
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off"]
    if a.shared:
        env.append("GGML_SHARED_WEIGHT_TEST=1")
    if a.edges:
        env.append("GGML_SHARED_WEIGHT_EDGES=1")
    if a.profile:
        env += ["GGML_HEXAGON_PROFILE=1", "GGML_HEXAGON_TIMING=1"]
    if a.model:
        env.append("GGML_SHARED_WEIGHT_MODEL=/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf")
    if a.backend == "CPU":
        int(a.cpu_mask, 16)
        env.append(f"GGML_SHARED_TEST_CPUMASK={a.cpu_mask}")
    command = f"cd {REMOTE} && exec " + shlex.join(env + [f"{REMOTE}/shared-weight-probe", a.backend,
                  f"{REMOTE}/{a.tag}.json", str(a.reps), str(a.threads)])
    record = {"devices": devices, "command": command,
              "git": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
              "hashes": shell(f"sha256sum {REMOTE}/shared-weight-probe {REMOTE}/*.so").stdout,
              "thermal_before": shell("dumpsys thermalservice").stdout}
    start = time.monotonic()
    with prefix.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"{a.tag}: running {time.monotonic()-start:.0f}s", flush=True)
    record.update(exit=process.returncode, elapsed_seconds=time.monotonic()-start,
                  thermal_after=shell("dumpsys thermalservice").stdout)
    prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2)+"\n")
    if record["exit"]:
        raise RuntimeError(f"probe failed; inspect {prefix}.log")
    adb("pull", f"{REMOTE}/{a.tag}.json", str(prefix.with_suffix(".json")))
    shell(f"rm {REMOTE}/{a.tag}.json")
    for name in ["cl_profiling.csv", "cl_trace.json"]:
        if shell(f"test -f {REMOTE}/{name}", check=False).returncode == 0:
            adb("pull", f"{REMOTE}/{name}", str(OUT / f"{a.tag}-{name}"))
            shell(f"rm {REMOTE}/{name}")
    print(f"saved {prefix}.json", flush=True)


if __name__ == "__main__":
    main()
