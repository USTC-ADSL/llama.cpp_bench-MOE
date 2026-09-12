#!/usr/bin/env python3
"""Measure real GGUF expert preparation on Android; retain samples and deployment identity."""
import argparse
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-android-opencl-hexagon"
OUT = ROOT / "task/handoff/handoff-2026-09-12-weight-prepare/evidence"
REMOTE = "/data/local/tmp/llama-weight-prepare"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="fd8657d6")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag), "invalid tag"
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / args.tag
    assert not prefix.with_suffix(".json").exists(), "use a fresh tag"

    def adb(*argv, check=True):
        return subprocess.run(["adb", "-s", args.serial, *argv], check=check, text=True, capture_output=True)

    def shell(command, check=True):
        return adb("shell", command, check=check)

    print(subprocess.run(["adb", "devices", "-l"], check=True, text=True, capture_output=True).stdout, flush=True)
    assert adb("get-state").stdout.strip() == "device", "selected device unavailable"
    assert not shell("pidof weight-prepare-probe", check=False).stdout.strip(), "probe already running"
    if args.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            assert shell(f"test -f {REMOTE}/weight-prepare-probe", check=False).returncode == 0, "unowned directory"
        else:
            shell(f"mkdir {REMOTE}")
        names = ["weight-prepare-probe", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = shell(f"sha256sum {REMOTE}/weight-prepare-probe {REMOTE}/*.so").stdout
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off"]
    if args.profile:
        env.append("GGML_OPENCL_PROFILE_EXPERT_PREPARE=1")
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + [f"{REMOTE}/weight-prepare-probe", MODEL,
        f"{REMOTE}/{args.tag}.json", str(args.reps), str(args.slots)])
    record = {"serial": args.serial, "command": command, "hashes": hashes,
              "model_stat": shell(f"stat -c '%n %s %Y' {MODEL}").stdout,
              "before_battery": shell("dumpsys battery").stdout,
              "before_thermal": shell("dumpsys thermalservice").stdout}
    start = time.monotonic()
    with prefix.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(["adb", "-s", args.serial, "shell", command], stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"{args.tag}: running {time.monotonic() - start:.0f}s", flush=True)
        record["exit"] = process.returncode
    record["elapsed_seconds"] = time.monotonic() - start
    record["after_battery"] = shell("dumpsys battery").stdout
    record["after_thermal"] = shell("dumpsys thermalservice").stdout
    prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
    assert record["exit"] == 0, f"probe failed; see {prefix}.log"
    adb("pull", f"{REMOTE}/{args.tag}.json", str(prefix.with_suffix(".json")))
    result = json.loads(prefix.with_suffix(".json").read_text())
    assert len(result["matrices"]) == 3 and len(result["expert_triple_migrations"]) == 4
    for name, data in result["expert_triple"].items():
        print(f"{name}: triple prepare median={data['prepare']['median_us']:.3f} us; "
              f"read-to-ready={data['warm_read_to_ready']['median_us']:.3f} us", flush=True)


if __name__ == "__main__":
    main()
