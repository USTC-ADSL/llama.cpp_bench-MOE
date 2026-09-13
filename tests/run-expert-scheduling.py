#!/usr/bin/env python3
"""Directly compare expert residency versus migration plus computation on Android."""
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
OUT = ROOT / "task/handoff/handoff-2026-09-13-expert-scheduling/evidence"
REMOTE = "/data/local/tmp/llama-expert-scheduling"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="fd8657d6")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--tokens", default="1,8,16,32,64,128")
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag), "invalid tag"
    assert all(1 <= int(t) <= 128 for t in args.tokens.split(","))
    assert 1 <= args.reps <= 1000 and 0 <= args.warmup <= 100 and 2 <= args.slots <= 60
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / args.tag
    assert not prefix.with_suffix(".log").exists(), "use a fresh tag"

    def adb(*argv, check=True):
        return subprocess.run(["adb", "-s", args.serial, *argv], check=check, text=True, capture_output=True)

    def shell(command, check=True):
        return adb("shell", command, check=check)

    print(subprocess.run(["adb", "devices", "-l"], check=True, text=True, capture_output=True).stdout, flush=True)
    assert adb("get-state").stdout.strip() == "device", "selected device unavailable"
    assert not shell("pidof expert-scheduling-probe", check=False).stdout.strip(), "probe already running"
    if args.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            assert shell(f"test -f {REMOTE}/expert-scheduling-probe", check=False).returncode == 0, "unowned directory"
        else:
            shell(f"mkdir {REMOTE}")
        names = ["expert-scheduling-probe", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                 "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = shell(f"sha256sum {REMOTE}/expert-scheduling-probe {REMOTE}/*.so").stdout
    print("Verifying model SHA256 before measurement", flush=True)
    model_hash = shell(f"sha256sum {MODEL}").stdout
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off", "GGML_HEXAGON_NHMX=1",
           "GGML_HEXAGON_MM_SELECT=3", f"GGML_HEXAGON_PROFILE={3 if args.profile else 0}",
           f"GGML_HEXAGON_VERBOSE={1 if args.profile else 0}"]
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + [f"{REMOTE}/expert-scheduling-probe", MODEL,
        f"{REMOTE}/{args.tag}.json", str(args.reps), str(args.warmup), str(args.slots), args.tokens])
    record = {"serial": args.serial, "command": command, "hashes": hashes, "model_sha256": model_hash,
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
    assert len(result["cases"]) == len(args.tokens.split(","))
    for case in result["cases"]:
        for reuse_case in case["reuse_cases"]:
            wins = [f"{c['source']}->{c['target']}:{c['kind']}" for c in reuse_case["comparisons"]
                    if c["move_win_fraction"] >= 0.8]
            print(f"{args.tag}: T={case['tokens']} R={reuse_case['reuse']} wins={', '.join(wins) or 'none'}", flush=True)
    shell(f"rm {REMOTE}/{args.tag}.json")


if __name__ == "__main__":
    main()
