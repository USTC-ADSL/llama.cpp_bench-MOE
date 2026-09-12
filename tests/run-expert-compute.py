#!/usr/bin/env python3
"""Run resident expert FFN computation measurements on the selected Android device."""
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
OUT = ROOT / "task/handoff/handoff-2026-09-12-expert-compute/evidence"
REMOTE = "/data/local/tmp/llama-expert-compute"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", default="fd8657d6")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--modes", default="cpu,opencl,hmx,hvx")
    parser.add_argument("--tokens", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag), "invalid tag"
    modes = args.modes.split(",")
    devices = {"cpu": "CPU", "opencl": "GPUOpenCL", "hmx": "HTP0", "hvx": "HTP0"}
    assert modes and len(set(modes)) == len(modes) and all(m in devices for m in modes)
    assert all(1 <= int(t) <= 512 for t in args.tokens.split(","))
    assert 1 <= args.reps <= 10000 and 0 <= args.warmup <= 1000
    OUT.mkdir(parents=True, exist_ok=True)

    def adb(*argv, check=True):
        return subprocess.run(["adb", "-s", args.serial, *argv], check=check, text=True, capture_output=True)

    def shell(command, check=True):
        return adb("shell", command, check=check)

    print(subprocess.run(["adb", "devices", "-l"], check=True, text=True, capture_output=True).stdout, flush=True)
    assert adb("get-state").stdout.strip() == "device", "selected device unavailable"
    assert not shell("pidof expert-compute-probe", check=False).stdout.strip(), "probe already running"
    if args.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            assert shell(f"test -f {REMOTE}/expert-compute-probe", check=False).returncode == 0, "unowned directory"
        else:
            shell(f"mkdir {REMOTE}")
        names = ["expert-compute-probe", "libggml.so", "libggml-base.so", "libggml-cpu.so",
                 "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = shell(f"sha256sum {REMOTE}/expert-compute-probe {REMOTE}/*.so").stdout
    model_stat = shell(f"stat -c '%n %s %Y' {MODEL}").stdout
    print("Verifying model SHA256 before measurement", flush=True)
    model_hash = shell(f"sha256sum {MODEL}").stdout
    for mode in modes:
        tag = f"{args.tag}-{mode}"
        prefix = OUT / tag
        assert not prefix.with_suffix(".log").exists(), "use a fresh tag"
        env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
               "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off",
               f"GGML_HEXAGON_NHMX={0 if mode == 'hvx' else 1}",
               f"GGML_HEXAGON_MM_SELECT={2 if mode == 'hvx' else 3}",
               f"GGML_HEXAGON_PROFILE={3 if args.profile else 0}",
               f"GGML_HEXAGON_VERBOSE={1 if args.profile else 0}"]
        command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + [f"{REMOTE}/expert-compute-probe", MODEL,
            f"{REMOTE}/{tag}.json", devices[mode], mode, str(args.reps), str(args.warmup), args.tokens])
        record = {"serial": args.serial, "command": command, "hashes": hashes, "model_stat": model_stat,
                  "model_sha256": model_hash,
                  "before_battery": shell("dumpsys battery").stdout,
                  "before_thermal": shell("dumpsys thermalservice").stdout}
        start = time.monotonic()
        with prefix.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(["adb", "-s", args.serial, "shell", command], stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f"{tag}: running {time.monotonic() - start:.0f}s", flush=True)
            record["exit"] = process.returncode
        record["elapsed_seconds"] = time.monotonic() - start
        record["after_battery"] = shell("dumpsys battery").stdout
        record["after_thermal"] = shell("dumpsys thermalservice").stdout
        prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
        assert record["exit"] == 0, f"probe failed; see {prefix}.log"
        adb("pull", f"{REMOTE}/{tag}.json", str(prefix.with_suffix(".json")))
        result = json.loads(prefix.with_suffix(".json").read_text())
        assert len(result["cases"]) == len(args.tokens.split(","))
        for case in result["cases"]:
            if case["status"] == "ok":
                print(f"{tag}: T={case['tokens']:3d} full={case['timings']['full']['median_us']:.3f} us", flush=True)
            else:
                print(f"{tag}: T={case['tokens']:3d} {case['status']}: {case['node']}", flush=True)
        shell(f"rm {REMOTE}/{tag}.json")


if __name__ == "__main__":
    main()
