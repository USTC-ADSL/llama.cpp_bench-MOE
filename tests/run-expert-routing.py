#!/usr/bin/env python3
"""Resident full-layer CPU/NPU routing comparison on the default Android device."""
import argparse
import gzip
import hashlib
import json
import pathlib
import shlex
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-android-opencl-hexagon"
OUT = ROOT / "task/handoff/handoff-2026-09-22-threshold-routing/evidence"
REMOTE = "/data/local/tmp/llama-expert-routing"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    thresholds = parser.add_mutually_exclusive_group()
    thresholds.add_argument("--threshold", type=int, default=16, help="minimum token count assigned to NPU")
    thresholds.add_argument("--cpu-max-tokens", type=int, help="inclusive maximum token count assigned to CPU")
    parser.add_argument("--parallel-only", action="store_true", help="time only all-NPU and mixed parallel; CPU reference remains untimed")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if args.cpu_max_tokens is not None:
        args.threshold = args.cpu_max_tokens + 1
    assert args.tag.replace("-", "").replace("_", "").isalnum()
    assert args.reps > 0 and args.warmup > 0 and 1 <= args.threshold <= 256
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = OUT / args.tag
    assert not any(OUT.glob(args.tag + ".*")), "use a fresh tag"

    def adb(*argv, check=True):
        return subprocess.run(["adb", "-s", "fd8657d6", *argv], check=check, capture_output=True, text=True)

    def shell(command, check=True):
        return adb("shell", command, check=check)

    print(subprocess.run(["adb", "devices", "-l"], check=True, capture_output=True, text=True).stdout, flush=True)
    assert adb("get-state").stdout.strip() == "device"
    assert not shell("pidof expert-routing-probe expert-routing-reference", check=False).stdout.strip(), "probe already running"
    if args.deploy:
        if shell(f"test -d {REMOTE}", check=False).returncode == 0:
            assert shell(f"test -f {REMOTE}/expert-routing-probe", check=False).returncode == 0, "unowned directory"
        else:
            shell(f"mkdir {REMOTE}")
        names = ["expert-routing-probe", "expert-routing-reference", "libllama.so", "libggml.so", "libggml-base.so",
                 "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
        if not args.reference:
            archived = OUT / "reference.json.gz"
            with gzip.open(archived, "rb") as stream, tempfile.TemporaryDirectory() as directory:
                reference = pathlib.Path(directory) / "reference.json"
                reference.write_bytes(stream.read())
                adb("push", str(reference), f"{REMOTE}/reference.json")
                assert shell(f"sha256sum {REMOTE}/reference.json").stdout.split()[0] == hashlib.sha256(reference.read_bytes()).hexdigest()
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1",
           "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", f"GGML_HEXAGON_PROFILE={3 if args.profile else 0}"]
    record = {"hashes": shell(f"sha256sum {REMOTE}/*.so {REMOTE}/expert-routing-*").stdout,
              "model": shell(f"stat -c '%n %s %Y' {MODEL}").stdout,
              "model_sha256": shell(f"sha256sum {MODEL}").stdout,
              "profile": args.profile,
              "before_thermal": shell("dumpsys thermalservice").stdout, "commands": []}

    def run(name, argv):
        command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
        record["commands"].append(command)
        with (OUT / f"{name}.log").open("w") as log:
            proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f"{name}: running {time.monotonic()-started:.0f}s", flush=True)
        record["last_exit"] = proc.returncode
        prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
        assert proc.returncode == 0, f"failed: {name}.log"

    if args.reference:
        prompt = ROOT / "tests/expert-routing-prompt.txt"
        adb("push", str(prompt), f"{REMOTE}/prompt.txt")
        run(args.tag + "-reference", [f"{REMOTE}/expert-routing-reference", MODEL, f"{REMOTE}/reference.json", f"{REMOTE}/prompt.txt"])
        adb("pull", f"{REMOTE}/reference.json", str(OUT / "reference.json"))
    record["reference_sha256"] = shell(f"sha256sum {REMOTE}/reference.json").stdout
    command = [f"{REMOTE}/expert-routing-probe", MODEL, f"{REMOTE}/{args.tag}.json", str(args.reps),
               str(args.warmup), f"{REMOTE}/reference.json", str(args.threshold)]
    if args.parallel_only:
        command.append("parallel-only")
    run(args.tag, command)
    adb("pull", f"{REMOTE}/{args.tag}.json", str(prefix.with_suffix(".json")))
    shell(f"rm {REMOTE}/{args.tag}.json")
    record["after_thermal"] = shell("dumpsys thermalservice").stdout
    prefix.with_suffix(".run.json").write_text(json.dumps(record, indent=2) + "\n")
    result = json.loads(prefix.with_suffix(".json").read_text())
    for case in result["cases"]:
        print("layers", case["layers"], {p: round(v["median_us"]/1000, 3) for p, v in case["summary"].items()}, flush=True)


if __name__ == "__main__":
    main()
