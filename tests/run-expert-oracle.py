#!/usr/bin/env python3
"""Run the resident offline placement study on fd8657d6."""
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
OUT = ROOT / "task/handoff/handoff-2026-09-23-oracle-routing/evidence"
BUILD = ROOT / "build-android-opencl-hexagon"
REMOTE = "/data/local/tmp/llama-expert-oracle"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", *args], capture_output=True, text=True, check=check)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--plans", type=pathlib.Path, help="reuse frozen search winners")
    args = parser.parse_args()
    assert args.tag.replace("-", "").replace("_", "").isalnum()
    assert adb("get-state").stdout.strip() == "device"
    assert not adb("shell", "pidof expert-oracle-probe expert-routing-reference", check=False).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / (args.tag + ".json")).exists()
    adb("shell", f"mkdir -p {REMOTE}")
    if args.deploy:
        names = ["expert-oracle-probe", "expert-routing-reference", "libllama.so", "libggml.so", "libggml-base.so",
                 "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert adb("shell", f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
        adb("push", str(ROOT / "tests/expert-routing-prompt.txt"), f"{REMOTE}/prompt.txt")
    record = {"device": adb("shell", "getprop ro.product.model").stdout,
              "before_thermal": adb("shell", "dumpsys thermalservice").stdout,
              "hashes": adb("shell", f"sha256sum {REMOTE}/*.so {REMOTE}/expert-*").stdout,
              "model": adb("shell", f"stat -c '%n %s %Y' {MODEL}").stdout, "commands": []}
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1",
           "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0"]

    def run(label, argv):
        command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
        record["commands"].append(command)
        with (OUT / f"{args.tag}-{label}.log").open("w") as log:
            proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f"{label}: {time.monotonic()-start:.0f}s", flush=True)
        record["last_exit"] = proc.returncode
        (OUT / f"{args.tag}.run.json").write_text(json.dumps(record, indent=2))
        assert proc.returncode == 0, f"{label} failed; inspect log"

    name = f"{'decode' if args.decode else 'prefill'}-{args.tokens}-reference.json"
    if adb("shell", f"test -f {REMOTE}/{name}", check=False).returncode:
        command = [f"{REMOTE}/expert-routing-reference", MODEL, f"{REMOTE}/{name}", f"{REMOTE}/prompt.txt", str(args.tokens)]
        if args.decode:
            command.append("decode")
        run("reference", command)
    adb("pull", f"{REMOTE}/{name}", str(OUT / name))
    command = [f"{REMOTE}/expert-oracle-probe", MODEL, f"{REMOTE}/{name}", f"{REMOTE}/{args.tag}.json", str(args.reps)]
    if args.plans:
        if args.plans.suffix == ".gz":
            with tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "frozen.json"
                with gzip.open(args.plans, "rb") as stream:
                    path.write_bytes(stream.read())
                adb("push", str(path), f"{REMOTE}/frozen.json")
        else:
            adb("push", str(args.plans), f"{REMOTE}/frozen.json")
        command.append(f"{REMOTE}/frozen.json")
        record["frozen_sha256"] = adb("shell", f"sha256sum {REMOTE}/frozen.json").stdout.split()[0]
    run("search", command)
    adb("pull", f"{REMOTE}/{args.tag}.json", str(OUT / f"{args.tag}.json"))
    adb("shell", f"rm {REMOTE}/{args.tag}.json")
    record["after_thermal"] = adb("shell", "dumpsys thermalservice").stdout
    (OUT / f"{args.tag}.run.json").write_text(json.dumps(record, indent=2))
    data = json.loads((OUT / f"{args.tag}.json").read_text())
    for case in data["confirmation"]:
        print(case["layers"], {k: round(v["median_us"]/1000, 3) for k, v in case["summary"].items()}, flush=True)


if __name__ == "__main__":
    main()
