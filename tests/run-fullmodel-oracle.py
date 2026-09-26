#!/usr/bin/env python3
"""Run a fixed-prompt resident whole-model experiment on fd8657d6."""
import argparse
import gzip
import hashlib
import json
import pathlib
import shlex
import subprocess
import time
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence"
BUILD = ROOT / "build-android-opencl-hexagon"
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"

def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", *args], capture_output=True, text=True, check=check)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["capture", "search", "cpu", "gpu", "npu", "custom", "branch", "custom-npu"])
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--tag", required=True)
    p.add_argument("--reference", default="-")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--output-dir", type=pathlib.Path, default=OUT)
    p.add_argument("--plan-file", type=pathlib.Path)
    p.add_argument("--prompt-file", type=pathlib.Path)
    p.add_argument("--transfer-placement", action="store_true", help="Use frozen expert ownership on another prompt/length; no route replay")
    p.add_argument("--shared-activations", action="store_true")
    p.add_argument("--direct-callback-io", action="store_true")
    p.add_argument("--alternate-callback-io", action="store_true")
    p.add_argument("--activation-arena", action="store_true")
    p.add_argument("--fused-host-reduce", action="store_true")
    p.add_argument("--alternate-host-reduce", action="store_true")
    args = p.parse_args()
    assert args.tag.replace("-", "").replace("_", "").isalnum()
    assert adb("get-state").stdout.strip() == "device"
    assert not adb("shell", "pidof fullmodel-oracle", check=False).stdout.strip()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assert not any((output / (args.tag + suffix)).exists() for suffix in (".json", ".json.gz", ".run.json"))
    adb("shell", f"mkdir -p {REMOTE}")
    if args.deploy:
        names = ["fullmodel-oracle", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert adb("shell", f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    prompt_path = args.prompt_file or (ROOT / "tests/expert-routing-prompt.txt" if args.deploy else None)
    if prompt_path:
        adb("push", str(prompt_path), f"{REMOTE}/prompt.txt")
        assert adb("shell", f"sha256sum {REMOTE}/prompt.txt").stdout.split()[0] == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if args.plan_file:
        opener = gzip.open if args.plan_file.suffix == ".gz" else open
        with opener(args.plan_file, "rb") as source:
            plan = source.read()
        with tempfile.NamedTemporaryFile() as temporary:
            temporary.write(plan)
            temporary.flush()
            adb("push", temporary.name, f"{REMOTE}/offline-plan.json")
        assert adb("shell", f"sha256sum {REMOTE}/offline-plan.json").stdout.split()[0] == hashlib.sha256(plan).hexdigest()
        args.reference = "offline-plan"
    record = {"device": adb("shell", "getprop ro.product.model").stdout,
              "model": adb("shell", f"stat -c '%n %s %Y' {MODEL}").stdout,
              "before_thermal": adb("shell", "dumpsys thermalservice").stdout,
              "hashes": adb("shell", f"sha256sum {REMOTE}/*.so {REMOTE}/fullmodel-oracle {REMOTE}/prompt.txt").stdout,
              "memory_before": adb("shell", "cat /proc/meminfo").stdout}
    record["prompt_source"] = str(prompt_path.resolve()) if prompt_path else "existing remote prompt"
    if args.reference != "-":
        assert args.reference.replace("-", "").replace("_", "").isalnum()
        record["reference_hash"] = adb("shell", f"sha256sum {REMOTE}/{args.reference}.json").stdout
    if args.plan_file:
        record["plan_source"] = str(args.plan_file.resolve())
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0"]
    if args.transfer_placement:
        assert args.mode in ("custom", "branch", "custom-npu")
        env.append("MOE_TRANSFER_PLACEMENT=1")
    if args.shared_activations:
        assert args.mode in ("custom", "branch", "custom-npu")
        env.append("MOE_SHARED_ACTIVATIONS=1")
    if args.direct_callback_io:
        assert args.mode in ("custom", "branch", "custom-npu")
        env.append("MOE_DIRECT_CALLBACK_IO=1")
    if args.alternate_callback_io:
        assert args.shared_activations and not args.direct_callback_io and args.reps % 4 == 0
        env.append("MOE_ALTERNATE_CALLBACK_IO=1")
    if args.activation_arena:
        assert args.shared_activations
        env.append("MOE_ACTIVATION_ARENA=1")
    if args.fused_host_reduce:
        assert args.mode in ("custom", "branch", "custom-npu")
        env.append("MOE_FUSED_HOST_REDUCE=1")
    if args.alternate_host_reduce:
        assert not args.fused_host_reduce and not args.alternate_callback_io and args.reps % 4 == 0
        env.append("MOE_ALTERNATE_HOST_REDUCE=1")
    reference = "-" if args.reference == "-" else f"{REMOTE}/{args.reference}.json"
    argv = [f"{REMOTE}/fullmodel-oracle", MODEL, f"{REMOTE}/prompt.txt", str(args.tokens), f"{REMOTE}/{args.tag}.json", args.mode, str(args.reps), reference]
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
    record["command"] = command
    manifest = output / f"{args.tag}.run.json"
    manifest.write_text(json.dumps(record, indent=2))
    with (output / f"{args.tag}.log").open("w") as log:
        proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
        start = time.monotonic()
        while proc.poll() is None:
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                print(f"{args.tag}: {time.monotonic()-start:.0f}s", flush=True)
    record["exit"] = proc.returncode
    record["after_thermal"] = adb("shell", "dumpsys thermalservice").stdout
    manifest.write_text(json.dumps(record, indent=2))
    assert proc.returncode == 0, f"failed; inspect {args.tag}.log"
    filenames = [f"{args.tag}.json"]
    if args.mode == "capture":
        filenames += [f"{args.tag}.json.layer{i}.bin" for i in range(24)]
    if args.mode != "search":
        filenames += [f"{args.tag}.json.logits.bin"]
    for name in filenames:
        adb("pull", f"{REMOTE}/{name}", str(output / name))
        assert adb("shell", f"sha256sum {REMOTE}/{name}").stdout.split()[0] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    report = json.loads((output / f"{args.tag}.json").read_text())
    print(report.get("timing", {"layers": len(report.get("layers", []))}), flush=True)

if __name__ == "__main__":
    main()
