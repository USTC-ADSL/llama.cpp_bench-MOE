#!/usr/bin/env python3
"""Run synchronized llama-bench residency comparisons on the default Android device."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import shlex
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-android-opencl-hexagon"
OUT = ROOT / "task/handoff/handoff-2026-09-12-weight-perf/evidence"
REMOTE = "/data/local/tmp/llama-weight-perf"
SERIAL = "fd8657d6"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", SERIAL, *args], text=True, capture_output=True, check=check)


def shell(command, check=True):
    return adb("shell", command, check=check)


def deploy(tag):
    if shell(f"test -d {REMOTE}", check=False).returncode == 0:
        assert shell(f"test -f {REMOTE}/llama-bench", check=False).returncode == 0, "unowned directory"
    else:
        shell(f"mkdir {REMOTE}")
    assert not shell("pidof llama-bench", check=False).stdout.strip(), "benchmark already running"
    names = ["llama-bench", "libllama-bench-impl.so", "libllama.so", "libllama-common.so", "libggml.so", "libggml-base.so",
             "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
    paths = [BUILD / "bin" / name for name in names]
    paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
    hashes = {}
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        adb("push", str(path), f"{REMOTE}/{path.name}")
        assert shell(f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == digest
        hashes[path.name] = digest
    (OUT / f"{tag}-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")


def snapshot():
    commands = {
        "memory": "cat /proc/meminfo",
        "battery": "dumpsys battery",
        "thermal": "dumpsys thermalservice",
    }
    return {name: shell(command, check=False).stdout for name, command in commands.items()}


def run(case, args):
    backend, mode = case.split("-", 1)
    devices = {"cpu": "none", "gpu": "GPUOpenCL", "htp": "HTP0", "mixed": "GPUOpenCL/HTP0"}[backend]
    prefix = f"{args.tag}-{case}"
    local = OUT / prefix
    assert not local.with_suffix(".json").exists(), "use a fresh tag"
    argv = [f"{REMOTE}/llama-bench", "-m", MODEL, "-dev", devices,
            "-ngl", "0" if backend == "cpu" else "17" if backend == "mixed" else "99",
            "-p", str(args.prompt), "-n", str(args.gen), "-b", "128", "-ub", str(args.ubatch),
            "-t", "4", "-r", str(args.reps), "-o", "json", "-lm", "mmap"]
    if backend == "mixed":
        argv += ["-ts", "8/9"]
    if mode == "norepack":
        argv += ["--no-repack"]
    if args.no_op_offload:
        argv += ["-nopo", "1"]
    if args.verbose:
        argv += ["-v"]
    if mode not in ("default", "norepack"):
        budgets = {"CPU": {"total_mib": 12288, "reserve_mib": 128, "expert_slots": args.slots}}
        for dev in devices.split("/"):
            if dev != "none":
                budgets[dev] = {"total_mib": 12288, "reserve_mib": 128, "expert_slots": args.slots}
        layers = []
        if backend == "mixed":
            layers = [{"first": 0, "last": 7, "backend": "CPU"},
                      {"first": 8, "last": 15, "backend": "GPUOpenCL"},
                      {"first": 16, "last": 23, "backend": "HTP0"}]
        weights = [] if mode == "fixed" else [{"pattern": r"ffn_(gate|up|down)_exps\.weight$", "residency": "swap"}]
        if args.weight_gib:
            assert mode == "swap", "weight capacity applies to swap cases"
            calibration = OUT / f"{args.calibration_tag}-{backend}-swap"
            registry = json.loads(calibration.with_suffix(".weights.json").read_text())
            measured = registry["devices"]
            prior = json.loads(calibration.with_suffix(".config.json").read_text())["budgets"]
            fixed_by_device = {dev: d["fixed_bytes"] for dev, d in measured.items()}
            per_slot = {dev: d["cache_bytes"] / prior[dev]["expert_slots"] for dev, d in measured.items()}
            target = args.weight_gib * 2**30
            banks = [w for w in registry["weights"] if w["residency"] == "swap"]
            def bank_order(weight):
                layer = int(weight["name"].split(".")[1])
                return (layer % 8, layer // 8, weight["name"]) if backend == "mixed" else (layer, weight["name"])
            banks.sort(key=bank_order)
            pinned = []
            # Whole banks avoid oversized device buffers. Leave at least 32 cache slots per shape.
            for weight in banks:
                if sum(fixed_by_device.values()) + weight["source_bytes"] + 32 * sum(per_slot.values()) > target:
                    break
                pinned.append(weight["name"])
                fixed_by_device[weight["backend"]] += weight["source_bytes"]
            weights.append({"pattern": "^(?:" + "|".join(re.escape(name) for name in pinned) + ")$", "residency": "fixed"})
            slots = math.floor((target - sum(fixed_by_device.values())) / sum(per_slot.values()))
            assert slots >= 32, "insufficient room for this prefill workload"
            for dev, budget in budgets.items():
                budget["expert_slots"] = slots
                # Weight capacity excludes runtime and reserve; allow context growth and source staging.
                budget["total_mib"] = math.ceil((fixed_by_device[dev] + per_slot[dev] * slots) / 2**20) + 256
        config = {"version": 1, "budgets": budgets, "layers": layers, "weights": weights}
        if args.report:
            config["report"] = f"{REMOTE}/{prefix}.weights.json"
        local.with_suffix(".config.json").write_text(json.dumps(config, indent=2) + "\n")
        adb("push", str(local.with_suffix(".config.json")), f"{REMOTE}/{prefix}.config.json")
        argv += ["--weight-config", f"{REMOTE}/{prefix}.config.json"]
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off"]
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
    record = {"case": case, "command": command, "before": snapshot(), "samples": []}
    print(f"START {prefix}", flush=True)
    started = time.monotonic()
    with local.with_suffix(".stdout").open("w") as out, local.with_suffix(".log").open("w") as err:
        process = subprocess.Popen(["adb", "-s", SERIAL, "shell", command], stdout=out, stderr=err)
        try:
            while process.poll() is None:
                time.sleep(2)
                status = shell("p=$(pidof llama-bench); if [ -n \"$p\" ]; then cat /proc/$p/status; fi", check=False).stdout
                record["samples"].append({"seconds": round(time.monotonic() - started, 3),
                    **{k: v.strip() for line in status.splitlines() if ":" in line
                       for k, v in [line.split(":", 1)] if k in ("VmRSS", "VmHWM", "VmSwap", "VmLck")}})
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(prefix)
        finally:
            if process.poll() is None:
                shell("pkill -TERM -x llama-bench", check=False)
                process.wait(timeout=30)
    record["exit"] = process.returncode
    record["seconds"] = round(time.monotonic() - started, 3)
    record["after"] = snapshot()
    if args.report and mode not in ("default", "norepack"):
        adb("pull", f"{REMOTE}/{prefix}.weights.json", str(local.with_suffix(".weights.json")), check=False)
        if local.with_suffix(".weights.json").exists():
            report = json.loads(local.with_suffix(".weights.json").read_text())
            record["devices"] = report["devices"]
            record["weight_capacity_bytes"] = sum(d["fixed_bytes"] + d["cache_bytes"] for d in report["devices"].values())
            if args.weight_gib:
                assert abs(record["weight_capacity_bytes"] / 2**30 - args.weight_gib) < 0.02
    if process.returncode == 0:
        record["bench"] = json.loads(local.with_suffix(".stdout").read_text())
    local.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    assert process.returncode == 0, f"{prefix} exited {process.returncode}; see {local}.log"
    for result in record["bench"]:
        assert len(result["samples_ns"]) == args.reps
        print(f"DONE {prefix} pp{result['n_prompt']} tg{result['n_gen']}: "
              f"{result['avg_ts']:.3f} +/- {result['stddev_ts']:.3f} tok/s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--case", nargs="+", required=True,
                        choices=[f"{b}-{m}" for b in ("cpu", "gpu", "htp", "mixed")
                                 for m in ("default", "fixed", "norepack", "swap")])
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--report", action="store_true", help="diagnostic run with per-eval weight reports")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--no-op-offload", action="store_true")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--ubatch", type=int, default=4)
    parser.add_argument("--slots", type=int, default=32)
    parser.add_argument("--weight-gib", type=int, choices=[4, 5, 6], help="fixed weights plus cache allocation target")
    parser.add_argument("--calibration-tag", default="cap01", help="same-build 32-slot report tag for capacity sizing")
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--gen", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    assert args.tag.isalnum()
    assert args.reps > 0 and args.slots > 0
    assert adb("get-state").stdout.strip() == "device", "default device unavailable"
    assert not shell("pidof llama-bench", check=False).stdout.strip(), "benchmark already running"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.deploy:
        deploy(args.tag)
    for case in args.case:
        run(case, args)


if __name__ == "__main__":
    main()
