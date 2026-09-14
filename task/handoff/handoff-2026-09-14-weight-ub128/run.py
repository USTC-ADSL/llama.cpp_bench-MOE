#!/usr/bin/env python3
"""Matched Android weight-capacity benchmarks; preserves original baseline runner."""
import argparse
import hashlib
import importlib.util
import json
import math
import pathlib
import re
import shlex
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "evidence"
OLD = HERE.parent / "handoff-2026-09-12-weight-perf/evidence"
REMOTE = "/data/local/tmp/llama-weight-ub128"
spec = importlib.util.spec_from_file_location("weight_perf", ROOT / "tests/run-weight-perf.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.OUT = OUT
base.REMOTE = REMOTE


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def capacity_prefix(weight_gib):
    return "" if weight_gib == 6 else f"g{weight_gib}-"


def configs(weight_gib=6):
    plans = {}
    target = weight_gib * 2**30
    prefix = capacity_prefix(weight_gib)
    for backend in ("cpu", "htp"):
        registry = json.loads((OLD / f"cap01-{backend}-swap.weights.json").read_text())
        prior = json.loads((OLD / f"cap01-{backend}-swap.config.json").read_text())
        fixed = {dev: row["fixed_bytes"] for dev, row in registry["devices"].items()}
        per_slot = {dev: row["cache_bytes"] // prior["budgets"][dev]["expert_slots"]
                    for dev, row in registry["devices"].items()}
        banks = sorted((w for w in registry["weights"] if w["residency"] == "swap"),
                       key=lambda w: (int(w["name"].split(".")[1]), w["name"]))
        pinned = []
        for weight in banks:
            if sum(fixed.values()) + weight["source_bytes"] + 128 * sum(per_slot.values()) > target:
                break
            pinned.append(weight["name"])
            fixed[weight["backend"]] += weight["source_bytes"]
        slots = (target - sum(fixed.values())) // sum(per_slot.values())
        assert slots >= 128
        weights = [{"pattern": r"ffn_(gate|up|down)_exps\.weight$", "residency": "swap"},
                   {"pattern": "^(?:" + "|".join(re.escape(name) for name in pinned) + ")$",
                    "residency": "fixed"}]
        budgets = {dev: {"total_mib": math.ceil((fixed[dev] + per_slot[dev] * slots) / 2**20) + 1024,
                         "reserve_mib": 128, "expert_slots": slots} for dev in fixed}
        config = {"version": 1, "budgets": budgets, "layers": [], "weights": weights}
        save(OUT / f"{prefix}{backend}.config.json", config)
        plans[backend] = {"pinned": pinned, "slots": slots, "fixed_bytes": fixed,
                          "cache_bytes": {dev: per_slot[dev] * slots for dev in fixed},
                          "predicted_weight_bytes": sum(fixed.values()) + sum(per_slot.values()) * slots}
    assert plans["cpu"]["pinned"] == plans["htp"]["pinned"]
    assert plans["cpu"]["slots"] == plans["htp"]["slots"]
    save(OUT / f"{prefix}allocation-plan.json", plans)
    print(json.dumps({b: {"fixed_banks": len(p["pinned"]), "slots": p["slots"],
                          "predicted_weight_gib": p["predicted_weight_bytes"] / 2**30}
                      for b, p in plans.items()}, indent=2), flush=True)


def snapshot():
    data = base.snapshot()
    data["cpu_frequency_caps"] = base.shell(
        "for p in /sys/devices/system/cpu/cpufreq/policy*; do echo $p; "
        "cat $p/scaling_max_freq; done", check=False).stdout
    return data


def run(args):
    tag = args.tag
    local = OUT / tag
    assert not local.with_suffix(".json").exists(), "Use a fresh tag"
    prefix = capacity_prefix(args.weight_gib)
    config = json.loads((OUT / f"{prefix}{args.backend}.config.json").read_text())
    if args.diagnostic or args.profile:
        config["report"] = f"{REMOTE}/{tag}.weights.json"
    save(local.with_suffix(".config.json"), config)
    base.adb("push", str(local.with_suffix(".config.json")), f"{REMOTE}/{tag}.config.json")
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}:/vendor/lib64", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3"]
    if args.profile:
        env += ["GGML_HEXAGON_PROFILE=1", "GGML_HEXAGON_VERBOSE=1"]
    argv = [f"{REMOTE}/llama-bench", "-m", base.MODEL,
            "-dev", "none" if args.backend == "cpu" else "HTP0",
            "-ngl", "0" if args.backend == "cpu" else "99",
            "-t", args.threads, "-p", args.prompt, "-n", "0", "-b", "512",
            "-ub", str(args.ubatch), "-r", str(args.reps), "-fa", "on",
            "-lm", "mmap", "-lzm", "off", "-o", "json", "-v",
            "--weight-config", f"{REMOTE}/{tag}.config.json"]
    if args.backend == "cpu":
        argv += ["-nopo", "1"]
    if args.diagnostic or args.profile:
        argv += ["--no-warmup"]
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
    record = {"command": command, "weight_gib": args.weight_gib,
              "diagnostic": args.diagnostic, "profile": args.profile,
              "before": snapshot(), "process_samples": []}
    print("START", tag, flush=True)
    started = time.monotonic()
    with local.with_suffix(".stdout").open("w") as out, local.with_suffix(".log").open("w") as err:
        proc = subprocess.Popen(["adb", "-s", base.SERIAL, "shell", command], stdout=out, stderr=err)
        try:
            while proc.poll() is None:
                time.sleep(2)
                status = base.shell("p=$(pidof llama-bench); if [ -n \"$p\" ]; then cat /proc/$p/status; fi", check=False).stdout
                record["process_samples"].append({"seconds": round(time.monotonic() - started, 3),
                    **{k: v.strip() for line in status.splitlines() if ":" in line
                       for k, v in [line.split(":", 1)] if k in ("VmRSS", "VmHWM", "VmSwap", "VmLck")}})
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(tag)
        finally:
            if proc.poll() is None:
                base.shell("pkill -TERM -x llama-bench", check=False)
                proc.wait(timeout=30)
            record["exit"] = proc.returncode
            record["seconds"] = round(time.monotonic() - started, 3)
            record["after"] = snapshot()
            save(local.with_suffix(".json"), record)
    if args.diagnostic or args.profile:
        base.adb("pull", f"{REMOTE}/{tag}.weights.json", str(local.with_suffix(".weights.json")), check=False)
        if local.with_suffix(".weights.json").exists():
            weights = json.loads(local.with_suffix(".weights.json").read_text())
            record["devices"] = weights["devices"]
            record["weight_bytes"] = sum(d["fixed_bytes"] + d["cache_bytes"] for d in weights["devices"].values())
    if proc.returncode == 0:
        record["bench"] = json.loads(local.with_suffix(".stdout").read_text())
    save(local.with_suffix(".json"), record)
    assert proc.returncode == 0, f"{tag} exited {proc.returncode}; see log"
    if "weight_bytes" in record:
        assert abs(record["weight_bytes"] / 2**30 - args.weight_gib) < 0.01
        print("CAPACITY", record["weight_bytes"], record["weight_bytes"] / 2**30, flush=True)
    for entry in record["bench"]:
        assert len(entry["samples_ns"]) == args.reps
        print(f"DONE {tag} t{entry['n_threads']} pp{entry['n_prompt']}: "
              f"{entry['avg_ts']:.3f} +/- {entry['stddev_ts']:.3f} tokens/s", flush=True)


def probe():
    path = base.BUILD / "bin/weight-slice-probe"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    prior = json.loads((HERE.parent / "handoff-2026-09-12-weight-prepare/evidence/slice-results.json").read_text())
    assert digest == prior["probe_sha256"]
    base.adb("push", str(path), f"{REMOTE}/weight-slice-probe")
    assert base.shell(f"sha256sum {REMOTE}/weight-slice-probe").stdout.split()[0] == digest
    record = {"probe_sha256": digest, "cases": []}
    for device in ("CPU", "HTP0"):
        command = f"cd {REMOTE} && exec " + shlex.join([
            "env", "LD_LIBRARY_PATH=.:/vendor/lib64", "ADSP_LIBRARY_PATH=.",
            "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=1",
            "./weight-slice-probe", device, "128", "127"])
        result = base.shell(command, check=False)
        (OUT / f"slice-{device}.log").write_text(result.stdout + result.stderr)
        record["cases"].append({"device": device, "command": command, "exit": result.returncode,
                                "stdout": result.stdout})
        save(OUT / "slice-results.json", record)
        assert result.returncode == 0, f"{device} slice probe failed"
        print(result.stdout, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "configure", "run", "probe"))
    parser.add_argument("--backend", choices=("cpu", "htp"))
    parser.add_argument("--tag")
    parser.add_argument("--threads", default="4")
    parser.add_argument("--prompt", default="128,512")
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--weight-gib", type=int, choices=(4, 5, 6), default=6)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert base.adb("get-state").stdout.strip() == "device"
    assert not base.shell("pidof llama-bench", check=False).stdout.strip()
    if args.action in ("prepare", "configure"):
        configs(args.weight_gib)
        if args.action == "prepare":
            prefix = capacity_prefix(args.weight_gib)
            base.deploy(f"{prefix}deploy")
            save(OUT / f"{prefix}device-before.json", snapshot())
            save(OUT / f"{prefix}model-stat.json", {"stat": base.shell(f"stat -c '%s %n' {base.MODEL}").stdout})
    elif args.action == "probe":
        probe()
    else:
        assert args.backend and args.tag and re.fullmatch(r"[a-z0-9-]+", args.tag)
        run(args)


if __name__ == "__main__":
    main()
