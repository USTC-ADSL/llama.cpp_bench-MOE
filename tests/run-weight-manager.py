#!/usr/bin/env python3
"""Android full-logit regression for bounded, explicitly placed weight caches."""
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
REMOTE = "/data/local/tmp/llama-weight-manager"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
SERIAL = "fd8657d6"
OUT = ROOT / "task/handoff/handoff-2026-09-11-weight-manager/evidence"


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", SERIAL, *args], text=True, capture_output=True, check=check)


def shell(command, check=True):
    return adb("shell", command, check=check)


def config(case, report):
    budgets = {"CPU": {"total_mib": 12288, "reserve_mib": 128, "expert_slots": 8}}
    devices = "none"
    layers = []
    if case.startswith("gpu"):
        devices = "GPUOpenCL"
    elif case.startswith("htp"):
        devices = "HTP0"
    elif case.startswith("mixed"):
        devices = "GPUOpenCL,HTP0"
        layers = [{"first": 0, "last": 7, "backend": "CPU"},
                  {"first": 8, "last": 15, "backend": "GPUOpenCL"},
                  {"first": 16, "last": 23, "backend": "HTP0"}]
    if devices != "none":
        for device in devices.split(","):
            budgets[device] = {"total_mib": 12288, "reserve_mib": 128, "expert_slots": 8}
    weights = []
    if "swap" in case or case in ("tiny-slots", "cache-hits"):
        weights = [{"pattern": r"ffn_(gate|up|down)_exps\.weight$", "residency": "swap"}]
        for value in budgets.values():
            value["total_mib"] = 3072
    if "whole-swap" in case:
        weights.append({"pattern": r"blk\.[01]\.(attn_q|ffn_gate_shexp)\.weight$", "residency": "swap"})
    if case == "tiny-slots":
        budgets["CPU"]["expert_slots"] = 2
    if case == "cache-hits":
        budgets["CPU"]["expert_slots"] = 256
    if case == "bad-budget":
        budgets["CPU"] = {"total_mib": 1, "reserve_mib": 0, "expert_slots": 8}
    if case == "bad-pin":
        weights = [{"pattern": r"token_embd\.weight$", "residency": "swap", "pin_experts": [0]}]
    if case == "mixed-swap-4g":
        for name, total in (("CPU", 1024), ("GPUOpenCL", 1024), ("HTP0", 2048)):
            budgets[name]["total_mib"] = total
    return devices, {"version": 1, "budgets": budgets, "layers": layers, "weights": weights, "report": report}


def deploy(tag):
    existing = shell(f"test -d {REMOTE}", check=False).returncode == 0
    if existing:
        assert shell(f"test -f {REMOTE}/weight-manager-probe", check=False).returncode == 0, "unowned test directory"
    else:
        shell(f"mkdir {REMOTE}")
    assert not shell("pidof weight-manager-probe", check=False).stdout.strip(), "probe already running"
    names = ["weight-manager-probe", "weight-slice-probe", "weight-manager-api-probe",
             "libllama.so", "libggml.so", "libggml-base.so",
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
    shell(f"chmod 755 {REMOTE}/weight-manager-probe")


def run(case, tag, ubatch, slots, semantic, stop_eog, default_layout):
    prefix = f"{tag}-{case}"
    local = OUT / prefix
    assert not local.with_suffix(".log").exists(), "use a fresh tag"
    devices, policy = config(case, f"{REMOTE}/{prefix}.weights.json")
    if slots is not None:
        for value in policy["budgets"].values():
            value["expert_slots"] = slots
    cfg = local.with_suffix(".config.json")
    cfg.write_text(json.dumps(policy, indent=2) + "\n")
    adb("push", str(cfg), f"{REMOTE}/{cfg.name}")
    config_arg = "-" if case.endswith("-ref") else f"{REMOTE}/{cfg.name}"
    argv = [f"{REMOTE}/weight-manager-probe", MODEL, devices, config_arg, f"{REMOTE}/{prefix}", str(ubatch)]
    reference = "mixed-fixed" if case.startswith("mixed") else case.split("-")[0] + "-ref"
    if case in ("tiny-slots", "cache-hits", "bad-budget", "bad-pin"):
        reference = "cpu-ref"
    teacher = OUT / f"{tag}-{reference}.tokens.txt"
    if not case.endswith("-ref") and case not in (reference, "bad-budget", "bad-pin", "tiny-slots"):
        assert teacher.exists(), f"run {reference} with tag {tag} first"
    if not semantic and not case.endswith("-ref") and teacher.exists() and case != reference:
        adb("push", str(teacher), f"{REMOTE}/teacher.txt")
        argv.append(f"{REMOTE}/teacher.txt")
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_OPENCL_KERNEL_CACHE_DIR=off"]
    if case == "cpu-fixed-lazy":
        env.append("WEIGHT_PROBE_LAZY_ON=1")
    if semantic:
        env.append("WEIGHT_PROBE_SEMANTIC=1")
    if stop_eog:
        env.append("WEIGHT_PROBE_STOP_EOG=1")
    if default_layout:
        env.append("WEIGHT_PROBE_DEFAULT_LAYOUT=1")
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
    started = time.monotonic()
    samples = []
    with local.with_suffix(".log").open("w") as log:
        log.write(command + "\n")
        log.flush()
        process = subprocess.Popen(["adb", "-s", SERIAL, "shell", command], stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                if time.monotonic() - started > 600:
                    shell("pkill -TERM -f '^/data/local/tmp/llama-weight-manager/weight-manager-probe '", check=False)
                    raise TimeoutError(case)
                sample = shell("pid=$(pidof weight-manager-probe); if [ -n \"$pid\" ]; then cat /proc/$pid/status; fi", check=False).stdout
                values = {line.split(":")[0]: line.split(":", 1)[1].strip() for line in sample.splitlines() if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:"))}
                if values:
                    samples.append({"seconds": round(time.monotonic() - started, 3), **values})
                time.sleep(1)
        finally:
            process.wait(timeout=30)
    result = {"case": case, "exit": process.returncode, "seconds": round(time.monotonic() - started, 3), "samples": samples}
    local.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
    for suffix in ("weights.json", "summary.json", "tokens.txt", "logits.f32", "output.txt", "prompt.txt", "prompt.tokens.txt"):
        adb("pull", f"{REMOTE}/{prefix}.{suffix}", str(OUT / f"{prefix}.{suffix}"), check=False)
    if case in ("bad-budget", "bad-pin", "tiny-slots"):
        expected = {"bad-budget": "weight budget denied", "bad-pin": "pin_experts requires a valid",
                    "tiny-slots": "needs 4 experts but pool has 2 slots"}[case]
        assert process.returncode != 0 and expected in local.with_suffix(".log").read_text(), result
        result["expected_failure"] = True
    else:
        assert process.returncode == 0, result
        summary = json.loads(local.with_suffix(".summary.json").read_text())
        result["summary"] = summary
        if semantic:
            output = local.with_suffix(".output.txt").read_text()
            result["output"] = output
            result["semantic_checks"] = {
                "recipient_bob": bool(re.search(r"\bbob\b", output, re.I)),
                "three_books": bool(re.search(r"\b(?:three|3) books\b", output, re.I)),
                "delivery_friday": bool(re.search(r"\bfriday\b", output, re.I)),
                "cancelled_monday": bool(re.search(r"\bmonday\b", output, re.I) and re.search(r"\bcancell?ed\b", output, re.I)),
            }
            local.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
            assert (0 < summary["steps"] <= 32 if stop_eog else summary["steps"] == 32) and not summary["teacher_forced"], result
            assert summary.get("stop_eog", False) == stop_eog, result
            assert len(local.with_suffix(".tokens.txt").read_text().splitlines()) == summary["steps"]
            assert len(local.with_suffix(".prompt.tokens.txt").read_text().splitlines()) == summary["prompt_tokens"]
            assert all(result["semantic_checks"].values()), result
        if teacher.exists() and case != reference:
            import numpy as np
            a = np.fromfile(OUT / f"{tag}-{reference}.logits.f32", dtype=np.float32)
            b = np.fromfile(local.with_suffix(".logits.f32"), dtype=np.float32)
            assert a.shape == b.shape and np.isfinite(b).all()
            result["logits_rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))
            result["logits_max_abs"] = float(np.max(np.abs(a - b)))
            result["tokens_equal"] = teacher.read_text() == local.with_suffix(".tokens.txt").read_text()
            if semantic:
                assert (OUT / f"{tag}-{reference}.prompt.tokens.txt").read_text() == local.with_suffix(".prompt.tokens.txt").read_text()
                assert json.loads((OUT / f"{tag}-{reference}.summary.json").read_text())["semantic"]
                assert json.loads((OUT / f"{tag}-{reference}.summary.json").read_text()).get("stop_eog", False) == stop_eog
            local.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
            assert result["tokens_equal"] and result["logits_rmse"] < 0.05, result
        report = local.with_suffix(".weights.json")
        if report.exists():
            info = json.loads(report.read_text())
            result["devices"] = info["devices"]
            for budget in info["devices"].values():
                assert budget["peak_bytes"] <= budget["total_bytes"]
            if "swap" in case or case == "cache-hits":
                assert sum(b["misses"] for b in info["devices"].values()) > 0
                assert sum(b["evictions"] for b in info["devices"].values()) > 0
            if case == "cache-hits":
                assert sum(b["hits"] for b in info["devices"].values()) > 0
    local.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--tag", default="dev01")
    parser.add_argument("--case", required=True, nargs="+", choices=["cpu-ref", "cpu-fixed", "cpu-fixed-lazy", "cpu-swap", "cpu-whole-swap",
        "gpu-ref", "gpu-swap", "gpu-whole-swap", "htp-ref", "htp-swap", "htp-whole-swap",
        "mixed-fixed", "mixed-swap", "mixed-swap-4g", "bad-budget", "bad-pin", "tiny-slots", "cache-hits"])
    parser.add_argument("--ubatch", type=int, default=1)
    parser.add_argument("--slots", type=int)
    parser.add_argument("--semantic", action="store_true", help="natural chat prompt near 128 tokens, 32 freely generated tokens and fact checks")
    parser.add_argument("--stop-eog", action="store_true", help="allow semantic generation to finish naturally before 32 tokens")
    parser.add_argument("--default-layout", action="store_true", help="retain default CPU weight repacking for fixed-weight regression")
    args = parser.parse_args()
    if args.stop_eog and not args.semantic:
        parser.error("--stop-eog requires --semantic")
    assert args.tag.isalnum(), "tag must be alphanumeric"
    assert adb("get-state").stdout.strip() == "device", "default device is unavailable"
    OUT.mkdir(parents=True, exist_ok=True)
    if args.deploy:
        deploy(args.tag)
    for case in args.case:
        run(case, args.tag, args.ubatch, args.slots, args.semantic, args.stop_eog, args.default_layout)


if __name__ == "__main__":
    main()
