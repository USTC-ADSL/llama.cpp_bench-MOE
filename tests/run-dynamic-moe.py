#!/usr/bin/env python3
"""Android functional checks; conversion and phase-switch timing are not steady-state benchmarks."""
import array
import hashlib
import json
import math
import pathlib
import shlex
import subprocess
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "build-android-opencl-hexagon"
OUT = ROOT / "task/handoff/handoff-2026-09-24-dynamic-placement" / os.environ.get("DYNAMIC_EVIDENCE", "evidence")
REMOTE = "/data/local/tmp/llama-dynamic-moe"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
TOKENS = int(os.environ.get("DYNAMIC_TOKENS", "16"))


def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", *args], text=True, capture_output=True, check=check)


def push(path, name):
    adb("push", str(path), f"{REMOTE}/{name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert adb("shell", f"sha256sum {REMOTE}/{name}").stdout.split()[0] == digest
    return digest


def main():
    assert adb("get-state").stdout.strip() == "device"
    assert not adb("shell", "pidof dynamic-moe fullmodel-oracle llama-cli", check=False).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not any(OUT.iterdir()), "use a fresh evidence directory; archived and failed runs must be preserved"
    adb("shell", f"mkdir -p {REMOTE}")
    names = ["dynamic-moe", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
    paths = [BUILD / "bin" / name for name in names] + [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so"]
    hashes = {p.name: push(p, p.name) for p in paths}
    hashes["prompt.txt"] = push(ROOT / "tests/expert-routing-prompt.txt", "prompt.txt")
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}",
           "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0"]

    def run(tag, config, mode):
        path = OUT / f"{tag}.policy.json"
        assert not (OUT / f"{tag}.json").exists(), "refusing to overwrite evidence"
        path.write_text(json.dumps(config, indent=2)+"\n")
        policy_hash = push(path, "policy.json")
        command = f"cd {REMOTE} && exec " + shlex.join(env + [f"{REMOTE}/dynamic-moe", MODEL,
            f"{REMOTE}/prompt.txt", str(TOKENS), "2", f"{REMOTE}/policy.json", f"{REMOTE}/{tag}.json", mode])
        with (OUT / f"{tag}.log").open("w") as log:
            proc = subprocess.run(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
        (OUT / f"{tag}.run.json").write_text(json.dumps({"command": command, "hashes": hashes,
            "policy_hash": policy_hash, "exit": proc.returncode}, indent=2)+"\n")
        assert proc.returncode == 0, f"inspect {tag}.log"
        for suffix in [".json", ".json.logits.bin"] + ([".json.capture.bin"] if config.get("capture_prefill") else []):
            local = OUT / (tag+suffix)
            adb("pull", f"{REMOTE}/{tag}{suffix}", str(local))
            assert hashlib.sha256(local.read_bytes()).hexdigest() == adb("shell", f"sha256sum {REMOTE}/{tag}{suffix}").stdout.split()[0]
        data = json.loads((OUT / f"{tag}.json").read_text())
        print(tag, [round(s["total_us"]/1000, 3) for s in data["samples"]], flush=True)
        if mode == "dynamic":
            for sample in data["samples"]:
                assert len(sample["layers"]) == 24
                for layer in sample["layers"]:
                    assert sum(layer["counts"]) == layer["tokens"]*4
                    assert len(layer["placement"]["owner"]) == 60
                    assert all(w["weight_buffer_bytes"] <= 256*1024*1024 for w in layer["waves"])
        return data

    cpu = {"capture_prefill": bool(os.environ.get("DYNAMIC_CAPTURE")), "group_size": 8, "prefill": {"dense": "cpu", "routed": "cpu", "shared": "cpu"},
           "decode": {"dense": "cpu", "routed": "cpu", "shared": "cpu"}}
    reference = run("native-cpu", cpu, "native")
    forced = [s["input_token"] for s in reference["samples"][1:]]
    cpu["decode_tokens"] = forced
    cpu["group_size"] = 1
    cpu["verify_layer"] = 0
    single = run("dynamic-cpu-single", cpu, "dynamic")
    mixed = json.loads((ROOT / "tests/dynamic-moe-policy.json").read_text())
    mixed["decode_tokens"] = forced
    if os.environ.get("DYNAMIC_VERIFY_ALL"):
        mixed["verify_layer"] = -2
    heterogeneous = run("dynamic-mixed", mixed, "dynamic")
    native_phase = run("native-phase-switch", mixed, "native")

    def compare(tag, threshold):
        actual, expected = array.array("f"), array.array("f")
        actual.frombytes((OUT / f"{tag}.json.logits.bin").read_bytes())
        expected.frombytes((OUT / "native-cpu.json.logits.bin").read_bytes())
        assert len(actual) == len(expected) == 3*reference["vocab_size"]
        errors = []
        for step in range(3):
            lo, hi = step*reference["vocab_size"], (step+1)*reference["vocab_size"]
            error = math.sqrt(sum((a-b)**2 for a, b in zip(actual[lo:hi], expected[lo:hi])) / sum(b*b for b in expected[lo:hi]))
            assert math.isfinite(error) and error < threshold, (tag, step, error)
            errors.append(error)
        return errors

    summary = {"scope": f"functional validation, {TOKENS}-token full-model prefill + 2 teacher-forced decode steps",
        "cpu_single_logit_nrmse": compare("dynamic-cpu-single", 0.00001),
        "mixed_logit_nrmse": compare("dynamic-mixed", 0.10),
        "native_phase_logit_nrmse": compare("native-phase-switch", 0.10),
        "mixed_peak_converted_weight_buffer_bytes": heterogeneous["peak_converted_weight_buffer_bytes"],
        "mixed_phase_switch": heterogeneous["phase_switch"],
        "timing_ms": {"native_cpu": [s["total_us"]/1000 for s in reference["samples"]],
            "dynamic_cpu_single": [s["total_us"]/1000 for s in single["samples"]],
            "mixed": [s["total_us"]/1000 for s in heterogeneous["samples"]],
            "native_phase": [s["total_us"]/1000 for s in native_phase["samples"]]},
        "limitations": "10% full-logit NRMSE is a smoke-test bound, not quality equivalence; no physical RAM bound or zero-copy claim"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
