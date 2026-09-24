#!/usr/bin/env python3
"""Compare fixed CPU Experts with swapped NPU/GPU routed/shared roles."""
import gzip
import hashlib
import json
import pathlib
import random
import shlex
import statistics
import subprocess
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OLD = ROOT / "task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence"
OUT = ROOT / "task/handoff/handoff-2026-09-24-reversed-placement/evidence"
BUILD = ROOT / "build-android-opencl-hexagon"
REMOTE = "/data/local/tmp/llama-reversed-placement"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"

def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", *args], text=True, capture_output=True, check=check)

def push(path, name):
    adb("push", str(path), f"{REMOTE}/{name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert adb("shell", f"sha256sum {REMOTE}/{name}").stdout.split()[0] == digest
    return digest

def main():
    assert adb("get-state").stdout.strip() == "device"
    assert not adb("shell", "pidof reversed-expert-probe fullmodel-oracle", check=False).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "a.json").exists() and not (OUT / "b.json").exists()
    adb("shell", f"mkdir -p {REMOTE}")
    hashes = {}
    names = ["reversed-expert-probe", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
    paths = [BUILD / "bin" / n for n in names] + [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so"]
    for path in paths:
        hashes[path.name] = push(path, path.name)
    with tempfile.TemporaryDirectory() as tmp:
        for original, name in [("search128.json.gz", "plan.json"), ("capture128-v2.json.layer0.bin.gz", "capture.bin")]:
            path = pathlib.Path(tmp) / name
            with gzip.open(OLD / original, "rb") as f:
                path.write_bytes(f.read())
            hashes[name] = push(path, name)
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", "GGML_HEXAGON_PROFILE=0"]
    results = []
    for order, tag in enumerate(("a", "b")):
        argv = [f"{REMOTE}/reversed-expert-probe", MODEL, f"{REMOTE}/capture.bin", f"{REMOTE}/plan.json", "0", f"{REMOTE}/{tag}.json", "32", str(order)]
        command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
        record = {"command": command, "hashes": hashes, "before_thermal": adb("shell", "dumpsys thermalservice").stdout,
                  "model": adb("shell", f"stat -c '%n %s %Y' {MODEL}").stdout}
        with (OUT / f"{tag}.log").open("w") as log:
            proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f"{tag}: {time.monotonic()-start:.0f}s", flush=True)
        record["exit"] = proc.returncode
        record["after_thermal"] = adb("shell", "dumpsys thermalservice").stdout
        (OUT / f"{tag}.run.json").write_text(json.dumps(record, indent=2)+"\n")
        assert proc.returncode == 0, f"inspect {tag}.log"
        adb("pull", f"{REMOTE}/{tag}.json", str(OUT / f"{tag}.json"))
        assert hashlib.sha256((OUT / f"{tag}.json").read_bytes()).hexdigest() == adb("shell", f"sha256sum {REMOTE}/{tag}.json").stdout.split()[0]
        data = json.loads((OUT / f"{tag}.json").read_text())
        results.append(data)
        print(tag, [round(t["median_us"]/1000, 3) for t in data["timing"]], flush=True)
    assert results[0]["plans"] == results[1]["plans"]
    policies, differences = [], []
    for data in results:
        rounds = {}
        for row in data["samples"]:
            assert sum(row["phases"]["pairs"]) == 512
            for start, end in row["phases"]["intervals_us"]:
                assert 0 <= start <= end <= row["phases"]["expert_wall_us"]
            by_policy = rounds.setdefault(row["round"], {})
            assert row["policy"] not in by_policy
            by_policy[row["policy"]] = row
        assert len(rounds) == 32 and all(set(r) == {0, 1} for r in rounds.values())
        differences.extend((r[0]["total_us"]-r[1]["total_us"])/1000 for r in rounds.values())
    for p in range(2):
        rows = [s for d in results for s in d["samples"] if s["policy"] == p]
        policies.append({"plan": results[0]["plans"][p], "samples": len(rows),
            "median_ms": statistics.median(r["total_us"] for r in rows)/1000,
            "process_median_ms": [d["timing"][p]["median_us"]/1000 for d in results],
            "all_numeric_checks_pass": all(r["check"]["passes_4percent_per_token"] for r in rows),
            "worst_token_nrmse": max(r["check"]["worst_token_nrmse"] for r in rows),
            "major_faults": sum(r["major_faults"] for r in rows),
            "pairs": rows[0]["phases"]["pairs"],
            "backend_total_ms": [statistics.median(sum(r["phases"]["backend_phases_us"][b]) for r in rows)/1000 for b in range(3)],
            "backend_compute_ms": [statistics.median(r["phases"]["backend_phases_us"][b][1] for r in rows)/1000 for b in range(3)],
            "phase_ms": {k: statistics.median(r["phases"][k] for r in rows)/1000 for k in ("gather_us", "expert_wall_us", "combine_us")}})
    rng = random.Random(240924)
    boot = sorted(statistics.median(rng.choices(differences, k=len(differences))) for _ in range(4000))
    summary = {"tokens": 128, "layer": 0, "scope": results[0]["scope"], "policies": policies,
               "reversed_latency_change_percent": 100*(policies[1]["median_ms"]/policies[0]["median_ms"]-1),
               "reversed_paired_saved_ms": statistics.median(differences), "bootstrap_95percent_saved_ms": [boot[100], boot[3899]],
               "reversed_paired_wins": sum(d > 0 for d in differences), "pairs": len(differences)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == "__main__":
    main()
