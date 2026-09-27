#!/usr/bin/env python3
"""Frozen first-pass confirmation: new processes, empty converted Expert caches."""
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-27-online-weights"
BUILD = ROOT / "build-android-opencl-hexagon"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [ROOT / "tests" / name for name in ["online-moe.cpp", "online-moe-runtime.h",
               "run-online-weights.py", "confirm-online-weights.py", "calibrate-online-weights.py"]]
    artifacts = [BUILD / "bin" / name for name in ["online-moe", "libllama.so", "libggml.so",
                 "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]]
    artifacts.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
    cases = [("weather", ["native", "online", "online", "native"]),
             ("workshop", ["online", "native", "native", "online"])]
    protocol = {"scope": "128-token complete first prefill; loaded model; empty Expert conversion cache; no prompt prepass",
                "calibration_sha256": digest(OUT / "online-cost-model.json"),
                "model_sha256": "8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473",
                "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "artifacts": {p.name: digest(p) for p in artifacts},
                "prompts": {c: digest(OUT / ("prompt-" + c + ".txt")) for c, _ in cases},
                "order": cases}
    freeze = OUT / "confirmation-freeze.json"
    if freeze.exists():
        assert json.loads(freeze.read_text()) == json.loads(json.dumps(protocol)), "frozen protocol changed"
    else:
        freeze.write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    for case, modes in cases:
        rows = []
        for i, mode in enumerate(modes):
            tag = f"confirm-{case}-{i}-{mode}"
            manifest = OUT / (tag + ".run.json")
            if not manifest.exists():
                cmd = [sys.executable, str(ROOT / "tests/run-online-weights.py"), "--deploy", "--tag", tag,
                       "--mode", mode, "--prompt", str(OUT / ("prompt-" + case + ".txt")),
                       "--cost-model", str(OUT / "online-cost-model.json"), "--fused-gpu", "--lazy-scratch",
                       "--memory", "--packed-head"]
                subprocess.run(cmd, cwd=ROOT, check=True)
            run = json.loads(manifest.read_text())
            assert run["exit"] == 0, "failed evidence retained"
            deployed = {pathlib.Path(line.split()[1]).name: line.split()[0] for line in run["artifacts"].splitlines()}
            assert all(deployed[k] == v for k, v in protocol["artifacts"].items())
            assert run["inputs"]["prompt.txt"]["sha256"] == protocol["prompts"][case]
            assert run["inputs"]["cost-model.json"]["sha256"] == protocol["calibration_sha256"]
            data = json.loads((OUT / (tag + ".json")).read_text())
            assert data["mode"] == mode and len(data["tokens"]) == 128
            rows.append({"tag": tag, "mode": mode, "ms": data["total_us"] / 1000,
                         "major_faults": data["major_faults"], "top1": data["top1"],
                         "extra_allowance_bytes": data.get("accounted_extra_allowance_bytes"),
                         "load_seconds": data["load_us"] / 1e6,
                         "maxrss_kib": data["maxrss_kib"]})
            print(tag, rows[-1]["ms"], flush=True)
        native = statistics.median(r["ms"] for r in rows if r["mode"] == "native")
        online = statistics.median(r["ms"] for r in rows if r["mode"] == "online")
        results[case] = {"runs": rows, "native_ms": native, "online_ms": online,
                         "throughput_gain_percent": (native / online - 1) * 100}
        (OUT / "confirmation-report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
