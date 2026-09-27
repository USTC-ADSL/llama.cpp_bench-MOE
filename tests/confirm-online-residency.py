#!/usr/bin/env python3
"""Independent v2 cohort; first prompt execution with symmetric output residency."""
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
    cases = [("river", ["native", "online", "online", "native"]),
             ("library", ["online", "native", "native", "online"])]
    sources = [ROOT / p for p in ["tests/online-moe.cpp", "tests/online-moe-runtime.h",
        "tests/run-online-weights.py", "tests/confirm-online-residency.py", "tests/expert-routing-probe.cpp",
        "include/llama.h", "src/llama-model.cpp", "ggml/src/ggml-hexagon/htp/hmx-mm-kernels-tiled.h"]]
    artifacts = [BUILD / "bin" / n for n in ["online-moe", "libllama.so", "libggml.so",
        "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]]
    artifacts += [BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so"]
    protocol = {
        "scope": "128-token complete first prefill; both modes restore CPU output residency, no Expert warmup",
        "timing": "restore separately and restore+prefill also retained; no prompt-specific preparation",
        "model_sha256": "8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473",
        "calibration_sha256": digest(OUT / "online-cost-model.json"),
        "sources": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "artifacts": {p.name: digest(p) for p in artifacts},
        "prompts": {c: digest(OUT / f"prompt-{c}.txt") for c, _ in cases}, "order": cases}
    freeze = OUT / "residency-confirmation-freeze.json"
    if freeze.exists():
        assert json.loads(freeze.read_text()) == json.loads(json.dumps(protocol)), "frozen protocol changed"
    else:
        freeze.write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    for case, modes in cases:
        rows, reference_tokens = [], None
        for i, mode in enumerate(modes):
            tag = f"residency-confirm-{case}-{i}-{mode}"
            manifest = OUT / f"{tag}.run.json"
            if not manifest.exists():
                cmd = [sys.executable, str(ROOT / "tests/run-online-weights.py"), "--deploy", "--tag", tag,
                    "--mode", mode, "--prompt", str(OUT / f"prompt-{case}.txt"),
                    "--cost-model", str(OUT / "online-cost-model.json"), "--fused-gpu", "--lazy-scratch",
                    "--memory", "--packed-head", "--bounded-source", "--resident-output"]
                if mode == "online":
                    cmd.append("--resident-shared-source")
                subprocess.run(cmd, cwd=ROOT, check=True)
            run = json.loads(manifest.read_text())
            assert run["exit"] == 0, "failed evidence retained"
            deployed = {pathlib.Path(line.split()[1]).name: line.split()[0] for line in run["artifacts"].splitlines()}
            assert all(deployed[k] == v for k, v in protocol["artifacts"].items())
            assert run["inputs"]["prompt.txt"]["sha256"] == protocol["prompts"][case]
            assert run["inputs"]["cost-model.json"]["sha256"] == protocol["calibration_sha256"]
            data = json.loads((OUT / f"{tag}.json").read_text())
            assert digest(OUT / f"{tag}.json") == run["output_sha256"]
            assert digest(OUT / f"{tag}.logits.bin") == run["logits_sha256"]
            assert data["mode"] == mode and len(data["tokens"]) == 128 and data["output_residency"]["enabled"]
            if reference_tokens is None:
                reference_tokens = data["tokens"]
            assert reference_tokens == data["tokens"], "token workload changed"
            rows.append({"tag": tag, "mode": mode, "ms": data["total_us"] / 1000,
                "restore_ms": data["output_residency"]["us"] / 1000,
                "restore_plus_prefill_ms": data["restore_plus_prefill_us"] / 1000,
                "major_faults": data["major_faults"], "maxrss_kib": data["maxrss_kib"],
                "load_seconds": data["load_us"] / 1e6,
                "accounted_extra_allowance_bytes": data.get("accounted_extra_allowance_bytes"),
                "source_cache_before": data["source_cache_before"], "source_cache_after": data["source_cache_after"]})
            print(tag, rows[-1]["ms"], flush=True)
        native = statistics.median(r["ms"] for r in rows if r["mode"] == "native")
        online = statistics.median(r["ms"] for r in rows if r["mode"] == "online")
        results[case] = {"runs": rows, "native_ms": native, "online_ms": online,
            "throughput_gain_percent": (native / online - 1) * 100}
        (OUT / "residency-confirmation-report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
