#!/usr/bin/env python3
"""Repeat the frozen threeway-768 resident MoE protocol with one weight layout."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-10-21-shared-weight-moe-e2e"
OLD = ROOT / "task/handoff/handoff-2026-09-28-threeway-768"
REMOTE = "/data/local/tmp/llama-shared-weight-moe-e2e-20261021"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["smoke", "bench"])
    args = p.parse_args()
    evidence = TASK / "evidence"
    evidence.mkdir(exist_ok=True)
    inputs = {}
    for prompt in ("wetland", "foundry"):
        for path in (OLD / f"prompts/{prompt}.txt", TASK / f"inputs/{prompt}-c8-g15.json"):
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = TASK / "frozen-inputs.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == inputs, "frozen input changed"
    else:
        manifest.write_text(json.dumps(inputs, indent=2) + "\n")
    runs = [("wetland", "shared", "smoke")] if args.phase == "smoke" else [
        (prompt, layout, suffix)
        for prompt in ("wetland", "foundry")
        for layout, suffix in (("native", "a"), ("shared", "a"), ("shared", "b"), ("native", "b"))
    ]
    for prompt, layout, suffix in runs:
        tag = f"{prompt}-{layout}-{suffix}"
        command = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"),
                   "--mode", "custom", "--tokens", "768", "--reps", "1" if args.phase == "smoke" else "6",
                   "--tag", tag, "--output-dir", str(evidence), "--remote-dir", REMOTE,
                   "--prompt-file", str(OLD / f"prompts/{prompt}.txt"),
                   "--plan-file", str(TASK / f"inputs/{prompt}-c8-g15.json"),
                   "--shared-activations", "--direct-callback-io", "--activation-arena",
                   "--activation-lifetimes", "--fused-host-reduce", "--overlap-gather", "--hexagon-timing"]
        if layout == "shared":
            command += ["--shared-weights", "cpu4x8"]
        print("RUN", tag, flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
