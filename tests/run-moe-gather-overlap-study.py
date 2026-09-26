#!/usr/bin/env python3
"""ABBA validation of independent per-backend input preparation."""
import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-host-parallel"
PRIOR = TASK.parent / "handoff-2026-09-26-model-extension"


def run(prompt, nt, tag, deploy=False, native=False, quality=False):
    report = TASK / "evidence" / f"{tag}.json"
    manifest = TASK / "evidence" / f"{tag}.run.json"
    if report.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
        print("REUSE", tag, flush=True)
        return
    args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", "npu" if native else "custom",
            "--tokens", str(nt), "--tag", tag, "--reps", "1" if quality else "6" if native else "12",
            "--output-dir", str(TASK / "evidence"), "--prompt-file", str(PRIOR / f"prompts/{prompt}.txt")]
    if deploy:
        args += ["--deploy"]
    if not native:
        plan = "technical-1024-cpu-plan.json.gz" if prompt == "technical" else f"evidence/{prompt}-{nt}-plan.json.gz"
        args += ["--plan-file", str(PRIOR / plan), "--shared-activations", "--direct-callback-io", "--activation-arena",
                 "--activation-lifetimes", "--fused-host-reduce"]
        if prompt == "technical":
            args += ["--transfer-placement"]
        if quality:
            args += ["--overlap-gather", "--quality-steps", "16", "--quality-tokens", str(TASK / f"{prompt}-{nt}-teacher-tokens.json")]
        else:
            args += ["--alternate-overlap-gather"]
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("pilot", "confirm"))
    a = p.parse_args()
    if a.stage == "pilot":
        run("technical", 768, "overlap-technical-768", deploy=True)
    else:
        for prompt, nt in (("museum", 128), ("museum", 768), ("programming", 768)):
            if nt == 768:
                run(prompt, nt, f"overlap-{prompt}-{nt}-npu-a", native=True)
            for repeat in ("a", "b"):
                run(prompt, nt, f"overlap-{prompt}-{nt}-{repeat}")
            if nt == 768:
                run(prompt, nt, f"overlap-{prompt}-{nt}-npu-b", native=True)
        run("museum", 128, "quality-overlap-museum-128", quality=True)


if __name__ == "__main__":
    main()
