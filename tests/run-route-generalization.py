#!/usr/bin/env python3
"""Small resident placement-transfer screen; device jobs run sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", type=Path)
    args = parser.parse_args()
    task = args.task_dir.resolve()
    evidence = task / "evidence"
    failures = []
    # Start at the historically positive length. The second prompt reverses lengths.
    for prompt, lengths in (("technical", (128, 64, 256, 512)),
                            ("narrative", (512, 256, 64, 128))):
        for nt in lengths:
            for mode, cohort in (("npu", "a"), ("custom", "a"), ("cpu", "a"),
                                 ("gpu", "a"), ("custom", "b"), ("npu", "b")):
                tag = f"{prompt}-{nt}-{mode}-{cohort}"
                manifest = evidence / f"{tag}.run.json"
                report = evidence / f"{tag}.json"
                if manifest.exists():
                    previous = json.loads(manifest.read_text())
                    if previous.get("exit") == 0 and report.exists():
                        print(f"SKIP verified completed process {tag}", flush=True)
                        continue
                    failures.append(tag)
                    print(f"SKIP previous failure {tag}; use a fresh tag to retry", flush=True)
                    continue
                command = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"),
                           "--mode", mode, "--tokens", str(nt), "--tag", tag, "--reps", "4",
                           "--prompt-file", str(task / "prompts" / f"{prompt}.txt"),
                           "--output-dir", str(evidence)]
                if mode == "custom":
                    source_nt = 128 if nt <= 128 else 256
                    plan = ROOT / f"task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence/search{source_nt}.json.gz"
                    command += ["--plan-file", str(plan), "--transfer-placement", "--shared-activations",
                                "--direct-callback-io", "--activation-arena", "--fused-host-reduce"]
                print(f"START {tag}", flush=True)
                result = subprocess.run(command, cwd=ROOT)
                if result.returncode:
                    failures.append(tag)
                    print(f"FAILED {tag}; retaining evidence", flush=True)
                devices = subprocess.run(["adb", "-s", "fd8657d6", "get-state"], capture_output=True, text=True)
                if devices.stdout.strip() != "device":
                    raise RuntimeError("fd8657d6 disconnected; stop without switching device")
    (task / "screen-failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"SCREEN COMPLETE failures={failures}", flush=True)


if __name__ == "__main__":
    main()
