#!/usr/bin/env python3
"""Resident same-prefix numerical diagnostic; does not fit placement parameters."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-host-parallel"
PRIOR = TASK.parent / "handoff-2026-09-26-model-extension"


def main():
    for prompt, nt in (("museum", 128), ("programming", 768)):
        stem = f"{prompt}-{nt}"
        for policy in ("cpu", "npu-forced", "custom-forced", "custom-greedy", "npu-greedy"):
            tag = f"quality-{stem}-{policy}"
            report = TASK / "evidence" / f"{tag}.json"
            manifest = TASK / "evidence" / f"{tag}.run.json"
            if report.exists() and manifest.exists() and json.loads(manifest.read_text()).get("exit") == 0:
                print("REUSE", tag, flush=True)
            else:
                mode = policy.split("-")[0]
                args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", mode,
                        "--tokens", str(nt), "--tag", tag, "--reps", "1", "--quality-steps", "16",
                        "--output-dir", str(TASK / "evidence"), "--prompt-file", str(PRIOR / f"prompts/{prompt}.txt")]
                if prompt == "museum" and policy == "cpu":
                    args += ["--deploy"]
                if mode == "custom":
                    args += ["--plan-file", str(PRIOR / f"evidence/{stem}-plan.json.gz"),
                             "--shared-activations", "--direct-callback-io", "--activation-arena", "--fused-host-reduce"]
                if policy.endswith("forced"):
                    args += ["--quality-tokens", str(TASK / f"{stem}-teacher-tokens.json")]
                print("RUN", " ".join(args), flush=True)
                subprocess.run(args, cwd=ROOT, check=True)
            if policy == "cpu":
                d = json.loads(report.read_text())
                tokens = [s["next_input_token"] for s in d["quality"]["steps"][:-1]]
                (TASK / f"{stem}-teacher-tokens.json").write_text(json.dumps(tokens) + "\n")


if __name__ == "__main__":
    main()
