#!/usr/bin/env python3
"""Technical-only calibration and frozen complete-model confirmation on fd8657d6."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TASK = ROOT / "task/handoff/handoff-2026-09-26-execution-model"
PRIOR = TASK.parent / "handoff-2026-09-26-model-extension"
BASE = TASK.parent / "handoff-2026-09-26-placement-model/model.json"
spec = importlib.util.spec_from_file_location("execution", ROOT / "tests/moe-execution-model.py")
execution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(execution)


def run(prompt, nt, tag, mode, plan=None, alternate=False, reference=None, cost=None):
    ev = TASK / "evidence"
    if (ev / f"{tag}.json").exists():
        manifest = execution.read(ev / f"{tag}.run.json")
        assert manifest["exit"] == 0
        binary_hash = hashlib.sha256((ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes()).hexdigest()
        assert f"{binary_hash}  /data/local/tmp/llama-fullmodel-oracle/fullmodel-oracle" in manifest["hashes"]
        print("REUSE", tag, flush=True)
        return
    args = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", mode, "--tokens", str(nt),
            "--tag", tag, "--reps", "12" if alternate else "6", "--output-dir", str(ev), "--prompt-file", str(prompt)]
    if plan:
        args += ["--plan-file", str(plan)]
    if reference:
        args += ["--reference", reference]
    if cost:
        args += ["--cost-model", str(cost)]
    if mode == "custom":
        args += ["--shared-activations", "--direct-callback-io", "--activation-arena", "--activation-lifetimes", "--fused-host-reduce"]
        if alternate:
            args += ["--alternate-overlap-gather"]
        elif execution.read(plan)["execution_policy"]["overlap_gather"]:
            args += ["--overlap-gather"]
    print("RUN", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def calibration():
    if (TASK / "freeze.json").exists():
        frozen = execution.read(TASK / "freeze.json")
        assert hashlib.sha256((TASK / "execution-model.json").read_bytes()).hexdigest() == frozen["model_sha256"]
        print("Frozen model exists; refusing to refit", flush=True)
        return
    provenance = {}
    for nt in (128, 512, 768, 1024):
        for family in execution.FAMILIES:
            output = TASK / f"technical-{nt}-{family}-plan.json"
            if nt == 768:
                capture_path = TASK / "evidence/technical-768-capture.json"
                capture, layers = execution.base.load_routes(capture_path)
                plan = execution.context.calibration_plan(execution.read(BASE), capture, layers, family, execution.SHARED[family])
                provenance[str(output)] = {"capture_sha256": hashlib.sha256(execution.raw(capture_path)).hexdigest(), "source": str(capture_path)}
            else:
                directory = PRIOR if nt == 1024 else TASK.parent / "handoff-2026-09-26-context-model"
                source = directory / output.name
                plan = execution.read(source)
                provenance[str(output)] = {"sha256": hashlib.sha256(execution.raw(source)).hexdigest(), "source": str(source)}
            execution.base.write(output, plan)
    execution.base.write(TASK / "route-provenance.json", provenance)
    prompt = PRIOR / "prompts/technical.txt"
    for index, nt in enumerate((128, 512, 768, 1024)):
        run(prompt, nt, f"technical-{nt}-native-a", "npu")
        families = execution.FAMILIES if index % 2 == 0 else tuple(reversed(execution.FAMILIES))
        for family in families:
            run(prompt, nt, f"technical-{nt}-{family}", "custom", TASK / f"technical-{nt}-{family}-plan.json", alternate=True)
        run(prompt, nt, f"technical-{nt}-native-b", "npu")
    model = execution.fit(TASK, BASE, [128, 512, 768, 1024])
    execution.base.write(TASK / "execution-model.json", model)
    freeze = {"model_sha256": hashlib.sha256((TASK / "execution-model.json").read_bytes()).hexdigest(),
              "binary_sha256": hashlib.sha256((ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes()).hexdigest(),
              "scope": "technical-only coefficients frozen before confirmation; no held-out timing used"}
    execution.base.write(TASK / "freeze.json", freeze)


def confirm():
    frozen = execution.read(TASK / "freeze.json")
    assert hashlib.sha256((TASK / "execution-model.json").read_bytes()).hexdigest() == frozen["model_sha256"]
    contract = {"model_sha256": frozen["model_sha256"], "prompts":{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (PRIOR / "prompts/museum.txt", PRIOR / "prompts/programming.txt", TASK / "prompts/astronomy.txt")},
                "cases":[["museum",128],["museum",768],["programming",768],["astronomy",384],["astronomy",768]],
                "scope":"frozen before confirmation, including new astronomy prompt; forced custom only if automatic fallback"}
    contract_path = TASK / "confirmation-contract.json"
    if contract_path.exists():
        assert execution.read(contract_path) == contract
    else:
        execution.base.write(contract_path, contract)
    for prompt_name, nt in contract["cases"]:
        prompt = (TASK if prompt_name == "astronomy" else PRIOR) / f"prompts/{prompt_name}.txt"
        source = PRIOR / f"evidence/{prompt_name}-{nt}-plan.json.gz"
        if prompt_name == "astronomy":
            tag = f"{prompt_name}-{nt}-capture"
            run(prompt, nt, tag, "capture")
            capture, layers = execution.base.load_routes(TASK / f"evidence/{tag}.json")
            source = TASK / f"{prompt_name}-{nt}-routes.json"
            execution.base.write(source, {**capture, "layers":layers})
        run(prompt, nt, f"{prompt_name}-{nt}-plan", "predict", source, cost=TASK / "execution-model.json")
        plan = TASK / f"evidence/{prompt_name}-{nt}-plan.json"
        d = execution.read(plan)
        capture = execution.base.read(source)
        host = execution.make_plan(execution.read(TASK / "execution-model.json"), capture, capture["layers"], d["path_decision"]["observed_planning_us"])
        assert host["execution_mode"] == d["execution_mode"] and host["execution_policy"] == d["execution_policy"]
        for a, b in zip(host["layers"], d["layers"]):
            assert a["selected"] == b["selected"]
            assert abs(a["prediction"]["total_us"] - b["prediction"]["total_us"]) < 1e-6
        execution.base.write(TASK / f"{prompt_name}-{nt}-host-plan.json", host)
        modes = ["npu", "auto"] + (["custom"] if d["execution_mode"] == "npu" else [])
        for repeat in ("a", "b"):
            for mode in (modes if repeat == "a" else reversed(modes)):
                run(prompt, nt, f"{prompt_name}-{nt}-{mode}-{repeat}", mode, plan if mode != "npu" else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "confirm"))
    args = p.parse_args()
    calibration() if args.stage == "calibrate" else confirm()
