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


def fast_validation():
    model = TASK / "execution-model.json"
    assert hashlib.sha256(model.read_bytes()).hexdigest() == execution.read(TASK / "freeze.json")["model_sha256"]
    result = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "binary_sha256": hashlib.sha256((ROOT / "build-android-opencl-hexagon/bin/fullmodel-oracle").read_bytes()).hexdigest(),
              "scope": "unchanged coefficients and placements; reaccount original inference observations using new measured planner cost; smoke runs validate outputs, not new throughput claims", "cases":[]}
    for name, nt in execution.read(TASK / "confirmation-contract.json")["cases"]:
        prompt = (TASK if name == "astronomy" else PRIOR) / f"prompts/{name}.txt"
        source = TASK / f"evidence/{name}-{nt}-plan.json"
        tag = f"fast-{name}-{nt}-plan"
        run(prompt, nt, tag, "predict", source, cost=model)
        before = execution.read(source); after = execution.read(TASK / f"evidence/{tag}.json")
        for key in ("custom_candidates", "layers", "execution_policy", "execution_mode"):
            assert before[key] == after[key], f"fast planner changed {key}"
        row = {"prompt":name,"tokens":nt,"old_decision_us":before["decision_device_total_us"],
               "new_decision_us":after["decision_device_total_us"],"reaccounted_samples":[]}
        for repeat in ("a","b"):
            old = execution.read(TASK / f"evidence/{name}-{nt}-auto-{repeat}.json")["timing"]["median_us"]
            native = execution.read(TASK / f"evidence/{name}-{nt}-npu-{repeat}.json")["timing"]["median_us"]
            row["reaccounted_samples"].append({"repeat":repeat,"native_ms":native/1000,"auto_ms":(old+after["decision_device_total_us"])/1000,
                "throughput_gain_percent":100*(native/(old+after["decision_device_total_us"])-1)})
        if (name,nt) in (("museum",128),("astronomy",768)):
            smoke = f"fast-{name}-{nt}-smoke"
            run(prompt, nt, smoke, "auto", TASK / f"evidence/{tag}.json")
            assert execution.raw(TASK / f"evidence/{smoke}.json.logits.bin") == execution.raw(TASK / f"evidence/{name}-{nt}-auto-a.json.logits.bin")
            d = execution.read(TASK / f"evidence/{smoke}.json")
            assert d["all_round_logits_bitwise_equal"] and d["overlap_gather"] == after["execution_policy"]["overlap_gather"]
            row["integration_logits_bitwise_equal"] = True
        result["cases"].append(row)
    prompt = PRIOR / "prompts/technical.txt"
    legacy_model = PRIOR / "context-model.json"
    run(prompt, 768, "legacy-context-plan", "predict", reference="technical-768-capture", cost=legacy_model)
    routes = execution.read(TASK / "technical-768-cpu-plan.json")
    host = execution.context.make_plan(execution.read(legacy_model), routes, routes["layers"])
    device = execution.read(TASK / "evidence/legacy-context-plan.json")
    for a,b in zip(host["layers"],device["layers"]):
        assert a["selected"] == b["selected"] and abs(a["prediction"]["total_us"]-b["prediction"]["total_us"]) < 1e-6
    result["legacy_v1_placement_parity"] = True
    forced = execution.read(TASK / "evidence/fast-museum-128-plan.json")
    forced["execution_mode"] = "npu"
    forced["path_decision"] = {"execution_mode":"npu", "scope":"forced native for v2 auto flag-reset functional test; not a model decision or performance result"}
    execution.base.write(TASK / "forced-native-functional-plan.json",forced)
    run(PRIOR / "prompts/museum.txt",128,"fast-native-functional","auto",TASK / "forced-native-functional-plan.json")
    native = execution.read(TASK / "evidence/fast-native-functional.json")
    assert native["mode"]=="npu" and not any(native[k] for k in ("shared_activations","activation_arena","activation_lifetimes","overlap_gather","fused_host_reduce"))
    assert execution.raw(TASK / "evidence/fast-native-functional.json.logits.bin") == execution.raw(TASK / "evidence/museum-128-npu-a.json.logits.bin")
    result["native_auto_flags_and_logits"] = True
    execution.base.write(TASK / "fast-validation.json",result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("calibrate", "confirm", "fast"))
    args = p.parse_args()
    {"calibrate":calibration,"confirm":confirm,"fast":fast_validation}[args.stage]()
