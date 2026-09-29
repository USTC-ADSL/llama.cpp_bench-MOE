#!/usr/bin/env python3
"""Bounded integration verification; leaves historical experiment evidence intact."""
import argparse
import importlib.util
import json
import math
import pathlib
import shlex
import struct
import subprocess
import sys

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
EV = TASK / "evidence"
spec = importlib.util.spec_from_file_location("execution_study", ROOT / "tests/run-moe-route-execution-study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)
study.TASK, study.EV = TASK, EV
NAMES = ["fullmodel-oracle", "moe-route-unit-probe", "route-resident-probe", "htp-indexed-gather-probe", "online-weight-probe"]


def build():
    compiler = pathlib.Path("/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++")
    commands = []
    for name in NAMES:
        cmd = [str(compiler), "-std=c++17", "-O2", "-Iggml/include", "-Iinclude", "-Ivendor/nlohmann",
               f"tests/{name}.cpp", "-Lbuild-android-opencl-hexagon/bin", "-lllama", "-lggml", "-lggml-base",
               "-o", f"build-android-opencl-hexagon/bin/{name}"]
        commands.append(shlex.join(cmd))
        subprocess.run(cmd, cwd=ROOT, check=True)
    (TASK / "probe-build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def probes():
    for kind in ("regression", "regression-no-nx", "regression-hvx", "resident-fused"):
        print("running", kind, flush=True)
        (study.regression if kind.startswith("regression") else study.probe)(kind)
    path = study.BUILD / "bin/online-weight-probe"
    study.adb("push", path, study.REMOTE + "/" + path.name)
    assert study.adb("shell", f"sha256sum {study.REMOTE}/{path.name}").split()[0] == study.sha(path)
    cmd = shlex.join(["env", f"LD_LIBRARY_PATH={study.REMOTE}", f"ADSP_LIBRARY_PATH={study.REMOTE}",
                     "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=2",
                     study.REMOTE + "/" + path.name, study.MODEL, study.REMOTE + "/online-merge.json", "4"])
    assert not (EV / "online-merge.json").exists()
    with (EV / "online-merge.log").open("w") as log:
        result = subprocess.run(["adb", "-s", "fd8657d6", "shell", cmd], stdout=log, stderr=subprocess.STDOUT)
    (EV / "online-merge.run.json").write_text(json.dumps({"command": cmd, "exit": result.returncode, "probe_sha256": study.sha(path)}, indent=2)+"\n")
    result.check_returncode()
    study.adb("pull", study.REMOTE + "/online-merge.json", EV / "online-merge.json")


def full():
    for nt in (128, 768):
        for policy in ("native", "base", "fused"):
            cmd = [sys.executable, str(ROOT / "tests/run-fullmodel-oracle.py"), "--mode", "npu" if policy == "native" else "custom",
                   "--tokens", str(nt), "--tag", f"merge-{nt}-{policy}", "--reps", "4", "--hexagon-timing",
                   "--prompt-file", str(study.OLD / "prompts/museum.txt"), "--output-dir", str(EV)]
            if policy != "native":
                cmd += study.FLAGS + ["--plan-file", str(study.OLD / f"plans/confirm-museum-{nt}-route.json"),
                                      "--route-model", str(study.OLD / "model-frozen.json")]
            if policy == "fused":
                cmd += ["--npu-device-gather", "--npu-fused-gather"]
            subprocess.run(cmd, cwd=ROOT, check=True)


def summarize():
    report = {"device": "fd8657d6", "upstream": "00af63567a3ac073919be039db8492f229de5ffc",
              "scope": "Compatibility regression; one process/policy/length, two warmups and four samples; not a controlled old/new performance comparison",
              "indexed_cases": {}, "fullmodel": []}
    for kind in ("regression", "regression-no-nx", "regression-hvx"):
        data = json.loads((EV / f"{kind}.json").read_text())
        assert len(data["cases"]) == 20 and all(c["bitwise_equal"] for c in data["cases"])
        report["indexed_cases"][kind] = 20
    resident = json.loads((EV / "resident-fused.json").read_text())
    assert resident["reverse_revisit_bitwise_equal"]
    assert all(c["check"]["passes_4percent_per_token"] for c in resident["cases"])
    report["resident_cases"] = len(resident["cases"])
    online = json.loads((EV / "online-merge.json").read_text())
    assert all(online["canonical_contract"].values())
    assert len(online["cases"]) == 12 and all(c["bitwise_equal"] for c in online["cases"])
    assert all(all(c["mixed_layout"].values()) for c in online["cases"])
    report["canonical_mixed_exact_cases"] = 12
    report["gpu_shared_cases"] = len(online["gpu_shared"])
    for nt in (128, 768):
        data = {p: json.loads((EV / f"merge-{nt}-{p}.json").read_text()) for p in ("native", "base", "fused")}
        for policy, d in data.items():
            run = json.loads((EV / f"merge-{nt}-{policy}.run.json").read_text())
            assert run["exit"] == 0 and len(d["timing"]["samples_us"]) == 4
        assert data["base"]["all_round_logits_bitwise_equal"] and data["fused"]["all_round_logits_bitwise_equal"]
        reference = None
        for policy in ("base", "fused"):
            assert data[policy]["current_route_scheduler"]
            for sample in data[policy]["samples"]:
                assert len(sample["layers"]) == 24
                route = [(l["pairs"], l["expert_token_counts"], l["scheduling"]["owner"], l["scheduling"]["shared"], l["allocations"]) for l in sample["layers"]]
                if reference is None:
                    reference = route
                assert route == reference
        blobs = {p: (EV / f"merge-{nt}-{p}.json.logits.bin").read_bytes() for p in data}
        assert blobs["base"] == blobs["fused"]
        a, b = [struct.unpack(f"{len(blobs[p]) // 4}f", blobs[p]) for p in ("native", "fused")]
        assert len(a) == len(b) and all(math.isfinite(x) for x in a + b)
        relative_l2 = math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/sum(x*x for x in a))
        assert data["native"]["top1"] == data["base"]["top1"] == data["fused"]["top1"]
        report["fullmodel"].append({"tokens": nt, "median_ms": {p: d["timing"]["median_us"]/1000 for p,d in data.items()},
                                    "base_fused_logits_bitwise_equal": True, "route_and_allocations_equal": True,
                                    "custom_repeat_bitwise_equal": True, "top1": data["native"]["top1"],
                                    "native_fused_logits_relative_l2": relative_l2})
    (TASK / "validation-summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "deploy", "probes", "full", "summarize"])
    args = parser.parse_args()
    EV.mkdir(parents=True, exist_ok=True)
    {"build": build, "deploy": study.deploy, "probes": probes, "full": full, "summarize": summarize}[args.command]()
