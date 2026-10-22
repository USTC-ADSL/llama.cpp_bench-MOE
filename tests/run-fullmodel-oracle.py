#!/usr/bin/env python3
"""Run a fixed-prompt resident whole-model experiment on fd8657d6."""
import argparse
import gzip
import hashlib
import json
import pathlib
import shlex
import subprocess
import time
import tempfile
from gpu_clock_collector import GpuClockCollector

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "task/handoff/handoff-2026-09-23-fullmodel-oracle/evidence"
BUILD = ROOT / "build-android-opencl-hexagon"
REMOTE = "/data/local/tmp/llama-fullmodel-oracle"
MODEL = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"

def adb(*args, check=True):
    return subprocess.run(["adb", "-s", "fd8657d6", *args], capture_output=True, text=True, check=check)

def main():
    global REMOTE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=["capture", "search", "profile", "predict", "auto", "cpu", "gpu", "npu", "custom", "branch", "custom-npu"])
    p.add_argument("--cost-model", type=pathlib.Path)
    p.add_argument("--route-model", type=pathlib.Path, help="Custom: search on each current route within explicit resident union banks")
    p.add_argument("--profile-candidates", type=pathlib.Path, help="Profile: per-layer arbitrary placements and optional resident unions")
    p.add_argument("--feedback-model", type=pathlib.Path, help="Causal shadow cost updates; does not move weights or change ownership")
    p.add_argument("--runtime-state", action="store_true", help="Read-only pre-load and per-request state/CPU-time telemetry; reports observation overhead")
    p.add_argument("--runtime-state-model", type=pathlib.Path, help="Auto: sample live state and choose before loading; plan-file supplies offline routes")
    p.add_argument("--hexagon-timing", action="store_true", help="Aggregate existing DSP batch counters and host queue intervals; no per-op log output")
    p.add_argument("--npu-device-gather", action="store_true", help="Prepare split NPU activations using GET_ROWS inside the branch graph")
    p.add_argument("--npu-fused-gather", action="store_true", help="Fuse retained NPU GET_ROWS into HMX input gathering")
    p.add_argument("--hexagon-op-profile", action="store_true", help="Also enable BASIC DSP op profiling; requires --hexagon-timing")
    p.add_argument("--gpu-clock-sampling", action="store_true", help="Independent root read-only sysfs observer; inference stays shell uid")
    p.add_argument("--gpu-sample-period-us", type=int, default=10000)
    p.add_argument("--profile-layers", help="Comma-separated layer indices for optimized compact candidate profiling")
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--host-threads", type=int, choices=range(1,9), help="Outer llama CPU graph threads; expert CPU backend remains at four")
    p.add_argument("--prefill-chunk", type=int, help="Fixed-placement prefill: retain full context and process equal chunks")
    p.add_argument("--router-backend", choices=["gpu"], help="Custom HTP dense path: place router projection weights on GPU")
    p.add_argument("--router-trace", action="store_true", help="Diagnostic scheduler buffer trace; not for formal timing")
    p.add_argument("--tag", required=True)
    p.add_argument("--reference", default="-")
    p.add_argument("--reps", type=int, default=8)
    p.add_argument("--deploy", action="store_true")
    p.add_argument("--remote-dir", default=REMOTE)
    p.add_argument("--shared-weights", choices=["cpu4x8"], help="Single physical expert layout through shared-weight middleware")
    p.add_argument("--shared-reduce", choices=["scalar","neon"], help="Shared-layout single-thread reduction implementation; scalar remains default")
    p.add_argument("--shared-dispatch", choices=["async","pool"], help="Shared-layout branch dispatch: per-layer async or two persistent workers")
    p.add_argument("--shared-gather", choices=["host","htp"], help="Shared-layout input: host row copies or one HTP input plus device GET_ROWS")
    p.add_argument("--shared-fused-gather", action="store_true", help="Fuse retained CPU4x8 GET_ROWS into HMX input tile preparation")
    p.add_argument("--output-dir", type=pathlib.Path, default=OUT)
    p.add_argument("--plan-file", type=pathlib.Path)
    p.add_argument("--prompt-file", type=pathlib.Path)
    p.add_argument("--transfer-placement", action="store_true", help="Use frozen expert ownership on another prompt/length; no route replay")
    p.add_argument("--shared-activations", action="store_true")
    p.add_argument("--direct-callback-io", action="store_true")
    p.add_argument("--alternate-callback-io", action="store_true")
    p.add_argument("--activation-arena", action="store_true")
    p.add_argument("--activation-lifetimes", action="store_true", help="Reuse disjoint tensor lifetimes within each shared activation arena")
    p.add_argument("--fused-host-reduce", action="store_true")
    p.add_argument("--alternate-host-reduce", action="store_true")
    p.add_argument("--host-reduce-threads", type=int, default=1)
    p.add_argument("--alternate-reduce-threads", action="store_true", help="ABBA serial/parallel fused reduction, preserving per-token FMA order")
    p.add_argument("--overlap-gather", action="store_true", help="Each backend starts compute after its own input preparation")
    p.add_argument("--alternate-overlap-gather", action="store_true", help="ABBA barrier/independent branch preparation")
    p.add_argument("--quality-steps", type=int, default=0, help="Diagnostic continuation after timing; resident weights and KV retained")
    p.add_argument("--quality-tokens", type=pathlib.Path, help="JSON token array for equal-prefix teacher forcing")
    p.add_argument("--decode-reps", type=int, default=0, help="Repeat the quality token sequence after two warmups; time synchronized decode only")
    p.add_argument("--dense-backend", choices=["cpu"], help="Custom mode: keep non-Expert computation on CPU")
    p.add_argument("--cpu-mapped-experts", action="store_true", help="Custom CPU dense: identity CPU Expert bank borrows immutable GGUF pages")
    p.add_argument("--native-decode-graph", action="store_true", help="CPU routed nt1 graph with persistent asynchronous shared branch")
    args = p.parse_args()
    assert args.remote_dir.startswith("/data/local/tmp/") and all(c.isalnum() or c in "/_-" for c in args.remote_dir)
    REMOTE = args.remote_dir
    resolved_mode = args.mode
    if args.mode == "auto":
        assert args.plan_file and not args.transfer_placement
        opener = gzip.open if args.plan_file.suffix == ".gz" else open
        with opener(args.plan_file, "rt") as source:
            auto_plan = json.load(source)
            resolved_mode = "runtime" if args.runtime_state_model else auto_plan["execution_mode"]
        assert resolved_mode in ("custom", "npu", "runtime")
        assert not any((args.shared_activations, args.direct_callback_io, args.activation_arena,
                        args.fused_host_reduce, args.alternate_callback_io, args.alternate_host_reduce,
                        args.activation_lifetimes, args.overlap_gather, args.alternate_overlap_gather,
                        args.alternate_reduce_threads, args.host_reduce_threads != 1))
        if resolved_mode == "custom":
            args.shared_activations = args.direct_callback_io = args.activation_arena = args.fused_host_reduce = True
            if auto_plan.get("model_type") == "branch_critical_path":
                args.activation_lifetimes = auto_plan["execution_policy"]["activation_lifetimes"]
                args.overlap_gather = auto_plan["execution_policy"]["overlap_gather"]
    assert args.tag.replace("-", "").replace("_", "").isalnum()
    assert adb("get-state").stdout.strip() == "device"
    assert not adb("shell", "pidof fullmodel-oracle", check=False).stdout.strip()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assert not any((output / (args.tag + suffix)).exists() for suffix in (".json", ".json.gz", ".run.json"))
    adb("shell", f"mkdir -p {REMOTE}")
    if args.deploy:
        names = ["fullmodel-oracle", "libllama.so", "libggml.so", "libggml-base.so", "libggml-cpu.so", "libggml-opencl.so", "libggml-hexagon.so"]
        paths = [BUILD / "bin" / name for name in names]
        paths.append(BUILD / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
        paths.append(BUILD / "ggml/src/ggml-hexagon/ship/libc++_shared.so")
        for path in paths:
            adb("push", str(path), f"{REMOTE}/{path.name}")
            assert adb("shell", f"sha256sum {REMOTE}/{path.name}").stdout.split()[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    prompt_path = args.prompt_file or (ROOT / "tests/expert-routing-prompt.txt" if args.deploy else None)
    if prompt_path:
        adb("push", str(prompt_path), f"{REMOTE}/prompt.txt")
        assert adb("shell", f"sha256sum {REMOTE}/prompt.txt").stdout.split()[0] == hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    if args.plan_file:
        opener = gzip.open if args.plan_file.suffix == ".gz" else open
        with opener(args.plan_file, "rb") as source:
            plan = source.read()
        with tempfile.NamedTemporaryFile() as temporary:
            temporary.write(plan)
            temporary.flush()
            adb("push", temporary.name, f"{REMOTE}/offline-plan.json")
        assert adb("shell", f"sha256sum {REMOTE}/offline-plan.json").stdout.split()[0] == hashlib.sha256(plan).hexdigest()
        args.reference = "offline-plan"
    record = {"device": adb("shell", "getprop ro.product.model").stdout,
              "model": adb("shell", f"stat -c '%n %s %Y' {MODEL}").stdout,
              "before_thermal": adb("shell", "dumpsys thermalservice").stdout,
              "hashes": adb("shell", f"sha256sum {REMOTE}/*.so {REMOTE}/fullmodel-oracle {REMOTE}/prompt.txt").stdout,
              "memory_before": adb("shell", "cat /proc/meminfo").stdout}
    record["prompt_source"] = str(prompt_path.resolve()) if prompt_path else "existing remote prompt"
    record["resolved_mode"] = resolved_mode
    if args.reference != "-":
        assert args.reference.replace("-", "").replace("_", "").isalnum()
        record["reference_hash"] = adb("shell", f"sha256sum {REMOTE}/{args.reference}.json").stdout
    if args.plan_file:
        record["plan_source"] = str(args.plan_file.resolve())
    env = ["env", f"LD_LIBRARY_PATH={REMOTE}", f"ADSP_LIBRARY_PATH={REMOTE}", "GGML_HEXAGON_DEVICES=1", "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3", f"GGML_HEXAGON_PROFILE={int(args.hexagon_op_profile)}"]
    if args.host_threads is not None:
        env.append(f"MOE_HOST_THREADS={args.host_threads}")
    if args.router_backend:
        assert args.mode == "custom" and not args.dense_backend
        env.append("MOE_ROUTER_BACKEND=" + args.router_backend)
    if args.router_trace:
        assert args.mode == "custom"
        env.append("MOE_ROUTER_TRACE=1")
    if args.shared_weights:
        assert args.mode in ("custom", "profile")
        env.append("MOE_SHARED_WEIGHTS=" + args.shared_weights)
    if args.shared_reduce:
        assert args.shared_weights
        env.append("MOE_SHARED_REDUCE=" + args.shared_reduce)
    if args.shared_dispatch:
        assert args.shared_weights
        env.append("MOE_SHARED_DISPATCH=" + args.shared_dispatch)
    if args.shared_gather:
        assert args.shared_weights
        env.append("MOE_SHARED_GATHER=" + args.shared_gather)
    if args.shared_fused_gather:
        assert args.shared_weights and args.shared_gather == "htp"
        env.append("GGML_HEXAGON_FUSE_CPU4X8_GATHER=1")
    if args.prefill_chunk is not None:
        assert args.mode == "custom" and args.transfer_placement
        assert 0 < args.prefill_chunk <= args.tokens and args.tokens % args.prefill_chunk == 0
        env.append(f"MOE_PREFILL_CHUNK={args.prefill_chunk}")
    assert not args.hexagon_op_profile or args.hexagon_timing
    if args.hexagon_timing:
        env.append("GGML_HEXAGON_TIMING=1")
    if args.gpu_clock_sampling:
        assert args.runtime_state and args.mode in ("npu", "custom", "auto")
    if args.runtime_state:
        assert args.mode not in ("search","profile")
        env.append("MOE_RUNTIME_STATE=1")
    if args.runtime_state_model:
        assert args.mode == "auto" and not args.cost_model and not args.feedback_model and not args.quality_steps
        adb("push", str(args.runtime_state_model), f"{REMOTE}/runtime-state-model.json")
        digest = hashlib.sha256(args.runtime_state_model.read_bytes()).hexdigest()
        assert adb("shell", f"sha256sum {REMOTE}/runtime-state-model.json").stdout.split()[0] == digest
        record["runtime_state_model_sha256"] = digest
        env.append(f"MOE_RUNTIME_STATE_MODEL={REMOTE}/runtime-state-model.json")
    assert 0 <= args.quality_steps <= 64
    assert 0 <= args.decode_reps <= 32 and (not args.decode_reps or args.quality_steps)
    if args.decode_reps:
        env.append(f"MOE_DECODE_REPS={args.decode_reps}")
    if args.dense_backend:
        assert args.mode == "custom"
        env.append(f"MOE_DENSE_BACKEND={args.dense_backend}")
    if args.cpu_mapped_experts:
        assert args.dense_backend == "cpu"
        env.append("MOE_CPU_MAPPED_EXPERTS=1")
    if args.native_decode_graph:
        assert args.cpu_mapped_experts and args.activation_lifetimes and args.shared_activations
        env.append("MOE_NATIVE_DECODE_GRAPH=1")
    if args.quality_steps:
        assert resolved_mode in ("cpu", "gpu", "npu", "custom", "custom-npu", "branch")
        env.append(f"MOE_QUALITY_STEPS={args.quality_steps}")
    if args.quality_tokens:
        data = args.quality_tokens.read_bytes()
        assert args.quality_steps and len(json.loads(data)) == args.quality_steps
        adb("push", str(args.quality_tokens), f"{REMOTE}/quality-tokens.json")
        record["quality_tokens_sha256"] = hashlib.sha256(data).hexdigest()
        assert adb("shell", f"sha256sum {REMOTE}/quality-tokens.json").stdout.split()[0] == record["quality_tokens_sha256"]
        env.append(f"MOE_QUALITY_TOKENS={REMOTE}/quality-tokens.json")
    if args.cost_model:
        assert args.mode == "predict"
        adb("push", str(args.cost_model), f"{REMOTE}/cost-model.json")
        digest = hashlib.sha256(args.cost_model.read_bytes()).hexdigest()
        assert adb("shell", f"sha256sum {REMOTE}/cost-model.json").stdout.split()[0] == digest
        record["cost_model_sha256"] = digest
        env.append(f"MOE_PLACEMENT_MODEL={REMOTE}/cost-model.json")
    for field, variable, filename, required in (
            ("route_model", "MOE_ROUTE_MODEL", "route-model.json", "custom"),
            ("profile_candidates", "MOE_PROFILE_CANDIDATES", "profile-candidates.json", "profile")):
        path = getattr(args, field)
        if path:
            assert args.mode == required
            data = path.read_bytes()
            adb("push", str(path), f"{REMOTE}/{filename}")
            digest = hashlib.sha256(data).hexdigest()
            assert adb("shell", f"sha256sum {REMOTE}/{filename}").stdout.split()[0] == digest
            record[field + "_sha256"] = digest
            record[field + "_source"] = str(path.resolve())
            env.append(f"{variable}={REMOTE}/{filename}")
    if args.feedback_model:
        assert resolved_mode == "custom" and not args.transfer_placement
        adb("push", str(args.feedback_model), f"{REMOTE}/feedback-model.json")
        digest = hashlib.sha256(args.feedback_model.read_bytes()).hexdigest()
        assert adb("shell", f"sha256sum {REMOTE}/feedback-model.json").stdout.split()[0] == digest
        record["feedback_model_sha256"] = digest
        env.append(f"MOE_FEEDBACK_MODEL={REMOTE}/feedback-model.json")
    if args.profile_layers:
        assert args.mode == "profile"
        layers = [int(i) for i in args.profile_layers.split(",")]
        assert layers and all(0 <= i < 24 for i in layers)
        env.append("MOE_PROFILE_LAYERS=" + ",".join(map(str, layers)))
    if args.transfer_placement:
        assert resolved_mode in ("custom", "branch", "custom-npu")
        env.append("MOE_TRANSFER_PLACEMENT=1")
    if args.shared_activations:
        assert resolved_mode in ("custom", "branch", "custom-npu")
        env.append("MOE_SHARED_ACTIVATIONS=1")
    if args.npu_device_gather:
        assert resolved_mode == "custom" and args.shared_activations and not args.quality_steps
        env.append("MOE_NPU_DEVICE_GATHER=1")
    if args.npu_fused_gather:
        assert args.npu_device_gather
        env.append("GGML_HEXAGON_FUSE_MOE_GATHER=1")
    if args.direct_callback_io:
        assert resolved_mode in ("custom", "branch", "custom-npu")
        env.append("MOE_DIRECT_CALLBACK_IO=1")
    if args.alternate_callback_io:
        assert args.shared_activations and not args.direct_callback_io and args.reps % 4 == 0
        env.append("MOE_ALTERNATE_CALLBACK_IO=1")
    if args.activation_arena:
        assert args.shared_activations
        env.append("MOE_ACTIVATION_ARENA=1")
    if args.activation_lifetimes:
        assert args.activation_arena and resolved_mode == "custom"
        env.append("MOE_ACTIVATION_LIFETIMES=1")
    if args.fused_host_reduce:
        assert resolved_mode in ("custom", "branch", "custom-npu")
        env.append("MOE_FUSED_HOST_REDUCE=1")
    if args.alternate_host_reduce:
        assert not args.fused_host_reduce and not args.alternate_callback_io and args.reps % 4 == 0
        env.append("MOE_ALTERNATE_HOST_REDUCE=1")
    assert 1 <= args.host_reduce_threads <= 8
    if args.host_reduce_threads > 1:
        assert resolved_mode in ("custom", "branch", "custom-npu") and args.fused_host_reduce
        env.append(f"MOE_REDUCE_THREADS={args.host_reduce_threads}")
    if args.alternate_reduce_threads:
        assert args.host_reduce_threads > 1 and args.reps % 4 == 0
        assert not any((args.alternate_host_reduce, args.alternate_callback_io, args.quality_steps))
        env.append("MOE_ALTERNATE_REDUCE_THREADS=1")
    if args.overlap_gather or args.alternate_overlap_gather:
        assert resolved_mode in ("custom", "branch", "custom-npu") and args.shared_activations
        if args.overlap_gather:
            env.append("MOE_OVERLAP_GATHER=1")
        if args.alternate_overlap_gather:
            assert args.reps % 4 == 0 and not any((args.overlap_gather, args.alternate_reduce_threads, args.alternate_host_reduce, args.alternate_callback_io, args.quality_steps))
            env.append("MOE_ALTERNATE_OVERLAP_GATHER=1")
    reference = "-" if args.reference == "-" else f"{REMOTE}/{args.reference}.json"
    argv = [f"{REMOTE}/fullmodel-oracle", MODEL, f"{REMOTE}/prompt.txt", str(args.tokens), f"{REMOTE}/{args.tag}.json", args.mode, str(args.reps), reference]
    command = f"cd {shlex.quote(REMOTE)} && exec " + shlex.join(env + argv)
    record["command"] = command
    record["host_start_unix_ns"] = time.time_ns()
    manifest = output / f"{args.tag}.run.json"
    manifest.write_text(json.dumps(record, indent=2))
    collector = None
    proc = None
    try:
        if args.gpu_clock_sampling:
            collector = GpuClockCollector(adb, REMOTE, BUILD / "bin/gpu-clock-sampler", output, args.tag, args.gpu_sample_period_us)
            collector.start()
        with (output / f"{args.tag}.log").open("w") as log:
            proc = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", command], stdout=log, stderr=subprocess.STDOUT)
            start = time.monotonic()
            while proc.poll() is None:
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print(f"{args.tag}: {time.monotonic()-start:.0f}s", flush=True)
    except BaseException as error:
        record["runner_error"] = repr(error)
        raise
    finally:
        if proc is not None:
            record["exit"] = proc.poll()
        try:
            if collector is not None:
                try:
                    collector.stop()
                except Exception as error:
                    record["collector_error"] = repr(error)
                    if "runner_error" not in record:
                        raise
        finally:
            if collector is not None:
                record["gpu_collector"] = collector.record
            manifest.write_text(json.dumps(record, indent=2))
    record["exit"] = proc.returncode
    record["host_end_unix_ns"] = time.time_ns()
    record["after_thermal"] = adb("shell", "dumpsys thermalservice").stdout
    manifest.write_text(json.dumps(record, indent=2))
    assert proc.returncode == 0, f"failed; inspect {args.tag}.log"
    filenames = [f"{args.tag}.json"]
    if args.runtime_state_model:
        filenames.append(f"{args.tag}.json.plan.json")
    if args.mode == "capture":
        filenames += [f"{args.tag}.json.layer{i}.bin" for i in range(24)]
    if args.mode not in ("search", "profile", "predict"):
        filenames += [f"{args.tag}.json.logits.bin"]
    if args.quality_steps:
        filenames += [f"{args.tag}.json.quality.bin"]
    if args.decode_reps:
        filenames += [f"{args.tag}.json.decode.bin"]
    for name in filenames:
        adb("pull", f"{REMOTE}/{name}", str(output / name))
        assert adb("shell", f"sha256sum {REMOTE}/{name}").stdout.split()[0] == hashlib.sha256((output / name).read_bytes()).hexdigest()
    report = json.loads((output / f"{args.tag}.json").read_text())
    if args.runtime_state_model:
        record["resolved_mode"] = report["mode"]
        manifest.write_text(json.dumps(record, indent=2))
    print(report.get("timing", {"layers": len(report.get("layers", []))}), flush=True)

if __name__ == "__main__":
    main()
