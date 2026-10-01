#!/usr/bin/env python3
"""Run serialized A/F measurements and retain exact commands, samples and device state."""
import argparse
import datetime
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default="fd8657d6")
    ap.add_argument("--directory", default="/data/local/tmp/dense-af-bench")
    ap.add_argument("--model", default="/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf")
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPUOpenCL", "HTP0"])
    ap.add_argument("--prompts", nargs="+", type=int, default=[64, 256])
    ap.add_argument("--past", type=int, default=0)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--stage", choices=["both", "attention", "ffn"], default="both")
    ap.add_argument("--flash", type=int, choices=[0, 1], default=0)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--round", default="1")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--gpu-dense-dp4a", type=int, choices=[0, 1], default=1,
                    help="Adreno Q4_0 dense matmul path; 1 avoids large half-accumulation error on early FFNs")
    ap.add_argument("--cooldown", type=float, default=10)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    adb = ["adb", "-s", args.serial]
    state = subprocess.check_output(adb + ["get-state"], text=True).strip()
    if state != "device":
        raise SystemExit(f"{args.serial}: {state}")
    online = subprocess.check_output(adb + ["shell", "cat /sys/devices/system/cpu/online"], text=True).strip()
    if online != "0-7":
        raise SystemExit(f"expected all eight CPUs online (0-7), found {online}")

    def snapshot():
        return subprocess.check_output(adb + ["shell", "getprop ro.soc.model; "
            "cat /sys/devices/system/cpu/online; "
            "taskset ff sh -c 'grep Cpus_allowed_list /proc/self/status'; "
            "dumpsys battery | grep -E 'temperature|level|status'; "
            "for p in /sys/devices/system/cpu/cpufreq/policy*; do "
            "echo $p; cat $p/scaling_governor $p/scaling_cur_freq $p/scaling_max_freq; done"],
            text=True, stderr=subprocess.STDOUT)

    for prompt in args.prompts:
        for device in args.devices:
            tag = f"r{args.round}-{device}-p{prompt}-fa{args.flash}-{args.stage}"
            log = args.output / (tag + ".log")
            info = args.output / (tag + ".json")
            if log.exists() or info.exists():
                raise SystemExit(f"refusing to overwrite {tag}")
            argv = ["timeout", "600", "taskset", "ff", "./dense-af-bench", "-m", args.model,
                    "--device", device, "--prompt", str(prompt), "--layers", args.layers,
                    "--past", str(args.past),
                    "--stage", args.stage, "--flash", str(args.flash), "--threads", "8",
                    "--reps", str(args.reps), "--warmup", str(args.warmup)]
            if args.reference:
                argv += ["--reference", "./reference"]
                if device == "CPU":
                    argv += ["--write-reference"]
            command = (f"cd {shlex.quote(args.directory)} && "
                       "LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 "
                       "GGML_HEXAGON_PROFILE=0 "
                       f"GGML_OPENCL_Q4_0_DENSE_DP4A={args.gpu_dense_dp4a} " + shlex.join(argv))
            time.sleep(args.cooldown)
            record = dict(serial=args.serial, command=command, before=snapshot(),
                          started=datetime.datetime.now(datetime.timezone.utc).isoformat())
            print(f"START {tag}", flush=True)
            started = time.monotonic()
            with log.open("w") as output:
                run = subprocess.run(adb + ["shell", command], stdout=output, stderr=subprocess.STDOUT)
            record.update(returncode=run.returncode, elapsed_seconds=time.monotonic()-started,
                          after=snapshot())
            rows = [json.loads(line[7:]) for line in log.read_text().splitlines() if line.startswith("RESULT ")]
            record["results"] = rows
            info.write_text(json.dumps(record, indent=2) + "\n")
            for block in ("attention", "ffn"):
                values = [statistics.median(r["compute_ms"]) for r in rows if r["block"] == block]
                errors = [r["relative_rms"] for r in rows if r["block"] == block]
                if values:
                    print(f"  {block}: {len(values)} layers, mean layer median={statistics.mean(values):.3f} ms, "
                          f"max relative RMS={max(errors):.5f}", flush=True)
            print(f"END {tag}: rc={run.returncode}, {record['elapsed_seconds']:.1f}s", flush=True)
            if run.returncode:
                print("\n".join(log.read_text().splitlines()[-8:]), flush=True)
                raise SystemExit(run.returncode)


if __name__ == "__main__":
    main()
