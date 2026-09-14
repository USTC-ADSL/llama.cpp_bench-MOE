#!/usr/bin/env python3
"""Run a diagnostic routing/logit probe with the existing Android inference libs."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import re
import shlex
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("capacity_runner", HERE / "run.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tag", required=True)
parser.add_argument("--prompt", type=int, choices=(128, 512), required=True)
parser.add_argument("--full-weight", action="store_true")
parser.add_argument("--weight-gib", type=int, choices=(4, 5, 6), default=6)
parser.add_argument("--trace", action="store_true")
parser.add_argument("--deploy", action="store_true")
parser.add_argument("--cache-check", action="store_true")
args = parser.parse_args()
assert re.fullmatch(r"gap-[a-z0-9-]+", args.tag)
assert r.base.adb("get-state").stdout.strip() == "device"
assert not r.base.shell("pidof llama-bench gap-probe", check=False).stdout.strip()
local = r.OUT / args.tag
assert not local.with_suffix(".run.json").exists()
binary = pathlib.Path("/tmp/llama-gap-probe")
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
if args.deploy:
    r.base.adb("push", str(binary), f"{r.REMOTE}/gap-probe")
    r.base.shell(f"chmod 755 {r.REMOTE}/gap-probe")
assert r.base.shell(f"sha256sum {r.REMOTE}/gap-probe").stdout.split()[0] == digest
config_path = "none"
if not args.full_weight:
    prefix = r.capacity_prefix(args.weight_gib)
    config = json.loads((r.OUT / f"{prefix}gpu.config.json").read_text())
    config["report"] = f"{r.REMOTE}/{args.tag}.weights.json"
    r.save(local.with_suffix(".config.json"), config)
    config_path = f"{r.REMOTE}/{args.tag}.config.json"
    r.base.adb("push", str(local.with_suffix(".config.json")), config_path)
output = f"{r.REMOTE}/{args.tag}.json"
env = [
    "env", f"LD_LIBRARY_PATH={r.REMOTE}:/vendor/lib64", f"ADSP_LIBRARY_PATH={r.REMOTE}",
    "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=3",
    f"GGML_OPENCL_KERNEL_CACHE_DIR={r.REMOTE}/cl-cache"]
if args.cache_check:
    env += ["LLAMA_WEIGHT_CACHE_CHECK=blk.16.ffn_up_exps"]
command = f"cd {r.REMOTE} && exec " + shlex.join(env + [
    f"{r.REMOTE}/gap-probe", r.base.MODEL, config_path, str(args.prompt), output,
    "1" if args.trace else "0"])
record = {"command": command, "probe_sha256": digest, "before": r.snapshot(),
          "library_hashes": r.base.shell(f"sha256sum {r.REMOTE}/libllama.so {r.REMOTE}/libggml-opencl.so").stdout}
r.save(local.with_suffix(".run.json"), record)
print("START", args.tag, flush=True)
started = time.monotonic()
with local.with_suffix(".stdout").open("w") as out, local.with_suffix(".log").open("w") as err:
    proc = subprocess.Popen(["adb", "-s", r.base.SERIAL, "shell", command], stdout=out, stderr=err)
    try:
        proc.wait(timeout=600)
    finally:
        if proc.poll() is None:
            r.base.shell("pkill -TERM -x gap-probe", check=False)
            proc.wait(timeout=30)
        record.update(exit=proc.returncode, seconds=time.monotonic() - started, after=r.snapshot())
        r.save(local.with_suffix(".run.json"), record)
assert proc.returncode == 0, f"Probe failed: {args.tag}"
r.base.adb("pull", output, str(local.with_suffix(".json")))
if not args.full_weight:
    r.base.adb("pull", f"{r.REMOTE}/{args.tag}.weights.json", str(local.with_suffix(".weights.json")))
data = json.loads(local.with_suffix(".json").read_text())
for item in data["results"]:
    print(args.tag, item["stage"], "nonfinite", item["nonfinite_logits"],
          "argmax", item["argmax"], "tok/s", args.prompt * 1e6 / item["microseconds"], flush=True)
