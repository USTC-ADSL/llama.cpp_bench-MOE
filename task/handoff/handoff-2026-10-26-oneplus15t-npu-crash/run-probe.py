#!/usr/bin/env python3
"""Record a root NPU probe on the explicitly selected OnePlus 15T."""
import argparse
import datetime
import json
import pathlib
import shlex
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--runtime", default="/data/local/tmp/llama-moe-bench")
parser.add_argument("--mm", default="2")
parser.add_argument("--nhvx")
parser.add_argument("--prompt", default="128")
parser.add_argument("--gen", default="0")
parser.add_argument("--repetitions", default="1")
parser.add_argument("--env", action="append", default=[])
args = parser.parse_args()
serial = "3B162U00GZY00000"
out = pathlib.Path(__file__).resolve().parent / "evidence" / args.tag
out.mkdir(parents=True, exist_ok=False)
env = ["LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64",
       "ADSP_LIBRARY_PATH=" + args.runtime,
       "GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1,HTP0:2",
       "GGML_HEXAGON_NHMX=1", "GGML_HEXAGON_MM_SELECT=" + args.mm]
if args.nhvx:
    env.append("GGML_HEXAGON_NHVX=" + args.nhvx)
env += args.env
bench = ["./llama-bench", "-m", "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf",
         "-dev", "HTP0:0/HTP0:1/HTP0:2", "-ngl", "99", "-ts", "1/1/1", "-t", "4",
         "-p", args.prompt, "-n", args.gen, "-b", "512", "-ub", "512", "-fa", "on",
         "-lm", "mmap", "-lzm", "off", "-r", args.repetitions, "-o", "json", "-v", "--progress"]
inner = "cd " + shlex.quote(args.runtime) + " && " + shlex.join(env + bench)
remote = "su -c " + shlex.quote("timeout 900 sh -c " + shlex.quote(inner))
command = ["adb", "-s", serial, "shell", remote]
record = {"serial": serial, "command": command,
          "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
print(out, flush=True)
started = time.monotonic()
with (out / "npu.stdout").open("w") as stdout, (out / "npu.log").open("w") as stderr:
    result = subprocess.run(command, stdout=stdout, stderr=stderr)
record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started)
(out / "run.json").write_text(json.dumps(record, indent=2) + "\n")
logcat = subprocess.run(["adb", "-s", serial, "logcat", "-d", "-t", "1000", "-s",
                         "llama-bench:V", "adsprpc:V", "fastrpc:V"], capture_output=True, text=True)
(out / "fastrpc.log").write_text(logcat.stdout + logcat.stderr)
print(json.dumps(record, indent=2), flush=True)
raise SystemExit(result.returncode)
