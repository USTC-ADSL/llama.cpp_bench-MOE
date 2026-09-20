#!/usr/bin/env python3
"""Deploy and retain fixed-route GPU/HTP benchmark sessions on the V79 device."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

SERIAL = "fd8657d6"


class DeviceUnavailable(RuntimeError):
    pass


class Runner:
    def __init__(self, output):
        self.output = output
        self.recovered = False
        self.commands = (output / "command.txt").open("x")

    def command(self, argv, stdout=None, stderr=None, check=True):
        self.commands.write(shlex.join(map(str, argv)) + "\n")
        self.commands.flush()
        return subprocess.run(argv, stdout=stdout if stdout is not None else subprocess.PIPE,
                              stderr=stderr if stderr is not None else subprocess.PIPE,
                              text=True, check=check)

    def online(self):
        result = self.command(["adb", "devices", "-l"])
        return any(len(fields := line.split()) >= 2 and fields[:2] == [SERIAL, "device"]
                   for line in result.stdout.splitlines())

    def ensure_online(self):
        if self.online():
            return
        if self.recovered:
            raise DeviceUnavailable("V79 device unavailable; recovery already used. Stop this conversation.")
        self.recovered = True
        self.command(["adb", "-s", SERIAL, "reconnect"], check=False)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            time.sleep(1)
            if self.online():
                return
        raise DeviceUnavailable("V79 still unavailable after one recovery and 20 seconds. Stop this conversation.")

    def adb(self, *args, **kwargs):
        self.ensure_online()
        result = self.command(["adb", "-s", SERIAL, *map(str, args)], check=False, **kwargs)
        if result.returncode:
            self.ensure_online()
            raise RuntimeError(f"adb command failed ({result.returncode}); device online: {args}")
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-dir", type=Path, default=Path("build-android-pd-final"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--remote-dir", default="/data/local/tmp/shared-expert-routed")
    p.add_argument("--remote-pack-dir", default="/data/local/tmp/shared-expert-resident-20260915")
    p.add_argument("--sessions", type=int, default=3)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--tokens", default="all", help="all (1,3,32) or comma-separated positive counts")
    p.add_argument("--breakdown", choices=["0", "1"], default="1", help="host stage timing")
    p.add_argument("--profile", choices=["0", "1"], default="0", help="HTP device profiler")
    args = p.parse_args()
    if args.sessions < 1 or args.repeat < 1 or args.warmup < 0:
        p.error("sessions/repeat must be positive and warmup nonnegative")
    if args.tokens != "all":
        counts = args.tokens.split(",")
        if any(not v.isascii() or not v.isdecimal() or not 1 <= int(v) <= 100000 for v in counts):
            p.error("tokens must be all or comma-separated integers in 1..100000")
        if len(set(map(int, counts))) != len(counts):
            p.error("duplicate token count")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    r = Runner(args.output_dir)
    program = "routed-expert-profile" if args.profile == "1" else "routed-expert-bench"
    source_root = Path(__file__).resolve().parent.parent
    for pattern in ("runtime/include/*.h", "runtime/src/*.cpp", "experiments/*.cpp", "experiments/support/*",
                    "experiments/*.py", "experiments/*.md", "CMakeLists.txt", "runtime/CMakeLists.txt",
                    "experiments/CMakeLists.txt", "tests/CMakeLists.txt", "tests/*routed*"):
        for source in source_root.glob(pattern):
            if not source.is_file():
                continue
            target = args.output_dir / "sources" / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    files = [args.build_dir / "bin" / name for name in
             [program, "libggml.so", "libggml-base.so", "libggml-cpu.so",
              "libggml-opencl.so", "libggml-hexagon.so"]]
    files.append(args.build_dir / "ggml/src/ggml-hexagon/libggml-htp-v79.so")
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    (args.output_dir / "binaries.json").write_text(json.dumps(
        {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}, indent=2) + "\n")
    for name, argv in [("git-status.txt", ["git", "status", "--short"]),
                       ("git-diff.patch", ["git", "diff", "HEAD"]),
                       ("commit.txt", ["git", "rev-parse", "HEAD"])]:
        with (args.output_dir / name).open("w") as out:
            r.command(argv, stdout=out)
    r.adb("shell", "mkdir -p " + shlex.quote(args.remote_dir))
    r.adb("push", "-Z", *files, args.remote_dir + "/")
    r.adb("shell", "chmod +x " + shlex.quote(args.remote_dir + "/" + program))
    info = r.adb("shell", "getprop ro.product.model; getprop ro.soc.model; uname -a; cat /proc/meminfo").stdout
    (args.output_dir / "device-info.txt").write_text(info)
    environment = {"LD_LIBRARY_PATH": args.remote_dir, "ADSP_LIBRARY_PATH": args.remote_dir,
                   "GGML_HEXAGON_EXPERIMENTAL": "1", "GGML_HEXAGON_ARCH": "79", "GGML_HEXAGON_HOSTBUF": "1",
                   "GGML_HEXAGON_NDEV": "1", "GGML_HEXAGON_NHVX": "0", "GGML_HEXAGON_USE_HMX": "1",
                   "GGML_HEXAGON_PROFILE": args.profile}
    for session in range(1, args.sessions + 1):
        local = args.output_dir / f"session-{session}"
        local.mkdir()
        remote_output = args.remote_dir + "/" + args.output_dir.name + f"-session-{session}"
        argv = ["./" + program, "--mode", "all", "--tokens", args.tokens,
                "--breakdown", args.breakdown,
                "--warmup", str(args.warmup), "--repeat", str(args.repeat),
                "--session", str(session), "--output-dir", remote_output]
        for layer in range(4):
            argv += [f"--layer{layer}-pack", f"{args.remote_pack_dir}/layer-{layer}.pack"]
        command = "cd " + shlex.quote(args.remote_dir) + " && " + " ".join(
            key + "=" + shlex.quote(value) for key, value in environment.items()) + " " + shlex.join(argv)
        print(f"Session {session}: {local}", flush=True)
        state_command = "cat /proc/meminfo; for z in /sys/class/thermal/thermal_zone*; do cat $z/type $z/temp 2>/dev/null; done"
        (local / "device-state-before.txt").write_text(r.adb("shell", state_command).stdout)
        try:
            with (local / "stdout.log").open("w") as out, (local / "stderr.log").open("w") as err:
                r.adb("shell", command, stdout=out, stderr=err)
        except DeviceUnavailable:
            raise
        except RuntimeError:
            # Do not issue further commands if the single device recovery failed.
            if r.online():
                r.command(["adb", "-s", SERIAL, "pull", remote_output + "/raw.jsonl", str(local / "raw.jsonl")], check=False)
                with (local / "logcat.txt").open("w") as log:
                    r.command(["adb", "-s", SERIAL, "logcat", "-d", "-t", "500"], stdout=log, check=False)
            raise
        r.adb("pull", remote_output + "/raw.jsonl", local / "raw.jsonl")
        (local / "device-state-after.txt").write_text(r.adb("shell", state_command).stdout)
    print(args.output_dir, flush=True)


if __name__ == "__main__":
    main()
