"""Owned, bounded-lifetime root observer; never changes inference privileges."""
import hashlib
import json
import math
import shlex
import subprocess
import time


class GpuClockCollector:
    def __init__(self, adb, remote, binary, output, tag, period_us):
        self.adb, self.remote, self.output, self.tag = adb, remote, output, tag
        self.trace = f"{remote}/{tag}.gpu-trace.jsonl"
        self.pidfile, self.stopfile = self.trace + ".pid", self.trace + ".stop"
        self.process = self.log = None
        self.pid = None
        assert 1000 <= period_us <= 1000000
        self.adb("push", str(binary), f"{remote}/gpu-clock-sampler")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        assert self.adb("shell", f"sha256sum {remote}/gpu-clock-sampler").stdout.split()[0] == digest
        for path in (self.trace, self.pidfile, self.stopfile):
            assert self.adb("shell", f"test ! -e {shlex.quote(path)}", check=False).returncode == 0
        identity = self.adb("shell", "su -c id").stdout
        assert "uid=0(root)" in identity
        inner = "exec " + shlex.join([f"{remote}/gpu-clock-sampler", self.trace, self.pidfile, self.stopfile, str(period_us), "120"])
        self.command = "exec su -c " + shlex.quote(inner)
        self.record = {"command": self.command, "binary_sha256": digest, "period_us": period_us, "identity": identity}

    def start(self):
        self.log = (self.output / f"{self.tag}.gpu-collector.log").open("w")
        self.process = subprocess.Popen(["adb", "-s", "fd8657d6", "shell", self.command], stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            ready = self.adb("shell", f"cat {self.pidfile}", check=False)
            if ready.returncode == 0:
                self.pid = int(ready.stdout.strip())
                assert self.pid > 1 and self.process.poll() is None
                self.record["pid"] = self.pid
                return
            assert self.process.poll() is None, "GPU observer exited before ready"
            time.sleep(.1)
        raise RuntimeError("GPU observer failed to become ready")

    def stop(self):
        if self.process is None:
            return
        try:
            # A failed stop-file write must not bypass the owned-PID fallback.
            try:
                self.adb("shell", f"touch {self.stopfile}")
            except Exception as error:
                self.record["stop_file_error"] = repr(error)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if self.pid is None:
                        self.pid = int(self.adb("shell", f"cat {self.pidfile}").stdout.strip())
                    cmdline = self.adb("shell", "su -c " + shlex.quote(f"cat /proc/{self.pid}/cmdline")).stdout
                    assert self.pid > 1 and f"{self.remote}/gpu-clock-sampler" in cmdline and self.trace in cmdline, "observer PID ownership changed"
                    self.adb("shell", "su -c " + shlex.quote(f"kill -TERM {self.pid}"))
                    self.process.wait(timeout=5)
                except Exception as error:
                    self.record["termination_error"] = repr(error)
                    self.record["remote_lifetime_cap_seconds"] = 120
                    # Close only our adb client if the device is unreachable.
                    # The device executable independently enforces its lifetime.
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill(); self.process.wait(timeout=2)
                    raise RuntimeError("could not confirm GPU observer termination; device lifetime remains bounded") from error
            self.record["exit"] = self.process.returncode
            local = self.output / f"{self.tag}.gpu-trace.jsonl"
            self.adb("pull", self.trace, str(local))
            digest = hashlib.sha256(local.read_bytes()).hexdigest()
            assert self.adb("shell", f"sha256sum {self.trace}").stdout.split()[0] == digest
            rows = [json.loads(line) for line in local.read_text().splitlines()]
            assert rows[0]["type"] == "header" and rows[-1]["type"] == "footer"
            assert rows[0]["pid"] == self.pid and rows[0]["uid"] == rows[0]["euid"] == 0
            assert rows[0]["period_us"] == self.record["period_us"]
            assert rows[-1]["samples"] == len(rows) - 2
            previous = rows[0]["start_monotonic_us"]
            for row in rows[1:-1]:
                assert row["type"] == "sample" and previous <= row["begin_us"] <= row["end_us"]
                assert all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in ("begin_us", "end_us", "gpu_hz", "busy_percent"))
                assert row["gpu_hz"] > 0 and 0 <= row["busy_percent"] <= 100
                previous = row["end_us"]
            assert previous <= rows[-1]["end_monotonic_us"] and rows[-1]["read_errors"] == 0
            self.record.update(trace_sha256=digest, header=rows[0], footer=rows[-1])
            assert self.process.returncode == 0 and not rows[-1]["timed_out"], "GPU observer failed or reached lifetime limit"
            self.adb("shell", f"rm {self.pidfile} {self.stopfile}")
        except Exception as error:
            self.record["stop_or_validation_error"] = repr(error)
            # Retain trace and ownership files as evidence on failure.
            raise
        finally:
            if self.log:
                self.log.close()
