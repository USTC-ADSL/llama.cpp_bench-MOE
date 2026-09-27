#!/usr/bin/env python3
"""Failure-path ownership/validation tests; never contact an Android device."""
import hashlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
from gpu_clock_collector import GpuClockCollector


class CollectorTests(unittest.TestCase):
    def make(self, directory, *, touch_fails=False, foreign_pid=False, bad_sample=False):
        c=GpuClockCollector.__new__(GpuClockCollector)
        c.remote="/owned";c.tag="test";c.output=pathlib.Path(directory);c.trace="/owned/test.gpu-trace.jsonl"
        c.pidfile=c.trace+".pid";c.stopfile=c.trace+".stop";c.pid=42;c.log=io.StringIO()
        c.record={"period_us":10000};c.process=Mock();c.process.returncode=0
        c.process.wait.side_effect=[subprocess.TimeoutExpired("owned-adb",5),0] if touch_fails or foreign_pid else [0]
        rows=[{"type":"header","pid":42,"uid":0,"euid":0,"period_us":10000,"start_monotonic_us":1},
              {"type":"sample","begin_us":2,"end_us":3,"gpu_hz":None if bad_sample else 222000000,"busy_percent":0},
              {"type":"footer","samples":1,"end_monotonic_us":4,"read_errors":int(bad_sample),"timed_out":False}]
        data="".join(json.dumps(r)+"\n" for r in rows).encode()
        calls=[]
        def adb(*args,**kwargs):
            calls.append(args)
            if args[0]=="pull":pathlib.Path(args[2]).write_bytes(data)
            cmd=args[1]
            if cmd.startswith("touch") and touch_fails:raise subprocess.CalledProcessError(1,cmd)
            if "/proc/42/cmdline" in cmd:out="/other/process" if foreign_pid else "/owned/gpu-clock-sampler\0"+c.trace
            elif cmd.startswith("sha256sum"):out=hashlib.sha256(data).hexdigest()+"  trace"
            else:out=""
            return subprocess.CompletedProcess(args,0,stdout=out,stderr="")
        c.adb=adb
        return c,calls

    def test_success_stops_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,calls=self.make(tmp);c.stop()
            self.assertEqual(c.record["footer"]["read_errors"],0)
            self.assertTrue(any(a[1].startswith("rm ") for a in calls))

    def test_failed_stopfile_still_terminates_owned_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,calls=self.make(tmp,touch_fails=True);c.stop()
            self.assertIn("stop_file_error",c.record)
            self.assertTrue(any("kill -TERM 42" in a[1] for a in calls))

    def test_foreign_pid_never_signalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,calls=self.make(tmp,foreign_pid=True)
            with self.assertRaisesRegex(RuntimeError,"could not confirm"):c.stop()
            self.assertFalse(any("kill -TERM" in a[1] for a in calls))
            c.process.terminate.assert_called_once()
            self.assertEqual(c.record["remote_lifetime_cap_seconds"],120)

    def test_invalid_observation_rejected_evidence_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            c,calls=self.make(tmp,bad_sample=True)
            with self.assertRaises(AssertionError):c.stop()
            self.assertIn("stop_or_validation_error",c.record)
            self.assertFalse(any(a[1].startswith("rm ") for a in calls))
            self.assertTrue((pathlib.Path(tmp)/"test.gpu-trace.jsonl").exists())


if __name__=="__main__":unittest.main()
