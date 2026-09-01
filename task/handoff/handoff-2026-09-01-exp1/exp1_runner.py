#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SERIAL_DEFAULT = "3B661501LA000000"
MODEL_DEFAULT = "/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
REMOTE_DIR_DEFAULT = "/data/local/tmp/llama-exp1-20260901"
PORT_DEFAULT = 18080


@dataclass(frozen=True)
class Config:
    name: str
    load_mode: str
    devices: str
    ngl: str
    hexagon_devices: str | None = None
    tensor_split: str | None = None
    run_inference: bool = True


CONFIGS = {
    "cpu_mmap": Config("cpu_mmap", "mmap", "none", "0"),
    "cpu_nommap": Config("cpu_nommap", "none", "none", "0", run_inference=False),
    "opencl_mmap": Config("opencl_mmap", "mmap", "GPUOpenCL", "all"),
    "htp1": Config("htp1", "mmap", "HTP0:0", "all", "HTP0:0", "1"),
    "htp2": Config("htp2", "mmap", "HTP0:0,HTP0:1", "all", "HTP0:0,HTP0:1", "1,1"),
    "htp4": Config(
        "htp4",
        "mmap",
        "HTP0:0,HTP0:1,HTP0:2,HTP0:3",
        "all",
        "HTP0:0,HTP0:1,HTP0:2,HTP0:3",
        "1,1,1,1",
    ),
}


def run(cmd: list[str], *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=check, timeout=timeout, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def adb(serial: str, *args: str, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return run(["adb", "-s", serial, *args], check=check, timeout=timeout)


def adb_shell(serial: str, command: str, *, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return adb(serial, "shell", command, check=check, timeout=timeout)


def device_http_json(
    serial: str,
    port: int,
    path: str,
    payload: Any | None = None,
    timeout: float = 30.0,
) -> Any:
    url = f"http://127.0.0.1:{port}{path}"
    curl = ["curl", "-sS", "-m", str(int(timeout)), "-H", "Content-Type: application/json"]
    if payload is not None:
        curl.extend(["--data-binary", json.dumps(payload, separators=(",", ":"))])
    curl.append(url)
    command = " ".join(shlex.quote(part) for part in curl)
    result = adb_shell(serial, command, check=False, timeout=timeout + 5)
    if result.returncode != 0:
        raise RuntimeError(result.stdout.strip() or f"device curl exited with {result.returncode}")
    return json.loads(result.stdout)


def wait_for_health(serial: str, port: int, process: subprocess.Popen[bytes], timeout_s: float) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout_s
    last_error = "server not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False, f"server exited with code {process.returncode}"
        try:
            health = device_http_json(serial, port, "/health", timeout=2.0)
            if health.get("status") == "ok":
                return True, "ready"
            last_error = json.dumps(health, ensure_ascii=False)
        except Exception as exc:  # server may not have opened the socket yet
            last_error = str(exc)
        time.sleep(0.5)
    return False, f"timeout waiting for health: {last_error}"


def get_pid(serial: str, name: str, timeout_s: float = 15.0) -> int | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = adb_shell(serial, f"pidof {shlex.quote(name)}", check=False, timeout=5)
        values = [value for value in result.stdout.strip().split() if value.isdigit()]
        if values:
            return int(values[0])
        time.sleep(0.2)
    return None


def device_clock_offset_ms(serial: str) -> float:
    offsets = []
    for _ in range(3):
        before = time.time() * 1000.0
        result = adb_shell(serial, "date +%s%3N", timeout=5)
        after = time.time() * 1000.0
        device_ms = float(result.stdout.strip())
        offsets.append(device_ms - ((before + after) / 2.0))
    return statistics.median(offsets)


def wait_for_idle(serial: str, remote_dir: str, model: str) -> dict[str, Any]:
    adb_shell(serial, "pkill -TERM llama-server 2>/dev/null || true", check=False)
    time.sleep(1.0)
    adb_shell(serial, "pkill -KILL llama-server 2>/dev/null || true", check=False)

    cache_result = adb_shell(
        serial,
        f"{shlex.quote(remote_dir)}/drop-file-cache {shlex.quote(model)}",
        check=False,
        timeout=120,
    )
    samples = []
    for _ in range(5):
        result = adb_shell(
            serial,
            "awk '/^MemAvailable:/ {a=$2} /^Cached:/ {c=$2} END {print a, c}' /proc/meminfo",
            timeout=5,
        )
        fields = result.stdout.strip().split()
        if len(fields) == 2:
            samples.append({"memavailable_kb": int(fields[0]), "cached_kb": int(fields[1])})
        time.sleep(1.0)
    return {
        "cache_drop_exit": cache_result.returncode,
        "cache_drop_output": cache_result.stdout.strip(),
        "idle_samples": samples,
    }


def make_prompt_tokens(serial: str, port: int, prompt_file: Path) -> list[int]:
    if prompt_file.exists():
        data = json.loads(prompt_file.read_text(encoding="utf-8"))
        tokens = data["tokens"]
        if len(tokens) != 512:
            raise RuntimeError(f"prompt token file contains {len(tokens)} tokens, expected 512")
        return [int(token) for token in tokens]

    seed = (
        " Mixture-of-experts inference routes every input token to a small subset of specialized experts."
        " This fixed paragraph is repeated to create a deterministic systems benchmark prompt."
    ) * 128
    tokenized = device_http_json(serial, port, "/tokenize", {"content": seed, "add_special": False}, timeout=30)
    tokens = [int(token) for token in tokenized["tokens"]]
    if len(tokens) < 512:
        raise RuntimeError(f"tokenizer returned only {len(tokens)} tokens")
    tokens = tokens[:512]
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(
        json.dumps({"description": "First 512 tokens of the fixed Exp1 seed text; no BOS inserted", "tokens": tokens}, indent=2),
        encoding="utf-8",
    )
    return tokens


def stream_completion(serial: str, port: int, tokens: list[int], raw_path: Path) -> dict[str, Any]:
    payload = {
        "prompt": tokens,
        "n_predict": 128,
        "stream": True,
        "timings_per_token": True,
        "return_progress": True,
        "return_tokens": True,
        "cache_prompt": False,
        "temperature": 0,
        "seed": 42,
        "ignore_eos": True,
        "sse_ping_interval": 1,
    }
    curl_command = " ".join(
        shlex.quote(part)
        for part in [
            "curl",
            "-N",
            "-sS",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(payload, separators=(",", ":")),
            f"http://127.0.0.1:{port}/completion",
        ]
    )

    start_monotonic = time.monotonic()
    start_wall_ms = time.time() * 1000.0
    first_token_monotonic = None
    first_token_wall_ms = None
    final = None
    events = []

    curl_process = subprocess.Popen(
        ["adb", "-s", serial, "exec-out", "sh", "-c", curl_command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert curl_process.stdout is not None
    with raw_path.open("w", encoding="utf-8") as raw:
        for raw_line in curl_process.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            raw.write(line + "\n")
            if not line.startswith("data: "):
                continue
            data = json.loads(line[6:])
            events.append(data)
            has_token = int(data.get("tokens_predicted", 0)) > 0
            if has_token and first_token_monotonic is None:
                first_token_monotonic = time.monotonic()
                first_token_wall_ms = time.time() * 1000.0
            if data.get("stop") or data.get("stopped_eos") or data.get("stopped_limit"):
                final = data

    curl_returncode = curl_process.wait(timeout=30)
    if curl_returncode != 0:
        raise RuntimeError(f"device streaming curl exited with {curl_returncode}")

    end_monotonic = time.monotonic()
    end_wall_ms = time.time() * 1000.0
    if first_token_monotonic is None:
        raise RuntimeError("stream completed without a token event")
    if final is None:
        for event in reversed(events):
            if "timings" in event:
                final = event
                break
    if final is None or "timings" not in final:
        raise RuntimeError("stream completed without final timing data")

    timings = final["timings"]
    return {
        "request_start_wall_ms": start_wall_ms,
        "first_token_wall_ms": first_token_wall_ms,
        "request_end_wall_ms": end_wall_ms,
        "ttft_e2e_ms": (first_token_monotonic - start_monotonic) * 1000.0,
        "request_total_ms": (end_monotonic - start_monotonic) * 1000.0,
        "ttft_server_ms": timings.get("prompt_ms"),
        "tpot_ms": timings.get("predicted_per_token_ms"),
        "tokens_evaluated": final.get("tokens_evaluated", timings.get("prompt_n")),
        "tokens_predicted": final.get("tokens_predicted", timings.get("predicted_n")),
        "timings": timings,
        "stop_type": final.get("stop_type"),
    }


def read_monitor(path: Path) -> list[dict[str, int]]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                parsed = {key: int(value) for key, value in row.items()}
            except (TypeError, ValueError):
                continue
            if parsed["timestamp_ms"] > 0:
                rows.append(parsed)
    return rows


def nearest_value(rows: list[dict[str, int]], timestamp_ms: float, key: str) -> int | None:
    valid = [row for row in rows if row.get(key, -1) >= 0]
    if not valid:
        return None
    return min(valid, key=lambda row: abs(row["timestamp_ms"] - timestamp_ms))[key]


def summarize_monitor(rows: list[dict[str, int]], request: dict[str, Any] | None, clock_offset_ms: float) -> dict[str, Any]:
    valid_pss = [row["pss_kb"] for row in rows if row["pss_kb"] >= 0]
    valid_avail = [row["memavailable_kb"] for row in rows if row["memavailable_kb"] >= 0]
    valid_maj = [row["majflt"] for row in rows if row["majflt"] >= 0]
    summary: dict[str, Any] = {
        "samples": len(rows),
        "peak_pss_kb": max(valid_pss) if valid_pss else None,
        "min_memavailable_kb": min(valid_avail) if valid_avail else None,
        "majflt_process_delta": (valid_maj[-1] - valid_maj[0]) if valid_maj else None,
    }
    if request is not None:
        start = request["request_start_wall_ms"] + clock_offset_ms
        first = request["first_token_wall_ms"] + clock_offset_ms
        end = request["request_end_wall_ms"] + clock_offset_ms
        maj_start = nearest_value(rows, start, "majflt")
        maj_first = nearest_value(rows, first, "majflt")
        maj_end = nearest_value(rows, end, "majflt")
        summary.update(
            {
                "majflt_request_delta": None if maj_start is None or maj_end is None else maj_end - maj_start,
                "majflt_prompt_delta": None if maj_start is None or maj_first is None else maj_first - maj_start,
                "majflt_decode_delta": None if maj_first is None or maj_end is None else maj_end - maj_first,
            }
        )
    return summary


def filtered_logcat(serial: str, start_device_ms: float) -> str:
    result = adb(serial, "logcat", "-b", "all", "-d", "-v", "epoch", check=False, timeout=30)
    keywords = re.compile(r"lmkd|lowmemorykiller|oom-kill|out of memory|am_kill|killing|llama-server", re.I)
    lines = []
    threshold_s = (start_device_ms - 1000.0) / 1000.0
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=1)
        try:
            timestamp_s = float(fields[0])
        except (ValueError, IndexError):
            continue
        if timestamp_s >= threshold_s and keywords.search(line):
            lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def parse_model_buffers(server_log: str) -> dict[str, float]:
    buffers: dict[str, float] = {}
    pattern = re.compile(r"load_tensors:\s+(.+?) model buffer size\s*=\s*([0-9.]+) MiB")
    for match in pattern.finditer(server_log):
        buffers[match.group(1).strip()] = float(match.group(2))
    return buffers


def run_one(args: argparse.Namespace, config: Config, repeat: int, prompt_file: Path) -> dict[str, Any]:
    run_dir = args.output_dir / config.name / f"run-{repeat:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    remote_monitor = f"{args.remote_dir}/monitor-{config.name}-{repeat:02d}.csv"
    local_monitor = run_dir / "monitor.csv"
    server_log_path = run_dir / "server.log"
    raw_sse_path = run_dir / "completion.sse"
    logcat_path = run_dir / "logcat-filtered.log"

    idle = wait_for_idle(args.serial, args.remote_dir, args.model)
    clock_offset_ms = device_clock_offset_ms(args.serial)
    start_device_ms = time.time() * 1000.0 + clock_offset_ms

    env = [
        f"LD_LIBRARY_PATH={shlex.quote(args.remote_dir)}",
        f"ADSP_LIBRARY_PATH={shlex.quote(args.remote_dir + ';/vendor/dsp;/system/lib/rfsa/adsp;/vendor/lib/rfsa/adsp')}",
    ]
    if config.hexagon_devices:
        env.append(f"GGML_HEXAGON_DEVICES={shlex.quote(config.hexagon_devices)}")

    server_args = [
        "./llama-server",
        "-m", args.model,
        "-lm", config.load_mode,
        "-lzm", "off",
        "-dev", config.devices,
        "-ngl", config.ngl,
        "-c", "2048",
        "-b", "512",
        "-ub", "128",
        "-t", "4",
        "-fa", "off",
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--parallel", "1",
        "--no-cont-batching",
        "--no-warmup",
        "-lv", "4",
    ]
    if config.tensor_split:
        server_args.extend(["-sm", "layer", "-ts", config.tensor_split])

    remote_command = (
        f"cd {shlex.quote(args.remote_dir)} && "
        + " ".join(env)
        + " exec "
        + " ".join(shlex.quote(part) for part in server_args)
    )

    monitor_process: subprocess.Popen[bytes] | None = None
    with server_log_path.open("wb") as server_log_handle:
        process = subprocess.Popen(
            ["adb", "-s", args.serial, "shell", remote_command],
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
        )

        pid = get_pid(args.serial, "llama-server")
        if pid is not None:
            monitor_cmd = (
                f"cd {shlex.quote(args.remote_dir)} && "
                f"exec ./device_monitor.sh {pid} {shlex.quote(remote_monitor)}"
            )
            monitor_process = subprocess.Popen(
                ["adb", "-s", args.serial, "shell", monitor_cmd],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        ready, ready_reason = wait_for_health(args.serial, args.port, process, args.load_timeout)
        request_metrics = None
        request_error = None
        if ready and config.run_inference:
            try:
                tokens = make_prompt_tokens(args.serial, args.port, prompt_file)
                request_metrics = stream_completion(args.serial, args.port, tokens, raw_sse_path)
            except Exception as exc:
                request_error = f"{type(exc).__name__}: {exc}"

        if pid is not None:
            adb_shell(args.serial, f"kill -TERM {pid} 2>/dev/null || true", check=False)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if pid is not None:
                adb_shell(args.serial, f"kill -KILL {pid} 2>/dev/null || true", check=False)
            process.kill()
            process.wait(timeout=5)

        if monitor_process is not None:
            try:
                monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor_process.terminate()
                try:
                    monitor_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    monitor_process.kill()
                    monitor_process.wait(timeout=2)

    time.sleep(0.5)
    adb(args.serial, "pull", remote_monitor, str(local_monitor), check=False, timeout=30)
    adb_shell(args.serial, f"rm -f {shlex.quote(remote_monitor)}", check=False)
    logcat_text = filtered_logcat(args.serial, start_device_ms)
    logcat_path.write_text(logcat_text, encoding="utf-8")
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    rows = read_monitor(local_monitor)
    monitor_summary = summarize_monitor(rows, request_metrics, clock_offset_ms)
    buffers = parse_model_buffers(server_log)

    target_kill_evidence = bool(
        re.search(r"(?:oom-kill|out of memory)", logcat_text, re.I)
        or re.search(r"(?:lmkd|lowmemorykiller|am_kill|killing).*llama-server", logcat_text, re.I)
        or re.search(r"llama-server.*(?:lmkd|lowmemorykiller|am_kill|killing)", logcat_text, re.I)
    )
    system_memory_pressure_events = len(
        re.findall(r"am_kill|lowmemorykiller|lmkd", logcat_text, re.I)
    )
    if ready and (not config.run_inference or (request_metrics is not None and request_error is None)):
        status = "success"
    elif target_kill_evidence or process.returncode in (137, -9):
        status = "killed_or_oom"
    elif not ready:
        status = "load_failed"
    else:
        status = "inference_failed"

    result = {
        "config": config.name,
        "repeat": repeat,
        "status": status,
        "server_ready": ready,
        "server_ready_reason": ready_reason,
        "server_exit_code": process.returncode,
        "pid": pid,
        "idle": idle,
        "clock_offset_ms": clock_offset_ms,
        "request": request_metrics,
        "request_error": request_error,
        "monitor": monitor_summary,
        "model_buffers_mib": buffers,
        "htp_model_buffer_total_mib": sum(value for name, value in buffers.items() if name.startswith("HTP")),
        "htp_repack_total_mib": sum(
            value for name, value in buffers.items() if name.startswith("HTP") and "REPACK" in name
        ),
        "lmkd_oom_evidence": target_kill_evidence,
        "system_memory_pressure_events": system_memory_pressure_events,
        "server_command": remote_command,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--start-repeat", type=int, default=1)
    parser.add_argument("--serial", default=SERIAL_DEFAULT)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--remote-dir", default=REMOTE_DIR_DEFAULT)
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = args.output_dir / "prompt_tokens.json"
    config = CONFIGS[args.config]
    results = []
    stop_repeat = args.start_repeat + args.repeats
    for repeat in range(args.start_repeat, stop_repeat):
        print(f"[{config.name}] starting repeat {repeat}", flush=True)
        result = run_one(args, config, repeat, prompt_file)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary_path = args.output_dir / config.name / "summary.json"
    all_results = []
    for result_path in sorted((args.output_dir / config.name).glob("run-*/result.json")):
        all_results.append(json.loads(result_path.read_text(encoding="utf-8")))
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if all(result["status"] == "success" for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
