#!/usr/bin/env bash
set -euo pipefail

DEVICE="${DEVICE:-fd8657d6}"
BUILD_DIR="${BUILD_DIR:-build-android-pd-final}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp/llama-opbench-20260712}"
MODEL_DIR_REMOTE="${MODEL_DIR_REMOTE:-/data/local/tmp/models/Phi-mini-MoE}"
MODEL_Q4_0_HOST="${MODEL_Q4_0_HOST:-MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf}"
MODEL_Q8_0_HOST="${MODEL_Q8_0_HOST:-MOE/models/Phi-mini-MoE-instruct-Q8_0.gguf}"
MODEL_Q4_0_REMOTE="${MODEL_Q4_0_REMOTE:-${MODEL_DIR_REMOTE}/Phi-mini-MoE-instruct-Q4_0.gguf}"
MODEL_Q8_0_REMOTE="${MODEL_Q8_0_REMOTE:-${MODEL_DIR_REMOTE}/Phi-mini-MoE-instruct-Q8_0.gguf}"
RESULT_DIR="${RESULT_DIR:-MOE/results/phi-mini-moe-sparse-opbench-${DEVICE}-20260712}"
RAW_DIR="${RESULT_DIR}/raw"

RUN_DEPLOY="${RUN_DEPLOY:-1}"
RUN_OPBENCH="${RUN_OPBENCH:-1}"
RUN_E2E="${RUN_E2E:-1}"
PUSH_Q8="${PUSH_Q8:-0}"
OPBENCH_TIMEOUT_SEC="${OPBENCH_TIMEOUT_SEC:-2400}"
E2E_TIMEOUT_SEC="${E2E_TIMEOUT_SEC:-900}"
E2E_REPS="${E2E_REPS:-1}"
COOLDOWN_SEC="${COOLDOWN_SEC:-5}"

HTP_HOSTBUF="${HTP_HOSTBUF:-1}"
HTP_HMX="${HTP_HMX:-1}"
HTP_NHVX="${HTP_NHVX:-0}"
HTP_PROFILE="${HTP_PROFILE:-1}"

OP_FILTER="MUL_MAT_PHI_ROUTER,TOPK_MOE_PHI,MUL_MAT_ID_PHI_GATE,MUL_MAT_ID_PHI_UP,MUL_MAT_ID_PHI_DOWN,MOE_FFN_SPARSE_PHI"
OP_PARAMS_REGEX="${OP_PARAMS_REGEX:-case=(MUL_MAT_PHI_ROUTER|TOPK_MOE_PHI|MUL_MAT_ID_PHI_GATE|MUL_MAT_ID_PHI_UP|MUL_MAT_ID_PHI_DOWN|MOE_FFN_SPARSE_PHI)}"

mkdir -p "${RAW_DIR}"

run_adb() {
    adb -s "${DEVICE}" "$@"
}

remote_exists() {
    run_adb shell "test -e '$1'"
}

deploy() {
    run_adb get-state >/dev/null
    run_adb shell "mkdir -p '${REMOTE_DIR}' '${MODEL_DIR_REMOTE}'"

    run_adb push "${BUILD_DIR}/bin/test-backend-ops" "${REMOTE_DIR}/"
    run_adb push "${BUILD_DIR}/bin/llama-bench" "${REMOTE_DIR}/"
    run_adb push "${BUILD_DIR}/bin/backend-op-bench" "${REMOTE_DIR}/"
    run_adb push "${BUILD_DIR}/bin/"*.so "${REMOTE_DIR}/"

    while IFS= read -r so; do
        run_adb push "${so}" "${REMOTE_DIR}/"
    done < <(find "${BUILD_DIR}/ggml/src/ggml-hexagon" -maxdepth 1 -name 'libggml-htp*.so' -print | sort)

    run_adb shell "chmod +x '${REMOTE_DIR}'/*"

    if ! remote_exists "${MODEL_Q4_0_REMOTE}"; then
        run_adb push "${MODEL_Q4_0_HOST}" "${MODEL_DIR_REMOTE}/"
    fi

    if [[ "${PUSH_Q8}" == "1" ]] && ! remote_exists "${MODEL_Q8_0_REMOTE}"; then
        run_adb push "${MODEL_Q8_0_HOST}" "${MODEL_DIR_REMOTE}/"
    fi
}

write_manifest() {
    {
        printf 'device=%s\n' "${DEVICE}"
        printf 'remote_dir=%s\n' "${REMOTE_DIR}"
        printf 'result_dir=%s\n' "${RESULT_DIR}"
        printf 'build_dir=%s\n' "${BUILD_DIR}"
        printf 'model_q4_0=%s\n' "${MODEL_Q4_0_REMOTE}"
        printf 'run_opbench=%s\n' "${RUN_OPBENCH}"
        printf 'run_e2e=%s\n' "${RUN_E2E}"
        printf 'op_filter=%s\n' "${OP_FILTER}"
        printf 'op_params_regex=%s\n' "${OP_PARAMS_REGEX}"
        printf 'opbench_backends=CPU GPUOpenCL HTP0\n'
        printf 'opbench_htp_env=GGML_HEXAGON_EXPERIMENTAL=1 GGML_HEXAGON_HOSTBUF=%s GGML_HEXAGON_NDEV=1 GGML_HEXAGON_NHVX=%s GGML_HEXAGON_USE_HMX=%s GGML_HEXAGON_PROFILE=%s\n' \
            "${HTP_HOSTBUF}" "${HTP_NHVX}" "${HTP_HMX}" "${HTP_PROFILE}"
        printf 'e2e_common=-t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0\n'
        printf 'e2e_cpu=-ngl 0 -dev none -ncmoe 999\n'
        printf 'e2e_gpu=taskset 80 -ngl 99 -dev GPUOpenCL -ncmoe 0 flash_attn=false\n'
        printf 'e2e_htp=taskset 80 -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 HMX=%s HOSTBUF=%s\n' "${HTP_HMX}" "${HTP_HOSTBUF}"
        printf 'e2e_workloads=pp128_tg0 pp0_tg32\n'
        printf 'e2e_reps=%s\n' "${E2E_REPS}"
        printf 'cooldown_sec=%s\n' "${COOLDOWN_SEC}"
        printf 'git_commit=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
        printf 'git_status_short_begin\n'
        git status --short 2>/dev/null || true
        printf 'git_status_short_end\n'
    } > "${RESULT_DIR}/manifest.txt"
}

run_remote_case() {
    local name="$1"
    local timeout_sec="$2"
    local remote_cmd="$3"
    local stdout_file="${RAW_DIR}/${name}.stdout"
    local stderr_file="${RAW_DIR}/${name}.stderr"
    local command_file="${RAW_DIR}/${name}.command"
    local exit_file="${RAW_DIR}/${name}.exit"

    printf '%s\n' "timeout ${timeout_sec} adb -s ${DEVICE} shell '${remote_cmd}'" > "${command_file}"
    set +e
    timeout "${timeout_sec}" adb -s "${DEVICE}" shell "${remote_cmd}" > "${stdout_file}" 2> "${stderr_file}"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${exit_file}"
    return 0
}

pull_htp_profiles() {
    local prefix="$1"
    for profile in hex_HTP0_profiling.csv hex_HTP0_stage_profiling.csv; do
        if run_adb shell "test -f '${REMOTE_DIR}/${profile}'"; then
            run_adb pull "${REMOTE_DIR}/${profile}" "${RAW_DIR}/${prefix}.${profile}" >/dev/null || true
            run_adb shell "rm -f '${REMOTE_DIR}/${profile}'" >/dev/null || true
        fi
    done
}

base_env() {
    printf "cd '%s' && export LD_LIBRARY_PATH='%s':\$LD_LIBRARY_PATH && export GGML_QNN_DISABLE_BACKEND=1 && export GGML_HETERO_DYNAMIC_ALLOW_QNN=0" \
        "${REMOTE_DIR}" "${REMOTE_DIR}"
}

htp_env() {
    printf "%s && export ADSP_LIBRARY_PATH='%s' && export GGML_HEXAGON_EXPERIMENTAL=1 && export GGML_HEXAGON_HOSTBUF=%s && export GGML_HEXAGON_NDEV=1 && export GGML_HEXAGON_NHVX=%s && export GGML_HEXAGON_USE_HMX=%s && export GGML_HEXAGON_PROFILE=%s" \
        "$(base_env)" "${REMOTE_DIR}" "${HTP_HOSTBUF}" "${HTP_NHVX}" "${HTP_HMX}" "${HTP_PROFILE}"
}

run_opbench() {
    local common="./test-backend-ops perf -o '${OP_FILTER}' -p '${OP_PARAMS_REGEX}' --output csv"

    run_remote_case "opbench_cpu" "${OPBENCH_TIMEOUT_SEC}" \
        "$(base_env) && ${common} -b CPU"

    sleep "${COOLDOWN_SEC}"

    run_remote_case "opbench_gpu_opencl" "${OPBENCH_TIMEOUT_SEC}" \
        "$(base_env) && taskset 80 ${common} -b GPUOpenCL"

    sleep "${COOLDOWN_SEC}"

    run_remote_case "opbench_htp0_hmx" "${OPBENCH_TIMEOUT_SEC}" \
        "$(htp_env) && rm -f hex_HTP0_profiling.csv hex_HTP0_stage_profiling.csv && taskset 80 ${common} -b HTP0"
    pull_htp_profiles "opbench_htp0_hmx"
}

run_e2e() {
    local common="./llama-bench -v -r ${E2E_REPS} -o jsonl -m '${MODEL_Q4_0_REMOTE}' -t 8 -c 1024 -b 128 -ub 128 -p 0 -n 0 --no-warmup --mmap 0"

    run_remote_case "e2e_cpu_pp128_tg0" "${E2E_TIMEOUT_SEC}" \
        "$(base_env) && ${common} -ngl 0 -dev none -ncmoe 999 -pg 128,0"
    sleep "${COOLDOWN_SEC}"
    run_remote_case "e2e_cpu_pp0_tg32" "${E2E_TIMEOUT_SEC}" \
        "$(base_env) && ${common} -ngl 0 -dev none -ncmoe 999 -pg 0,32"
    sleep "${COOLDOWN_SEC}"

    run_remote_case "e2e_gpu_opencl_pp128_tg0" "${E2E_TIMEOUT_SEC}" \
        "$(base_env) && taskset 80 ${common} -ngl 99 -dev GPUOpenCL -ncmoe 0 -pg 128,0"
    sleep "${COOLDOWN_SEC}"
    run_remote_case "e2e_gpu_opencl_pp0_tg32" "${E2E_TIMEOUT_SEC}" \
        "$(base_env) && taskset 80 ${common} -ngl 99 -dev GPUOpenCL -ncmoe 0 -pg 0,32"
    sleep "${COOLDOWN_SEC}"

    run_remote_case "e2e_htp0_hmx_pp128_tg0" "${E2E_TIMEOUT_SEC}" \
        "$(htp_env) && export GGML_SCHED_DEBUG=2 && rm -f hex_HTP0_profiling.csv hex_HTP0_stage_profiling.csv && taskset 80 ${common} -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 -pg 128,0"
    pull_htp_profiles "e2e_htp0_hmx_pp128_tg0"
    sleep "${COOLDOWN_SEC}"
    run_remote_case "e2e_htp0_hmx_pp0_tg32" "${E2E_TIMEOUT_SEC}" \
        "$(htp_env) && export GGML_SCHED_DEBUG=2 && rm -f hex_HTP0_profiling.csv hex_HTP0_stage_profiling.csv && taskset 80 ${common} -fa 1 -ngl 99 -dev HTP0 -ncmoe 0 -pg 0,32"
    pull_htp_profiles "e2e_htp0_hmx_pp0_tg32"
}

summarize() {
    python3 - "${RESULT_DIR}" <<'PY'
import csv
import json
import math
import re
import sys
from pathlib import Path

result = Path(sys.argv[1])
raw = result / "raw"

def parse_params(s):
    out = {}
    for key in ["case", "type_a", "type_b", "n_tokens", "b", "m", "n", "k", "n_mats", "n_used", "gating_func"]:
        m = re.search(rf"(?:^|,){key}=([^,]+)", s)
        if m:
            out[key] = m.group(1)
    ne = re.search(r"ne=\[(\d+),(\d+),(\d+),(\d+)\]", s)
    if ne and "n_tokens" not in out:
        out["n_tokens"] = ne.group(2)
    return out

rows = []
for path in sorted(raw.glob("opbench_*.stdout")):
    backend_case = path.stem.replace("opbench_", "")
    exit_path = path.with_suffix(".exit")
    exit_code = exit_path.read_text().strip() if exit_path.exists() else ""
    with path.open(newline="") as f:
        reader = csv.DictReader(line for line in f if line.startswith('"'))
        for row in reader:
            params = parse_params(row.get("op_params", ""))
            time_us = row.get("time_us", "")
            flops = row.get("flops", "")
            try:
                time_us_f = float(time_us)
            except ValueError:
                time_us_f = 0.0
            try:
                flops_f = float(flops)
            except ValueError:
                flops_f = 0.0
            rows.append({
                "kind": "opbench",
                "backend_case": backend_case,
                "backend_name": row.get("backend_name", ""),
                "op_name": row.get("op_name", ""),
                "case": params.get("case", row.get("op_name", "")),
                "type_a": params.get("type_a", ""),
                "type_b": params.get("type_b", ""),
                "n_tokens": params.get("n_tokens", params.get("n", "")),
                "broadcast_b": params.get("b", ""),
                "m": params.get("m", ""),
                "n": params.get("n", ""),
                "k": params.get("k", ""),
                "n_mats": params.get("n_mats", ""),
                "n_used": params.get("n_used", ""),
                "time_us": time_us,
                "time_ms": f"{time_us_f / 1000.0:.6f}" if time_us else "",
                "n_runs": row.get("n_runs", ""),
                "supported": row.get("supported", ""),
                "passed": row.get("passed", ""),
                "error_message": row.get("error_message", ""),
                "flops": flops,
                "gflops": f"{flops_f / 1e9:.6f}" if flops else "",
                "memory_kb": row.get("memory_kb", ""),
                "raw_file": path.name,
                "exit": exit_code,
            })

summary_fields = [
    "kind", "backend_case", "backend_name", "op_name", "case", "type_a", "type_b",
    "n_tokens", "broadcast_b", "m", "n", "k", "n_mats", "n_used",
    "time_us", "time_ms", "n_runs", "supported", "passed", "error_message",
    "flops", "gflops", "memory_kb", "raw_file", "exit",
]
with (result / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(rows)

bench_rows = []
for path in sorted(raw.glob("e2e_*.stdout")):
    exit_path = path.with_suffix(".exit")
    exit_code = exit_path.read_text().strip() if exit_path.exists() else ""
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        pp = int(obj.get("n_prompt", obj.get("pp", 0)) or 0)
        tg = int(obj.get("n_gen", obj.get("tg", 0)) or 0)
        metric = "avg_ts"
        tok_s = obj.get("avg_ts")
        if pp > 0 and tg == 0 and obj.get("avg_pp_ts") is not None:
            metric = "avg_pp_ts"
            tok_s = obj.get("avg_pp_ts")
        elif tg > 0 and pp == 0 and obj.get("avg_tg_ts") is not None:
            metric = "avg_tg_ts"
            tok_s = obj.get("avg_tg_ts")
        bench_rows.append({
            "case": path.stem,
            "backend": "HTP0 FastRPC" if "htp0" in path.stem else ("GPUOpenCL" if "gpu" in path.stem else "CPU"),
            "pp": pp,
            "tg": tg,
            "tok_s": tok_s if tok_s is not None else "",
            "metric": metric,
            "avg_ns": obj.get("avg_ns", ""),
            "avg_pp_ns": obj.get("avg_pp_ns", ""),
            "avg_tg_ns": obj.get("avg_tg_ns", ""),
            "devices": obj.get("devices", ""),
            "flash_attn": obj.get("flash_attn", ""),
            "build_commit": obj.get("build_commit", ""),
            "exit": exit_code,
            "raw_file": path.name,
        })

bench_fields = ["case", "backend", "pp", "tg", "tok_s", "metric", "avg_ns", "avg_pp_ns", "avg_tg_ns", "devices", "flash_attn", "build_commit", "exit", "raw_file"]
with (result / "llama_bench_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=bench_fields)
    writer.writeheader()
    writer.writerows(bench_rows)

def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

supported = [r for r in rows if r["supported"] == "1" and r["passed"] == "1" and fnum(r["time_us"])]
unsupported = [r for r in rows if r["supported"] != "1" or r["passed"] != "1"]

lines = []
lines.append("# Phi-mini-MoE sparse operator timing")
lines.append("")
lines.append("Primary table: `summary.csv`. `time_us` is average wall time per operator run; for `MOE_FFN_SPARSE_PHI` it is one full sparse MoE mini-graph run.")
lines.append("")
lines.append("## Backend command policy")
lines.append("")
lines.append("- CPU end-to-end: `-ngl 0 -dev none -ncmoe 999`, no FlashAttention flag, matching the previous CPU run.")
lines.append("- GPUOpenCL end-to-end: `taskset 80 -ngl 99 -dev GPUOpenCL -ncmoe 0`, FlashAttention remains false, matching the previous GPU run.")
lines.append("- HTP0 FastRPC end-to-end: `taskset 80 -fa 1 -ngl 99 -dev HTP0 -ncmoe 0`, with `GGML_HEXAGON_USE_HMX=1`.")
lines.append("- Operator microbenchmarks have no attention path; they compare only router/top-k and sparse expert MoE operators.")
lines.append("")
lines.append("## Fastest supported timings by operator")
lines.append("")
lines.append("| op | quant | n_tokens | b | CPU us | GPU us | HTP us |")
lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
index = {}
for r in supported:
    key = (r["op_name"], r["type_a"], r["n_tokens"], r["broadcast_b"])
    index.setdefault(key, {})[r["backend_case"]] = fnum(r["time_us"])
for key in sorted(index, key=lambda x: (x[0], x[1], int(x[2] or 0), x[3])):
    vals = index[key]
    def fmt(name):
        v = vals.get(name)
        return f"{v:.2f}" if v is not None else ""
    lines.append(f"| `{key[0]}` | `{key[1]}` | {key[2]} | `{key[3]}` | {fmt('cpu')} | {fmt('gpu_opencl')} | {fmt('htp0_hmx')} |")

if unsupported:
    lines.append("")
    lines.append("## Unsupported or failed operator cases")
    lines.append("")
    lines.append("| backend | op | params | error |")
    lines.append("| --- | --- | --- | --- |")
    for r in unsupported[:80]:
        params = r["case"]
        if r["type_a"]:
            params += f" {r['type_a']} n_tokens={r['n_tokens']} b={r['broadcast_b']}"
        lines.append(f"| `{r['backend_case']}` | `{r['op_name']}` | `{params}` | `{r['error_message']}` |")
    if len(unsupported) > 80:
        lines.append(f"| ... | ... | ... | {len(unsupported) - 80} more rows in summary.csv |")

if bench_rows:
    lines.append("")
    lines.append("## End-to-end check")
    lines.append("")
    lines.append("| backend | workload | tok/s | metric | flash_attn | devices |")
    lines.append("| --- | --- | ---: | --- | --- | --- |")
    for r in bench_rows:
        tok = fnum(r["tok_s"])
        tok_s = f"{tok:.3f}" if tok is not None else ""
        lines.append(f"| `{r['backend']}` | `pp{r['pp']}_tg{r['tg']}` | {tok_s} | `{r['metric']}` | `{r['flash_attn']}` | `{r['devices']}` |")

lines.append("")
lines.append("## Artifacts")
lines.append("")
lines.append("- `summary.csv`: per-operator timing table.")
lines.append("- `llama_bench_summary.csv`: strict end-to-end command check.")
lines.append("- `raw/`: stdout, stderr, exact command, exit code, and HTP profile CSVs.")

(result / "analysis.md").write_text("\n".join(lines) + "\n")
PY
}

if [[ "${RUN_DEPLOY}" == "1" ]]; then
    deploy
fi

write_manifest

if [[ "${RUN_OPBENCH}" == "1" ]]; then
    run_opbench
fi

if [[ "${RUN_E2E}" == "1" ]]; then
    run_e2e
fi

summarize

printf 'Results written to %s\n' "${RESULT_DIR}"
