#!/usr/bin/env bash
set -Eeuo pipefail

DEVICE=${DEVICE:-3B661501LA000000}
REMOTE_BIN_DIR=${REMOTE_BIN_DIR:-/data/local/tmp/llama_test}
MODEL_PATH=${MODEL_PATH:-/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf}
QNN_DIR=${QNN_DIR:-/data/local/tmp/models/Qwen2.5-3B-AoT/qnn}
LOCAL_ROOT=${LOCAL_ROOT:-results/pd-div-${DEVICE}-$(date -u +%Y%m%d-%H%M)}

CASE_TIMEOUT_SEC=${CASE_TIMEOUT_SEC:-1800}
COOLDOWN_SEC=${COOLDOWN_SEC:-30}
REPS=${REPS:-1}
KEEP_RAW=${KEEP_RAW:-1}

CTX_SIZE=${CTX_SIZE:-2048}
THREADS=${THREADS:-4}
FASTRPC_THREADS=${FASTRPC_THREADS:-8}
BENCH_TIMING_LEVEL=${BENCH_TIMING_LEVEL:-3}
BENCH_OUTPUT_FORMAT=${BENCH_OUTPUT_FORMAT:-json}
FLASH_ATTN=${FLASH_ATTN:-1}

BACKENDS_TEXT=${BACKENDS:-"cpu opencl fastrpc"}
SWITCH_WORKLOADS_TEXT=${SWITCH_WORKLOADS:-"128:16 128:64 128:128 256:16 256:64 256:128 512:16 512:64 512:128"}
RUN_SINGLE_BACKEND=${RUN_SINGLE_BACKEND:-0}
SINGLE_WORKLOADS_TEXT=${SINGLE_WORKLOADS:-"128:0 256:0 512:0 0:16 0:64 0:128"}
RUN_CPU_DECODE=${RUN_CPU_DECODE:-0}

QNN_TRACE=${QNN_TRACE:-0}
FASTRPC_DEVICE=${FASTRPC_DEVICE:-HTP0}
FASTRPC_TASKSET=${FASTRPC_TASKSET:-80}
FASTRPC_PD_HOSTBUF=${FASTRPC_PD_HOSTBUF:-1}
FASTRPC_PD_USE_HMX=${FASTRPC_PD_USE_HMX:-1}
FASTRPC_PD_NHVX=${FASTRPC_PD_NHVX:-0}

SUMMARY_ONLY=0
if [ "${1:-}" = "--summary-only" ]; then
    SUMMARY_ONLY=1
    if [ -n "${2:-}" ]; then
        LOCAL_ROOT=$2
    fi
fi

if [ "${BENCH_TIMING_LEVEL}" != "3" ]; then
    printf 'warning: BENCH_TIMING_LEVEL=%s; switch route/KV/reserve fields require level 3\n' "${BENCH_TIMING_LEVEL}" >&2
fi
if [ "${BENCH_OUTPUT_FORMAT}" != "json" ]; then
    printf 'warning: BENCH_OUTPUT_FORMAT=%s; summary parser is designed for json\n' "${BENCH_OUTPUT_FORMAT}" >&2
fi

RAW_DIR="${LOCAL_ROOT}/raw"
SUMMARY_DIR="${LOCAL_ROOT}/summary"
COMMANDS="${LOCAL_ROOT}/commands.sh"
CASES="${LOCAL_ROOT}/cases.tsv"
MANIFEST="${LOCAL_ROOT}/manifest.txt"

IFS=' ' read -r -a backends <<< "${BACKENDS_TEXT}"
IFS=' ' read -r -a switch_workloads <<< "${SWITCH_WORKLOADS_TEXT}"
IFS=' ' read -r -a single_workloads <<< "${SINGLE_WORKLOADS_TEXT}"

if [ "${SUMMARY_ONLY}" -eq 0 ]; then
    mkdir -p "${RAW_DIR}" "${SUMMARY_DIR}"
    printf '#!/usr/bin/env bash\n' > "${COMMANDS}"
    printf 'case\tcategory\tbackend\tprefill\tdecode\tworkload\tpp\ttg\tstdout\tstderr\texit\tprofile\tprofile_pull\tcommand\tenv\n' > "${CASES}"
else
    mkdir -p "${SUMMARY_DIR}"
fi

remote_base_env() {
    printf 'cd %s && mkdir -p %s/results && export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=%s && export LLAMA_BENCH_FAST_EXIT=1' \
        "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}"
}

disable_qnn_env() {
    printf ' && export GGML_QNN_DISABLE_BACKEND=1'
    printf ' && export GGML_HETERO_DYNAMIC_ALLOW_QNN=0'
}

hexagon_env() {
    printf ' && export GGML_HEXAGON_EXPERIMENTAL=1'
    printf ' && export GGML_HEXAGON_HOSTBUF=%s' "${FASTRPC_PD_HOSTBUF}"
    printf ' && export GGML_HEXAGON_NDEV=1'
    printf ' && export GGML_HEXAGON_NHVX=%s' "${FASTRPC_PD_NHVX}"
    printf ' && export GGML_HEXAGON_USE_HMX=%s' "${FASTRPC_PD_USE_HMX}"
    printf ' && unset GGML_HEXAGON_OPFILTER'
}

qnn_env() {
    printf ' && export GGML_HEXAGON_EXPERIMENTAL=1'
    printf ' && export GGML_QNN_AOT_CONFIG=%s/config.json' "${QNN_DIR}"
    printf ' && export GGML_QNN_AOT_MODEL_DIR=%s' "${QNN_DIR}"
    printf ' && export GGML_QNN_AOT_DISABLE_SEED_KV=1'
    printf ' && export GGML_QNN_AOT_WRITE_GENERIC_KV=1'
    if [ "${QNN_TRACE}" = "1" ]; then
        printf ' && export GGML_QNN_AOT_TRACE_ASSIGN=1'
        printf ' && export GGML_QNN_AOT_TRACE_MATCH=1'
    fi
}

fastrpc_env() {
    disable_qnn_env
    hexagon_env
    printf ' && unset GGML_QNN_AOT_CONFIG'
    printf ' && unset GGML_QNN_AOT_MODEL_DIR'
}

route_env() {
    local prefill=$1
    local decode=$2
    if { [ "${prefill}" = fastrpc ] || [ "${decode}" = fastrpc ]; } &&
       { [ "${prefill}" = qnn-npu ] || [ "${decode}" = qnn-npu ]; }; then
        qnn_env
        hexagon_env
    elif [ "${prefill}" = fastrpc ] || [ "${decode}" = fastrpc ]; then
        fastrpc_env
    elif [ "${prefill}" = qnn-npu ] || [ "${decode}" = qnn-npu ]; then
        qnn_env
    else
        disable_qnn_env
    fi
}

dynamic_env() {
    local prefill=$1
    local decode=$2
    local profile=$3
    printf ' && export GGML_HETERO_DYNAMIC_MODE=phase'
    printf ' && export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=%s' "${prefill}"
    printf ' && export GGML_HETERO_DYNAMIC_DECODE_ROUTE=%s' "${decode}"
    printf ' && export GGML_HETERO_DYNAMIC_TRACE=1'
    printf ' && export GGML_HETERO_DYNAMIC_TRACE_TIMING=1'
    printf ' && export GGML_HETERO_TRACE_SHARE=1'
    printf ' && export GGML_HETERO_DYNAMIC_PRERESERVE=1'
    printf ' && export GGML_HETERO_TRACE_ROUTE_BACKEND=1'
    printf ' && export GGML_HETERO_PROFILE=1'
    printf ' && export GGML_HETERO_PROFILE_SYNC=1'
    printf ' && export GGML_HETERO_PROFILE_FLUSH=1'
    printf ' && export GGML_HETERO_PROFILE_CSV=%s' "${profile}"
}

route_extra_env() {
    local prefill=$1
    local decode=$2
    if { [ "${prefill}" = qnn-npu ] && [ "${decode}" = opencl ]; }; then
        printf ' && export GGML_HETERO_QNN_SHARED_HOST=1'
        printf ' && export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1'
    fi
    if { [ "${prefill}" = cpu ] && [ "${decode}" = opencl ]; } ||
       { [ "${prefill}" = opencl ] && [ "${decode}" = cpu ]; }; then
        printf ' && export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1'
        printf ' && export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1'
    fi
}

canonical_backend() {
    case "$1" in
        cpu|none) printf 'cpu' ;;
        opencl|gpu|GPUOpenCL) printf 'opencl' ;;
        qnn|qnn-npu|npu) printf 'qnn-npu' ;;
        fastrpc|hexagon|htp|HTP|htp0|HTP0) printf 'fastrpc' ;;
        *) printf '%s' "$1" ;;
    esac
}

route_name() {
    canonical_backend "$1"
}

device_name_for_backend() {
    case "$1" in
        cpu) printf 'none' ;;
        opencl) printf 'GPUOpenCL' ;;
        qnn-npu) printf 'qnn-npu' ;;
        fastrpc) printf '%s' "${FASTRPC_DEVICE}" ;;
        *) return 1 ;;
    esac
}

route_args() {
    local prefill=$1
    local decode=$2
    local prefill_dev
    local decode_dev
    prefill_dev=$(device_name_for_backend "${prefill}")
    decode_dev=$(device_name_for_backend "${decode}")

    if [ "${prefill}" = cpu ] && [ "${decode}" = cpu ]; then
        printf -- '-ngl 0 -dev none'
    elif [ "${prefill_dev}" = "${decode_dev}" ]; then
        printf -- '-ngl 99 -dev %s' "${decode_dev}"
    elif [ "${prefill}" = cpu ]; then
        printf -- '-ngl 99 -dev %s' "${decode_dev}"
    elif [ "${decode}" = cpu ]; then
        printf -- '-ngl 99 -dev %s' "${prefill_dev}"
    else
        printf -- '-ngl 99 -dev %s/%s' "${prefill_dev}" "${decode_dev}"
    fi
}

taskset_for_route() {
    if [ "$1" = cpu ] && [ "$2" = cpu ]; then
        printf 'C0'
    elif [ "$1" = fastrpc ] || [ "$2" = fastrpc ]; then
        printf '%s' "${FASTRPC_TASKSET}"
    else
        printf '80'
    fi
}

threads_for_route() {
    if [ "$1" = fastrpc ] || [ "$2" = fastrpc ]; then
        printf '%s' "${FASTRPC_THREADS}"
    else
        printf '%s' "${THREADS}"
    fi
}

bench_batch_for_workload() {
    local pp=$1
    if [ "${pp}" -gt 0 ]; then
        printf '%s' "${pp}"
    else
        printf '512'
    fi
}

flash_attn_args() {
    if [ -n "${FLASH_ATTN}" ]; then
        printf -- '-fa %s' "${FLASH_ATTN}"
    fi
}

workload_name() {
    printf 'pp%s_tg%s' "$1" "$2"
}

write_summary() {
    python3 - "${LOCAL_ROOT}" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
cases = root / "cases.tsv"
summary = root / "summary"
summary.mkdir(parents=True, exist_ok=True)

def read_text(path):
    try:
        return pathlib.Path(path).read_text(errors="replace")
    except (FileNotFoundError, IsADirectoryError, OSError, TypeError):
        return ""

def read_exit(path):
    text = read_text(path).strip()
    return text if text else "missing"

def parse_json_rows(path):
    text = read_text(path).strip()
    if not text:
        return []
    candidates = [text]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            return [data]

    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            rows.append(data)
    return rows

def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)

def int_or_none(value):
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None

def float_or_none(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def fmt_int(value):
    return "" if value is None else str(int(value))

def fmt_float(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"

def fmt_ms_from_ns(value):
    ns = int_or_none(value)
    if ns is None:
        return ""
    return f"{ns / 1_000_000:.3f}"

def fmt_tokens_per_s_from_ns(ns):
    ns = int_or_none(ns)
    if ns is None or ns <= 0:
        return ""
    return f"{1_000_000_000 / ns:.6f}"

def canonical_backend(value):
    value = (value or "").strip().lower()
    if value in ("", "none"):
        return value
    if value in ("gpuopencl", "gpu", "opencl"):
        return "opencl"
    if value in ("qnn", "qnn-npu", "npu"):
        return "qnn-npu"
    if value in ("fastrpc", "hexagon", "htp", "htp0"):
        return "fastrpc"
    if value == "cpu":
        return "cpu"
    return value

def route_label(row):
    category = row.get("category", "")
    prefill = row.get("prefill", "")
    decode = row.get("decode", "")
    if category in ("switch", "no_switch_baseline") and (prefill or decode):
        return f"{prefill}->{decode}"
    return row.get("backend", "")

def decode_backend(row):
    return canonical_backend(row.get("decode") or row.get("backend") or "")

def pp_tg_key(row):
    return (decode_backend(row), row.get("pp", ""), row.get("tg", ""))

case_rows = []
if cases.exists():
    with cases.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["rc"] = read_exit(row.get("exit", ""))
            json_rows = parse_json_rows(row.get("stdout", ""))
            if not json_rows:
                case_rows.append({
                    **row,
                    "json_row_index": "",
                    "devices": "",
                    "n_prompt": "",
                    "n_gen": "",
                    "avg_ns": "",
                    "avg_ts": "",
                    "stddev_ns": "",
                    "stddev_ts": "",
                    "avg_pp_ns": "",
                    "avg_pp_ts": "",
                    "avg_tg_ns": "",
                    "avg_tg_ts": "",
                    "avg_tg_first_ns": "",
                    "avg_tg_first_ts": "",
                    "avg_tg_steady_ns": "",
                    "avg_tg_steady_ts": "",
                    "avg_tg_route_ns": "",
                    "avg_tg_kv_ns": "",
                    "avg_tg_reserve_ns": "",
                })
                continue
            for i, out in enumerate(json_rows):
                merged = {**row, "json_row_index": str(i)}
                for key in (
                    "devices", "n_prompt", "n_gen", "avg_ns", "avg_ts",
                    "stddev_ns", "stddev_ts", "avg_pp_ns", "avg_pp_ts",
                    "avg_tg_ns", "avg_tg_ts", "avg_tg_first_ns",
                    "avg_tg_first_ts", "avg_tg_steady_ns", "avg_tg_steady_ts",
                    "avg_tg_route_ns", "avg_tg_kv_ns", "avg_tg_reserve_ns",
                ):
                    merged[key] = to_text(out.get(key, ""))
                case_rows.append(merged)

def enrich(row):
    route_ns = int_or_none(row.get("avg_tg_route_ns"))
    kv_ns = int_or_none(row.get("avg_tg_kv_ns"))
    reserve_ns = int_or_none(row.get("avg_tg_reserve_ns"))
    first_ns = int_or_none(row.get("avg_tg_first_ns"))
    timed = None
    compute = None
    if route_ns is not None and kv_ns is not None and reserve_ns is not None:
        timed = route_ns + kv_ns + reserve_ns
        if first_ns is not None:
            compute = first_ns - timed
    row["decode_backend"] = decode_backend(row)
    row["route"] = route_label(row)
    row["avg_ms"] = fmt_ms_from_ns(row.get("avg_ns"))
    row["avg_pp_ms"] = fmt_ms_from_ns(row.get("avg_pp_ns"))
    row["avg_tg_ms"] = fmt_ms_from_ns(row.get("avg_tg_ns"))
    row["avg_tg_first_ms"] = fmt_ms_from_ns(row.get("avg_tg_first_ns"))
    row["avg_tg_steady_ms"] = fmt_ms_from_ns(row.get("avg_tg_steady_ns"))
    row["avg_tg_route_ms"] = fmt_ms_from_ns(row.get("avg_tg_route_ns"))
    row["avg_tg_kv_ms"] = fmt_ms_from_ns(row.get("avg_tg_kv_ns"))
    row["avg_tg_reserve_ms"] = fmt_ms_from_ns(row.get("avg_tg_reserve_ns"))
    row["tg_first_switch_timed_ns"] = fmt_int(timed)
    row["tg_first_switch_timed_ms"] = fmt_ms_from_ns(timed)
    row["tg_first_compute_ns"] = fmt_int(compute)
    row["tg_first_compute_ms"] = fmt_ms_from_ns(compute)
    row["tg_first_compute_ts"] = fmt_tokens_per_s_from_ns(compute)
    row["baseline_case"] = ""
    row["baseline_tg_first_ns"] = ""
    row["baseline_tg_first_ms"] = ""
    row["switch_overhead_ns"] = ""
    row["switch_overhead_ms"] = ""
    row["switch_overhead_residual_ns"] = ""
    row["switch_overhead_residual_ms"] = ""
    row["switch_overhead_note"] = ""
    return row

timing_rows = [enrich(dict(row)) for row in case_rows]

baseline_by_key = {}
for row in timing_rows:
    if row.get("category") not in ("no_switch_baseline", "single_backend"):
        continue
    if int_or_none(row.get("avg_tg_first_ns")) is None:
        continue
    key = pp_tg_key(row)
    existing = baseline_by_key.get(key)
    if existing is None or (
        existing.get("category") != "no_switch_baseline"
        and row.get("category") == "no_switch_baseline"
    ):
        baseline_by_key[key] = row

for row in timing_rows:
    if row.get("category") != "switch":
        continue
    first_ns = int_or_none(row.get("avg_tg_first_ns"))
    baseline = baseline_by_key.get(pp_tg_key(row))
    if baseline is None:
        row["switch_overhead_note"] = "missing same decode backend pp/tg no-switch baseline"
        continue
    baseline_first_ns = int_or_none(baseline.get("avg_tg_first_ns"))
    row["baseline_case"] = baseline.get("case", "")
    row["baseline_tg_first_ns"] = fmt_int(baseline_first_ns)
    row["baseline_tg_first_ms"] = fmt_ms_from_ns(baseline_first_ns)
    if first_ns is None or baseline_first_ns is None:
        row["switch_overhead_note"] = "missing first-token timing"
        continue
    overhead = first_ns - baseline_first_ns
    timed = int_or_none(row.get("tg_first_switch_timed_ns"))
    residual = overhead - timed if timed is not None else None
    row["switch_overhead_ns"] = fmt_int(overhead)
    row["switch_overhead_ms"] = fmt_ms_from_ns(overhead)
    row["switch_overhead_residual_ns"] = fmt_int(residual)
    row["switch_overhead_residual_ms"] = fmt_ms_from_ns(residual)
    row["switch_overhead_note"] = "same decode backend and pp/tg baseline"

def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)

def markdown_cell(value):
    text = str(value if value is not None else "")
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

def write_markdown(path, rows, fields):
    with path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("| " + " | ".join("---" for _ in fields) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(markdown_cell(row.get(k, "")) for k in fields) + " |\n")

speed_fields = [
    "case", "category", "route", "workload", "rc", "devices",
    "n_prompt", "n_gen", "avg_ms", "avg_ts",
    "avg_pp_ms", "avg_pp_ts", "avg_tg_ms", "avg_tg_ts",
    "avg_tg_first_ms", "avg_tg_first_ts",
    "avg_tg_steady_ms", "avg_tg_steady_ts",
    "avg_tg_route_ms", "avg_tg_kv_ms", "avg_tg_reserve_ms",
    "tg_first_compute_ms", "tg_first_compute_ts",
    "switch_overhead_ms", "baseline_case",
]

timing_fields = [
    "case", "category", "prefill", "decode", "decode_backend", "workload", "pp", "tg", "rc", "devices",
    "avg_tg_first_ns", "avg_tg_steady_ns",
    "avg_tg_route_ns", "avg_tg_kv_ns", "avg_tg_reserve_ns",
    "tg_first_switch_timed_ns", "tg_first_compute_ns", "tg_first_compute_ts",
    "baseline_case", "baseline_tg_first_ns",
    "switch_overhead_ns", "switch_overhead_ms",
    "switch_overhead_residual_ns", "switch_overhead_residual_ms",
    "switch_overhead_note",
    "avg_ns", "avg_ts", "avg_pp_ns", "avg_pp_ts", "avg_tg_ns", "avg_tg_ts",
    "stdout", "stderr", "exit",
]

write_csv(summary / "speed_summary.csv", timing_rows, speed_fields)
write_markdown(summary / "speed_summary.md", timing_rows, speed_fields)
write_csv(summary / "timing_summary.csv", timing_rows, timing_fields)
write_markdown(summary / "timing_summary.md", timing_rows, timing_fields)

switch_rows = [r for r in timing_rows if r.get("category") == "switch"]
baseline_rows = [r for r in timing_rows if r.get("category") == "no_switch_baseline"]
write_csv(summary / "switch_timing.csv", switch_rows, timing_fields)
write_markdown(summary / "switch_timing.md", switch_rows, timing_fields)
write_csv(summary / "no_switch_baseline.csv", baseline_rows, timing_fields)
write_markdown(summary / "no_switch_baseline.md", baseline_rows, timing_fields)

failures = [r for r in timing_rows if r.get("rc") != "0"]
write_csv(summary / "failures.csv", failures, speed_fields)
write_markdown(summary / "failures.md", failures, speed_fields)

phase_re = re.compile(
    r"maybe_apply_dynamic_route: timing phase=(?P<phase>\S+) n_tokens=(?P<n_tokens>\d+) "
    r"route_apply=(?P<route_apply>\S+)(?: label=(?P<label>\S+) reason=(?P<reason>\S+) "
    r"decide_us=(?P<decide_us>\d+) apply_us=(?P<apply_us>\d+) target=(?P<target>.*))?"
)
phase_rows = []
for row in timing_rows:
    if row.get("json_row_index") not in ("", "0"):
        continue
    if not row.get("stderr"):
        continue
    stderr = pathlib.Path(row.get("stderr", ""))
    if not stderr.exists():
        continue
    for line_no, line in enumerate(stderr.read_text(errors="replace").splitlines(), 1):
        m = phase_re.search(line)
        if not m:
            continue
        gd = m.groupdict()
        phase_rows.append({
            "case": row.get("case", ""),
            "category": row.get("category", ""),
            "prefill": row.get("prefill", ""),
            "decode": row.get("decode", ""),
            "workload": row.get("workload", ""),
            "pp": row.get("pp", ""),
            "tg": row.get("tg", ""),
            "line": line_no,
            "phase": gd.get("phase") or "",
            "n_tokens": gd.get("n_tokens") or "",
            "route_apply": gd.get("route_apply") or "",
            "label": gd.get("label") or "",
            "reason": gd.get("reason") or "",
            "decide_us": gd.get("decide_us") or "",
            "apply_us": gd.get("apply_us") or "",
            "target": gd.get("target") or "",
        })

phase_fields = [
    "case", "category", "prefill", "decode", "workload", "pp", "tg", "line",
    "phase", "n_tokens", "route_apply", "label", "reason", "decide_us", "apply_us", "target",
]
write_csv(summary / "phase_timing.csv", phase_rows, phase_fields)

with (summary / "README.md").open("w") as f:
    f.write("# PD div timing summary\n\n")
    f.write(f"Local root: `{root}`\n\n")
    f.write("This runner invokes `llama-bench` with `--bench-timing-level 3 -o json` by default.\n\n")
    f.write("Important fields:\n\n")
    f.write("- `avg_tg_first_ns`: wall time of the first generated/decode token in llama-bench.\n")
    f.write("- `avg_tg_steady_ns`: average steady decode token time after the first token.\n")
    f.write("- `avg_tg_route_ns`, `avg_tg_kv_ns`, `avg_tg_reserve_ns`: level-3 switch-token subcomponents reported by llama-bench.\n")
    f.write("- `tg_first_compute_ns = avg_tg_first_ns - avg_tg_route_ns - avg_tg_kv_ns - avg_tg_reserve_ns`.\n")
    f.write("- `tg_first_compute_ts = 1e9 / tg_first_compute_ns`; this is a one-token derived speed after subtracting the reported route/KV/reserve components, not a hardware-kernel-only metric.\n")
    f.write("- `switch_overhead_ns = switched avg_tg_first_ns - same decode backend no-switch baseline avg_tg_first_ns`, matched by decode backend, `pp`, and `tg`.\n")
    f.write("- `switch_overhead_residual_ns = switch_overhead_ns - (avg_tg_route_ns + avg_tg_kv_ns + avg_tg_reserve_ns)`; use it as a rough consistency check, not as a separate measured phase.\n\n")
    f.write("Generated files:\n\n")
    f.write("- `speed_summary.csv` / `.md`: compact human-readable speed and timing table.\n")
    f.write("- `timing_summary.csv` / `.md`: full timing table for every case.\n")
    f.write("- `switch_timing.csv` / `.md`: switch cases with no-switch baseline overhead estimates.\n")
    f.write("- `no_switch_baseline.csv` / `.md`: same-backend dynamic-route baseline rows.\n")
    f.write("- `phase_timing.csv`: parsed stderr route timing lines, when present.\n")
    f.write("- `failures.csv` / `.md`: nonzero or missing exit-code cases.\n")
    f.write("\nRaw stdout/stderr/exit/profile/command/env files are retained by default. Set `KEEP_RAW=0` only for disposable local runs.\n")

print(root)
PY
}

cleanup_run_artifacts() {
    if [ "${SUMMARY_ONLY}" -eq 0 ] && [ "${KEEP_RAW}" != "1" ]; then
        rm -rf "${RAW_DIR}"
        rm -f "${COMMANDS}" "${CASES}"
    fi
}

finish_summary() {
    set +e
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt" 2> "${SUMMARY_DIR}/summary.err"
    local summary_rc=$?
    if [ "${summary_rc}" -eq 0 ]; then
        cleanup_run_artifacts
    fi
    return 0
}

if [ "${SUMMARY_ONLY}" -eq 1 ]; then
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt" 2> "${SUMMARY_DIR}/summary.err"
    exit 0
fi

trap finish_summary EXIT

record_case() {
    local name=$1
    local category=$2
    local backend=$3
    local prefill=$4
    local decode=$5
    local workload=$6
    local pp=$7
    local tg=$8
    local out=$9
    local err=${10}
    local exit_file=${11}
    local profile=${12}
    local profile_pull=${13}
    local command_file=${14}
    local env_file=${15}

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${category}" "${backend}" "${prefill}" "${decode}" "${workload}" "${pp}" "${tg}" \
        "${out}" "${err}" "${exit_file}" "${profile}" "${profile_pull}" "${command_file}" "${env_file}" >> "${CASES}"
}

cooldown() {
    if [ "${COOLDOWN_SEC}" -gt 0 ]; then
        printf '[%s] cooldown %ss\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${COOLDOWN_SEC}" >&2
        sleep "${COOLDOWN_SEC}"
    fi
}

run_remote() {
    local name=$1
    local category=$2
    local backend=$3
    local prefill=$4
    local decode=$5
    local workload=$6
    local pp=$7
    local tg=$8
    local profile=$9
    local remote_cmd=${10}

    local out="${RAW_DIR}/${name}.stdout"
    local err="${RAW_DIR}/${name}.stderr"
    local exit_file="${RAW_DIR}/${name}.exit"
    local profile_pull="${RAW_DIR}/${name}.profile.pull.log"
    local command_file="${RAW_DIR}/${name}.command"
    local env_file="${RAW_DIR}/${name}.env"

    printf '[%s] start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" >&2
    printf '%s\n' "timeout ${CASE_TIMEOUT_SEC} adb -s ${DEVICE} shell '${remote_cmd}'" > "${command_file}"
    printf '%s\n' "timeout ${CASE_TIMEOUT_SEC} adb -s ${DEVICE} shell '${remote_cmd}'" >> "${COMMANDS}"
    printf '%s\n' "${remote_cmd}" | sed 's/ && /\n/g' | sed -n 's/^export //p' > "${env_file}"

    set +e
    timeout "${CASE_TIMEOUT_SEC}" adb -s "${DEVICE}" shell "${remote_cmd}" > "${out}" 2> "${err}"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${exit_file}"

    if [ -n "${profile}" ]; then
        set +e
        adb -s "${DEVICE}" pull "${profile}" "${RAW_DIR}/${name}.profile.csv" > "${profile_pull}" 2>&1
        set -e
    else
        printf 'no profile requested\n' > "${profile_pull}"
    fi

    record_case "${name}" "${category}" "${backend}" "${prefill}" "${decode}" "${workload}" \
        "${pp}" "${tg}" "${out}" "${err}" "${exit_file}" "${profile}" "${profile_pull}" "${command_file}" "${env_file}"
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"

    printf '[%s] finish %s rc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${rc}" >&2
    cooldown
}

build_bench_cmd() {
    local prefill_route=$1
    local decode_route=$2
    local pp=$3
    local tg=$4
    local profile=$5

    local args
    args=$(route_args "${prefill_route}" "${decode_route}")
    local mask
    mask=$(taskset_for_route "${prefill_route}" "${decode_route}")
    local threads
    threads=$(threads_for_route "${prefill_route}" "${decode_route}")
    local bench_batch
    bench_batch=$(bench_batch_for_workload "${pp}")
    local fa_args
    fa_args=$(flash_attn_args)

    local cmd
    cmd="$(remote_base_env)"
    cmd="${cmd}$(route_env "${prefill_route}" "${decode_route}")"
    cmd="${cmd}$(route_extra_env "${prefill_route}" "${decode_route}")"
    cmd="${cmd} && rm -f ${profile}"
    cmd="${cmd}$(dynamic_env "${prefill_route}" "${decode_route}" "${profile}")"
    cmd="${cmd} && taskset ${mask} ./llama-bench -v -r ${REPS} --bench-timing-level ${BENCH_TIMING_LEVEL} -o ${BENCH_OUTPUT_FORMAT} -m ${MODEL_PATH} ${args} -t ${threads} -c ${CTX_SIZE} -b ${bench_batch} -ub ${bench_batch} -p 0 -n 0 -pg ${pp},${tg} --no-warmup --mmap 0 ${fa_args}"
    printf '%s' "${cmd}"
}

run_no_switch_baseline() {
    local backend=$1
    local pp=$2
    local tg=$3
    local route
    route=$(route_name "${backend}")
    local workload
    workload=$(workload_name "${pp}" "${tg}")
    local name="baseline_${route}_${workload}"
    local profile="${REMOTE_BIN_DIR}/results/${name}.profile.csv"
    local cmd
    cmd=$(build_bench_cmd "${route}" "${route}" "${pp}" "${tg}" "${profile}")
    run_remote "${name}" no_switch_baseline "${route}" "${route}" "${route}" "${workload}" "${pp}" "${tg}" "${profile}" "${cmd}"
}

run_single_backend() {
    local backend=$1
    local pp=$2
    local tg=$3
    local route
    route=$(route_name "${backend}")
    local workload
    workload=$(workload_name "${pp}" "${tg}")
    local name="single_${route}_${workload}"
    local profile="${REMOTE_BIN_DIR}/results/${name}.profile.csv"
    local cmd
    cmd=$(build_bench_cmd "${route}" "${route}" "${pp}" "${tg}" "${profile}")
    run_remote "${name}" single_backend "${route}" "${route}" "${route}" "${workload}" "${pp}" "${tg}" "${profile}" "${cmd}"
}

run_switch() {
    local prefill_backend=$1
    local decode_backend=$2
    local pp=$3
    local tg=$4
    local prefill_route
    local decode_route
    prefill_route=$(route_name "${prefill_backend}")
    decode_route=$(route_name "${decode_backend}")

    local workload
    workload=$(workload_name "${pp}" "${tg}")
    local name="switch_${prefill_route}_to_${decode_route}_${workload}"
    local profile="${REMOTE_BIN_DIR}/results/${name}.profile.csv"
    local cmd
    cmd=$(build_bench_cmd "${prefill_route}" "${decode_route}" "${pp}" "${tg}" "${profile}")

    run_remote "${name}" switch "" "${prefill_route}" "${decode_route}" "${workload}" "${pp}" "${tg}" "${profile}" "${cmd}"
}

backend_in_list() {
    local needle
    needle=$(route_name "$1")
    local backend
    for backend in "${backends[@]}"; do
        if [ "$(route_name "${backend}")" = "${needle}" ]; then
            return 0
        fi
    done
    return 1
}

should_run_switch() {
    local prefill
    local decode
    prefill=$(route_name "$1")
    decode=$(route_name "$2")
    if [ "${prefill}" = "${decode}" ]; then
        return 1
    fi
    if [ "${decode}" = cpu ] && [ "${RUN_CPU_DECODE}" != "1" ]; then
        return 1
    fi
    return 0
}

required_devices() {
    local seen=""
    local backend
    for backend in "${backends[@]}"; do
        local route
        route=$(route_name "${backend}")
        case "${route}" in
            opencl)
                if [[ " ${seen} " != *" GPUOpenCL "* ]]; then
                    printf '%s\n' 'GPUOpenCL'
                    seen="${seen} GPUOpenCL"
                fi
                ;;
            qnn-npu)
                if [[ " ${seen} " != *" qnn-npu "* ]]; then
                    printf '%s\n' 'qnn-npu'
                    seen="${seen} qnn-npu"
                fi
                ;;
            fastrpc)
                if [[ " ${seen} " != *" ${FASTRPC_DEVICE} "* ]]; then
                    printf '%s\n' "${FASTRPC_DEVICE}"
                    seen="${seen} ${FASTRPC_DEVICE}"
                fi
                ;;
        esac
    done
}

preflight() {
    adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
    adb -s "${DEVICE}" shell "test -f ${MODEL_PATH}"
    if backend_in_list qnn-npu; then
        adb -s "${DEVICE}" shell "test -f ${QNN_DIR}/config.json"
    fi

    adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} && export GGML_HEXAGON_EXPERIMENTAL=1 && ./llama-bench --list-devices" \
        > "${LOCAL_ROOT}/list-devices.stdout" 2> "${LOCAL_ROOT}/list-devices.stderr"

    local dev
    while IFS= read -r dev; do
        if [ -n "${dev}" ] && ! grep -q "${dev}" "${LOCAL_ROOT}/list-devices.stdout"; then
            printf 'error: device list does not include %s; see %s\n' "${dev}" "${LOCAL_ROOT}/list-devices.stdout" >&2
            exit 1
        fi
    done < <(required_devices)

    adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && sha256sum llama-bench libllama.so libggml.so libggml-opencl.so libggml-qnn.so libggml-hexagon.so 2>/dev/null" \
        > "${LOCAL_ROOT}/remote-sha256.txt" 2> "${LOCAL_ROOT}/remote-sha256.stderr" || true

    {
        printf 'device=%s\n' "${DEVICE}"
        printf 'remote_bin_dir=%s\n' "${REMOTE_BIN_DIR}"
        printf 'model_path=%s\n' "${MODEL_PATH}"
        printf 'qnn_dir=%s\n' "${QNN_DIR}"
        printf 'local_root=%s\n' "${LOCAL_ROOT}"
        printf 'case_timeout_sec=%s\n' "${CASE_TIMEOUT_SEC}"
        printf 'cooldown_sec=%s\n' "${COOLDOWN_SEC}"
        printf 'reps=%s\n' "${REPS}"
        printf 'bench_timing_level=%s\n' "${BENCH_TIMING_LEVEL}"
        printf 'bench_output_format=%s\n' "${BENCH_OUTPUT_FORMAT}"
        printf 'ctx_size=%s\n' "${CTX_SIZE}"
        printf 'threads=%s\n' "${THREADS}"
        printf 'fastrpc_threads=%s\n' "${FASTRPC_THREADS}"
        printf 'flash_attn=%s\n' "${FLASH_ATTN}"
        printf 'backends=%s\n' "${backends[*]}"
        printf 'switch_workloads=%s\n' "${switch_workloads[*]}"
        printf 'run_cpu_decode=%s\n' "${RUN_CPU_DECODE}"
        printf 'run_single_backend=%s\n' "${RUN_SINGLE_BACKEND}"
        printf 'single_workloads=%s\n' "${single_workloads[*]}"
        printf 'fastrpc_device=%s\n' "${FASTRPC_DEVICE}"
        printf 'fastrpc_pd_hostbuf=%s\n' "${FASTRPC_PD_HOSTBUF}"
        printf 'fastrpc_pd_use_hmx=%s\n' "${FASTRPC_PD_USE_HMX}"
        printf 'fastrpc_pd_nhvx=%s\n' "${FASTRPC_PD_NHVX}"
    } > "${MANIFEST}"
}

run_baselines_for_switches() {
    local backend
    local workload
    for backend in "${backends[@]}"; do
        local route
        route=$(route_name "${backend}")
        if [ "${route}" = cpu ] && [ "${RUN_CPU_DECODE}" != "1" ]; then
            continue
        fi
        for workload in "${switch_workloads[@]}"; do
            IFS=: read -r pp tg <<< "${workload}"
            run_no_switch_baseline "${route}" "${pp}" "${tg}"
        done
    done
}

run_single_backend_matrix() {
    local backend
    local workload
    for backend in "${backends[@]}"; do
        for workload in "${single_workloads[@]}"; do
            IFS=: read -r pp tg <<< "${workload}"
            run_single_backend "${backend}" "${pp}" "${tg}"
        done
    done
}

run_switch_matrix() {
    local prefill
    local decode
    local workload
    for prefill in "${backends[@]}"; do
        for decode in "${backends[@]}"; do
            if ! should_run_switch "${prefill}" "${decode}"; then
                continue
            fi
            for workload in "${switch_workloads[@]}"; do
                IFS=: read -r pp tg <<< "${workload}"
                run_switch "${prefill}" "${decode}" "${pp}" "${tg}"
            done
        done
    done
}

main() {
    preflight
    run_baselines_for_switches
    if [ "${RUN_SINGLE_BACKEND}" = "1" ]; then
        run_single_backend_matrix
    fi
    run_switch_matrix
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"
    printf 'LOCAL_ROOT=%s\n' "${LOCAL_ROOT}"
}

main "$@"
