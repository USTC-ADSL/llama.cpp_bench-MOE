#!/usr/bin/env bash
set -Eeuo pipefail

DEVICE=${DEVICE:-fd8657d6}
REMOTE_BIN_DIR=${REMOTE_BIN_DIR:-/data/local/tmp/bench-PD}
MODEL_PATH=${MODEL_PATH:-/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf}
QNN_DIR=${QNN_DIR:-/data/local/tmp/models/Qwen2.5-3B-AoT/qnn}
LOCAL_ROOT=${LOCAL_ROOT:-results/bench-PD-fd-$(date -u +%Y%m%d-%H%M)}
COOLDOWN_SEC=${COOLDOWN_SEC:-120}
CASE_TIMEOUT_SEC=${CASE_TIMEOUT_SEC:-1800}
REPS=${REPS:-1}
KEEP_RAW=${KEEP_RAW:-0}

SUMMARY_ONLY=0
if [ "${1:-}" = "--summary-only" ]; then
    SUMMARY_ONLY=1
    if [ -n "${2:-}" ]; then
        LOCAL_ROOT=$2
    fi
fi

RAW_DIR="${LOCAL_ROOT}/raw"
SUMMARY_DIR="${LOCAL_ROOT}/summary"
COMMANDS="${LOCAL_ROOT}/commands.sh"
CASES="${LOCAL_ROOT}/cases.tsv"
MANIFEST="${LOCAL_ROOT}/manifest.txt"

if [ "${SUMMARY_ONLY}" -eq 0 ]; then
    mkdir -p "${RAW_DIR}" "${SUMMARY_DIR}"
    printf '#!/usr/bin/env bash\n' > "${COMMANDS}"
    printf 'case\tcategory\tbackend\tprefill\tdecode\tworkload\tpp\ttg\tstdout\tstderr\texit\tprofile\tprofile_pull\n' > "${CASES}"
else
    mkdir -p "${SUMMARY_DIR}"
fi

single_workloads=(
    "128:0"
    "256:0"
    "512:0"
    "0:1"
    "0:32"
    "0:128"
    "0:256"
)

switch_workloads=(
    "256:1"
)

backends=(cpu opencl qnn)

remote_base_env() {
    printf 'cd %s && mkdir -p %s/results && export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=%s && export LLAMA_BENCH_FAST_EXIT=1' \
        "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}"
}

disable_qnn_env() {
    printf ' && export GGML_QNN_DISABLE_BACKEND=1'
    printf ' && export GGML_HETERO_DYNAMIC_ALLOW_QNN=0'
}

qnn_env() {
    printf ' && export GGML_HEXAGON_EXPERIMENTAL=1'
    printf ' && export GGML_QNN_AOT_CONFIG=%s/config.json' "${QNN_DIR}"
    printf ' && export GGML_QNN_AOT_MODEL_DIR=%s' "${QNN_DIR}"
    printf ' && export GGML_QNN_AOT_DISABLE_SEED_KV=1'
    printf ' && export GGML_QNN_AOT_WRITE_GENERIC_KV=1'
    printf ' && export GGML_QNN_AOT_TRACE_ASSIGN=1'
    printf ' && export GGML_QNN_AOT_TRACE_MATCH=1'
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
    printf ' && export GGML_HETERO_PROFILE=1'
    printf ' && export GGML_HETERO_PROFILE_SYNC=1'
    printf ' && export GGML_HETERO_PROFILE_FLUSH=1'
    printf ' && export GGML_HETERO_PROFILE_CSV=%s' "${profile}"
}

backend_args() {
    case "$1" in
        cpu)    printf -- '-ngl 0 -dev none' ;;
        opencl) printf -- '-ngl 99 -dev GPUOpenCL' ;;
        qnn)    printf -- '-ngl 99 -dev qnn-npu' ;;
        *)      return 1 ;;
    esac
}

route_name() {
    case "$1" in
        cpu)    printf 'cpu' ;;
        opencl) printf 'opencl' ;;
        qnn)    printf 'qnn-npu' ;;
        *)      return 1 ;;
    esac
}

route_args() {
    local prefill=$1
    local decode=$2
    if { [ "${prefill}" = qnn-npu ] && [ "${decode}" = opencl ]; } ||
       { [ "${prefill}" = opencl ] && [ "${decode}" = qnn-npu ]; }; then
        printf -- '-ngl 99 -dev qnn-npu,GPUOpenCL'
    elif [ "${prefill}" = qnn-npu ] || [ "${decode}" = qnn-npu ]; then
        printf -- '-ngl 99 -dev qnn-npu'
    elif [ "${prefill}" = opencl ] || [ "${decode}" = opencl ]; then
        printf -- '-ngl 99 -dev GPUOpenCL'
    else
        printf -- '-ngl 0 -dev none'
    fi
}

taskset_for_backend() {
    case "$1" in
        cpu) printf 'C0' ;;
        *)   printf '80' ;;
    esac
}

taskset_for_route() {
    if [ "$1" = cpu ] && [ "$2" = cpu ]; then
        printf 'C0'
    else
        printf '80'
    fi
}

workload_name() {
    printf 'pp%s_tg%s' "$1" "$2"
}

write_summary() {
    python3 - "${LOCAL_ROOT}" <<'PY'
import csv
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
    except FileNotFoundError:
        return ""

def read_exit(path):
    text = read_text(path).strip()
    return text if text else "missing"

def csv_rows(path):
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    with p.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))

case_rows = []
if cases.exists():
    with cases.open(newline="", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["rc"] = read_exit(row["exit"])
            rows = csv_rows(row["stdout"])
            if not rows:
                case_rows.append({
                    **row,
                    "csv_row_index": "",
                    "devices": "",
                    "n_prompt": "",
                    "n_gen": "",
                    "avg_ns": "",
                    "avg_ts": "",
                    "stddev_ns": "",
                    "stddev_ts": "",
                })
                continue
            for i, out in enumerate(rows):
                case_rows.append({
                    **row,
                    "csv_row_index": str(i),
                    "devices": out.get("devices", ""),
                    "n_prompt": out.get("n_prompt", ""),
                    "n_gen": out.get("n_gen", ""),
                    "avg_ns": out.get("avg_ns", ""),
                    "avg_ts": out.get("avg_ts", ""),
                    "stddev_ns": out.get("stddev_ns", ""),
                    "stddev_ts": out.get("stddev_ts", ""),
                })

def fmt_ms(ns_text):
    try:
        return f"{int(ns_text) / 1_000_000:.3f}"
    except (TypeError, ValueError):
        return ""

def route_label(row):
    if row.get("category") == "switch":
        prefill = row.get("prefill", "")
        decode = row.get("decode", "")
        if prefill or decode:
            return f"{prefill}->{decode}"
    return row.get("backend", "")

speed_fields = [
    "route", "workload", "rc", "devices",
    "n_prompt", "n_gen", "avg_ms", "tokens/s",
]

speed_rows = []
for row in case_rows:
    speed_rows.append({
        "case": row.get("case", ""),
        "category": row.get("category", ""),
        "route": route_label(row),
        "workload": row.get("workload", ""),
        "rc": row.get("rc", ""),
        "devices": row.get("devices", ""),
        "n_prompt": row.get("n_prompt", ""),
        "n_gen": row.get("n_gen", ""),
        "avg_ms": fmt_ms(row.get("avg_ns", "")),
        "tokens/s": row.get("avg_ts", ""),
    })

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

def write_speed_table(stem, rows):
    write_csv(summary / f"{stem}.csv", rows, speed_fields)
    write_markdown(summary / f"{stem}.md", rows, speed_fields)

write_speed_table("speed_summary", speed_rows)
for obsolete in ("all_results.csv", "all_results.md"):
    try:
        (summary / obsolete).unlink()
    except FileNotFoundError:
        pass
write_speed_table("single_backend", [r for r in speed_rows if r.get("category") == "single_backend"])
write_speed_table("switch_pp512", [r for r in speed_rows if r.get("category") == "switch"])
write_speed_table("failures", [r for r in speed_rows if r.get("rc") != "0"])

phase_re = re.compile(
    r"maybe_apply_dynamic_route: timing phase=(?P<phase>\S+) n_tokens=(?P<n_tokens>\d+) "
    r"route_apply=(?P<route_apply>\S+)(?: label=(?P<label>\S+) reason=(?P<reason>\S+) "
    r"decide_us=(?P<decide_us>\d+) apply_us=(?P<apply_us>\d+) target=(?P<target>.*))?"
)
phase_rows = []
for row in case_rows:
    if row.get("csv_row_index") not in ("", "0"):
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
with (summary / "phase_timing.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=phase_fields)
    writer.writeheader()
    writer.writerows(phase_rows)

with (summary / "README.md").open("w") as f:
    f.write("# bench-PD fd summary\n\n")
    f.write(f"Local root: `{root}`\n\n")
    f.write("Core formal matrix: `cpu`, `opencl`/`GPUOpenCL`, `qnn`/`qnn-npu`.\n\n")
    f.write("FastRPC/HTP0 is not part of this run's success criteria; see `list-devices.stdout` for the observed device list.\n\n")
    f.write("Each invocation uses `-r 1`; the runner sleeps according to `COOLDOWN_SEC` between completed invocations.\n\n")
    f.write("Single-backend workloads: `pp128_tg32`, `pp256_tg1`, `pp256_tg32`, `pp256_tg12`.\n\n")
    f.write("Switch workloads: ordered non-self routes among `cpu`, `opencl`, and `qnn-npu` with `pp512_tg1`, `pp512_tg32`, `pp512_tg128`.\n\n")
    f.write("Files:\n\n")
    f.write("- `speed_summary.csv`: compact speed table for all runs. `tokens/s` is llama-bench `avg_ts`; `avg_ms` is llama-bench `avg_ns` converted to milliseconds.\n")
    f.write("- `speed_summary.md`: same data as `speed_summary.csv`, formatted as a Markdown pipe table.\n")
    f.write("- `single_backend.csv`: compact speed rows for single-backend combined pp/tg runs.\n")
    f.write("- `single_backend.md`: Markdown pipe table for single-backend speed rows.\n")
    f.write("- `switch_pp512.csv`: compact speed rows for non-self phase-route switch runs.\n")
    f.write("- `switch_pp512.md`: Markdown pipe table for non-self phase-route switch rows.\n")
    f.write("- `phase_timing.csv`: parsed `maybe_apply_dynamic_route` timing lines from stderr.\n")
    f.write("- `failures.csv`: rows whose invocation exit code is nonzero or missing.\n")
    f.write("- `failures.md`: Markdown pipe table for failures.\n")
    f.write("\nRaw stdout/stderr/profile files are retained only when the runner is invoked with `KEEP_RAW=1`.\n")

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

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${category}" "${backend}" "${prefill}" "${decode}" "${workload}" "${pp}" "${tg}" \
        "${out}" "${err}" "${exit_file}" "${profile}" "${profile_pull}" >> "${CASES}"
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

    printf '[%s] start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" >&2
    printf '%s\n' "timeout ${CASE_TIMEOUT_SEC} adb -s ${DEVICE} shell '${remote_cmd}'" >> "${COMMANDS}"

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
        "${pp}" "${tg}" "${out}" "${err}" "${exit_file}" "${profile}" "${profile_pull}"
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"

    printf '[%s] finish %s rc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${rc}" >&2
    cooldown
}

run_single_backend() {
    local backend=$1
    local pp=$2
    local tg=$3
    local args
    args=$(backend_args "${backend}")
    local mask
    mask=$(taskset_for_backend "${backend}")
    local workload
    workload=$(workload_name "${pp}" "${tg}")
    local name="single_${backend}_${workload}"
    local cmd
    cmd="$(remote_base_env)"
    if [ "${backend}" = qnn ]; then
        cmd="${cmd}$(qnn_env)"
    else
        cmd="${cmd}$(disable_qnn_env)"
    fi
    cmd="${cmd} && taskset ${mask} ./llama-bench -v -r ${REPS} -o csv -m ${MODEL_PATH} ${args} -t 4 -c 2048 -b ${pp} -ub ${pp} -p 0 -n 0 -pg ${pp},${tg} --no-warmup --mmap 0"
    run_remote "${name}" single_backend "${backend}" "" "" "${workload}" "${pp}" "${tg}" "" "${cmd}"
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
    local args
    args=$(route_args "${prefill_route}" "${decode_route}")
    local mask
    mask=$(taskset_for_route "${prefill_backend}" "${decode_backend}")
    local workload
    workload=$(workload_name "${pp}" "${tg}")
    local name="switch_${prefill_backend}_to_${decode_backend}_${workload}"
    local profile="${REMOTE_BIN_DIR}/results/${name}.profile.csv"
    local cmd
    cmd="$(remote_base_env)"
    if [ "${prefill_backend}" = qnn ] || [ "${decode_backend}" = qnn ]; then
        cmd="${cmd}$(qnn_env)"
    else
        cmd="${cmd}$(disable_qnn_env)"
    fi
    if { [ "${prefill_backend}" = qnn ] && [ "${decode_backend}" = opencl ]; }; then
        cmd="${cmd} && export GGML_HETERO_QNN_SHARED_HOST=1 && export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1"
    fi
    if { [ "${prefill_backend}" = cpu ] && [ "${decode_backend}" = opencl ]; } ||
       { [ "${prefill_backend}" = opencl ] && [ "${decode_backend}" = cpu ]; }; then
        cmd="${cmd} && export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1 && export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1"
    fi
    cmd="${cmd} && rm -f ${profile}"
    cmd="${cmd}$(dynamic_env "${prefill_route}" "${decode_route}" "${profile}")"
    cmd="${cmd} && taskset ${mask} ./llama-bench -v -r ${REPS} -o csv -m ${MODEL_PATH} ${args} -t 4 -c 2048 -b ${pp} -ub ${pp} -p 0 -n 0 -pg ${pp},${tg} --no-warmup --mmap 0"
    run_remote "${name}" switch "" "${prefill_route}" "${decode_route}" "${workload}" "${pp}" "${tg}" "${profile}" "${cmd}"
}

preflight() {
    adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
    adb -s "${DEVICE}" shell "test -f ${MODEL_PATH}"
    adb -s "${DEVICE}" shell "test -f ${QNN_DIR}/config.json"
    adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH && ./llama-bench --list-devices" \
        > "${LOCAL_ROOT}/list-devices.stdout" 2> "${LOCAL_ROOT}/list-devices.stderr"
    adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && sha256sum llama-bench libllama.so libggml.so libggml-opencl.so libggml-qnn.so 2>/dev/null" \
        > "${LOCAL_ROOT}/remote-sha256.txt" 2> "${LOCAL_ROOT}/remote-sha256.stderr" || true

    {
        printf 'device=%s\n' "${DEVICE}"
        printf 'remote_bin_dir=%s\n' "${REMOTE_BIN_DIR}"
        printf 'model_path=%s\n' "${MODEL_PATH}"
        printf 'qnn_dir=%s\n' "${QNN_DIR}"
        printf 'local_root=%s\n' "${LOCAL_ROOT}"
        printf 'cooldown_sec=%s\n' "${COOLDOWN_SEC}"
        printf 'case_timeout_sec=%s\n' "${CASE_TIMEOUT_SEC}"
        printf 'reps=%s\n' "${REPS}"
        printf 'keep_raw=%s\n' "${KEEP_RAW}"
        printf 'single_workloads=%s\n' "${single_workloads[*]}"
        printf 'switch_workloads=%s\n' "${switch_workloads[*]}"
        printf 'formal_backends=cpu opencl qnn-npu\n'
        printf 'fastrpc_htp0_policy=record list-devices only, not formal success criteria\n'
    } > "${MANIFEST}"
}

main() {
    preflight

    for backend in "${backends[@]}"; do
        for workload in "${single_workloads[@]}"; do
            IFS=: read -r pp tg <<< "${workload}"
            run_single_backend "${backend}" "${pp}" "${tg}"
        done
    done

    for prefill in "${backends[@]}"; do
        for decode in "${backends[@]}"; do
            if [ "${prefill}" = "${decode}" ]; then
                continue
            fi
            for workload in "${switch_workloads[@]}"; do
                IFS=: read -r pp tg <<< "${workload}"
                run_switch "${prefill}" "${decode}" "${pp}" "${tg}"
            done
        done
    done

    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"
    printf 'LOCAL_ROOT=%s\n' "${LOCAL_ROOT}"
}

main "$@"
