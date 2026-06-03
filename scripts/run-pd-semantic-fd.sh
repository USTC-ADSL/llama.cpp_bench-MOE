#!/usr/bin/env bash
set -Eeuo pipefail

DEVICE=${DEVICE:-fd8657d6}
REMOTE_BIN_DIR=${REMOTE_BIN_DIR:-/data/local/tmp/bench-PD}
MODEL_PATH=${MODEL_PATH:-/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf}
QNN_DIR=${QNN_DIR:-/data/local/tmp/models/Qwen2.5-3B-AoT/qnn}
LOCAL_ROOT=${LOCAL_ROOT:-results/pd-semantic-fd-$(date -u +%Y%m%d-%H%M)}
CASE_TIMEOUT_SEC=${CASE_TIMEOUT_SEC:-1200}
COOLDOWN_SEC=${COOLDOWN_SEC:-10}
N_GEN=${N_GEN:-16}
SEED=${SEED:-123}
CTX_SIZE=${CTX_SIZE:-2048}
BATCH_SIZE=${BATCH_SIZE:-128}
UBATCH_SIZE=${UBATCH_SIZE:-128}
THREADS=${THREADS:-4}
QNN_TRACE=${QNN_TRACE:-0}
COMPLETION_VERBOSE=${COMPLETION_VERBOSE:-1}
PROMPT=${PROMPT:-Write two concise sentences explaining why the sky looks blue during the day.}

RAW_DIR="${LOCAL_ROOT}/raw"
SUMMARY_DIR="${LOCAL_ROOT}/summary"
COMMANDS="${LOCAL_ROOT}/commands.sh"
CASES="${LOCAL_ROOT}/cases.tsv"
MANIFEST="${LOCAL_ROOT}/manifest.txt"

mkdir -p "${RAW_DIR}" "${SUMMARY_DIR}"
printf '#!/usr/bin/env bash\n' > "${COMMANDS}"
printf 'case\tcategory\tbackend\tprefill\tdecode\texpected_decode\trc\tsemantic_ok\troute_ok\tprompt_ms\teval_ms\tresponse_excerpt\tstdout\tstderr\texit\tcommand\tenv\tresponse\n' > "${CASES}"

FAILURES=0

shell_quote() {
    printf "'"
    printf '%s' "$1" | sed "s/'/'\\\\''/g"
    printf "'"
}

remote_base_env() {
    printf 'cd %s && export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH && export ADSP_LIBRARY_PATH=%s' \
        "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}" "${REMOTE_BIN_DIR}"
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
    if [ "${QNN_TRACE}" = "1" ]; then
        printf ' && export GGML_QNN_AOT_TRACE_ASSIGN=1'
        printf ' && export GGML_QNN_AOT_TRACE_MATCH=1'
    fi
}

dynamic_env() {
    local prefill=$1
    local decode=$2
    printf ' && export GGML_HETERO_DYNAMIC_MODE=phase'
    printf ' && export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=%s' "${prefill}"
    printf ' && export GGML_HETERO_DYNAMIC_DECODE_ROUTE=%s' "${decode}"
    printf ' && export GGML_HETERO_DYNAMIC_TRACE=1'
    printf ' && export GGML_HETERO_DYNAMIC_TRACE_TIMING=1'
    printf ' && export GGML_HETERO_TRACE_SHARE=1'
    printf ' && export GGML_HETERO_DYNAMIC_PRERESERVE=1'
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
        printf -- '-ngl 99 -dev qnn-npu/GPUOpenCL'
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

write_summary() {
    python3 - "${LOCAL_ROOT}" <<'PY'
import csv
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cases = root / "cases.tsv"
summary = root / "summary"
summary.mkdir(parents=True, exist_ok=True)

rows = []
if cases.exists():
    with cases.open(newline="", errors="replace") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

fields = [
    "case", "category", "backend", "prefill", "decode", "expected_decode",
    "rc", "semantic_ok", "route_ok", "prompt_ms", "eval_ms", "response_excerpt",
    "stdout", "stderr", "exit", "command", "env", "response",
]

def write_csv(path, items):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in items)

def md_cell(value):
    return str(value or "").replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")

def write_md(path, items):
    compact = [
        "case", "category", "backend", "prefill", "decode", "rc",
        "semantic_ok", "route_ok", "prompt_ms", "eval_ms", "response_excerpt",
    ]
    with path.open("w") as f:
        f.write("| " + " | ".join(compact) + " |\n")
        f.write("| " + " | ".join("---" for _ in compact) + " |\n")
        for row in items:
            f.write("| " + " | ".join(md_cell(row.get(k, "")) for k in compact) + " |\n")

failures = [
    row for row in rows
    if row.get("rc") != "0" or row.get("semantic_ok") != "1" or row.get("route_ok") != "1"
]

write_csv(summary / "semantic_results.csv", rows)
write_md(summary / "semantic_results.md", rows)
write_csv(summary / "failures.csv", failures)
write_md(summary / "failures.md", failures)

with (summary / "README.md").open("w") as f:
    f.write("# PD semantic fd summary\n\n")
    f.write(f"Local root: `{root}`\n\n")
    f.write("This runner uses `llama-completion` for deterministic semantic generation, not `llama-bench` smoke output.\n\n")
    f.write("Formal matrix: single-backend `cpu`, `opencl`/`GPUOpenCL`, `qnn`/`qnn-npu`, plus every ordered non-self phase switch among those three backends.\n\n")
    f.write("Every case keeps raw stdout, stderr, exit status, command, environment, and response text under `raw/`.\n")

print(root)
PY
}

finish_summary() {
    set +e
    write_summary > "${SUMMARY_DIR}/last_summary_path.txt" 2> "${SUMMARY_DIR}/summary.err"
    return 0
}
trap finish_summary EXIT

save_command_and_env() {
    local remote_cmd=$1
    local command_file=$2
    local env_file=$3
    printf '%s\n' "timeout ${CASE_TIMEOUT_SEC} adb -s ${DEVICE} shell '${remote_cmd}'" > "${command_file}"
    printf '%s\n' "timeout ${CASE_TIMEOUT_SEC} adb -s ${DEVICE} shell '${remote_cmd}'" >> "${COMMANDS}"
    printf '%s\n' "${remote_cmd}" | sed 's/ && /\n/g' | sed -n 's/^export //p' > "${env_file}"
}

response_excerpt() {
    tr '\r\n\t' '   ' | sed 's/  */ /g' | cut -c 1-220
}

run_remote_case() {
    local name=$1
    local category=$2
    local backend=$3
    local prefill=$4
    local decode=$5
    local expected_decode=$6
    local remote_cmd=$7

    local stdout_file="${RAW_DIR}/${name}.stdout"
    local stderr_file="${RAW_DIR}/${name}.stderr"
    local exit_file="${RAW_DIR}/${name}.exit"
    local command_file="${RAW_DIR}/${name}.command"
    local env_file="${RAW_DIR}/${name}.env"
    local response_file="${RAW_DIR}/${name}.response.txt"

    printf '[%s] start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" >&2
    save_command_and_env "${remote_cmd}" "${command_file}" "${env_file}"

    set +e
    timeout "${CASE_TIMEOUT_SEC}" adb -s "${DEVICE}" shell "${remote_cmd}" > "${stdout_file}" 2> "${stderr_file}"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${exit_file}"

    local stdout_text
    local response
    local excerpt
    local semantic_ok=0
    local route_ok=1
    local prompt_ms
    local eval_ms

    stdout_text="$(tr -d '\r' < "${stdout_file}")"
    response="${stdout_text#${PROMPT} }"
    printf '%s\n' "${response}" > "${response_file}"
    excerpt="$(printf '%s\n' "${response}" | response_excerpt)"

    if [ "${rc}" = "0" ] &&
       printf '%s\n' "${response}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'because|scatter|scattering|atmosphere|wavelength|rayleigh|molecules'; then
        semantic_ok=1
    fi

    if [ "${category}" = "switch" ]; then
        route_ok=0
        if grep -Fq "apply_hetero_plan: updated hetero plan via decode: backend=${expected_decode}" "${stderr_file}" ||
           grep -Fq "apply_hetero_plan: applied hetero route from decode: attn=${expected_decode},ffn=${expected_decode},output=${expected_decode}" "${stderr_file}" ||
           grep -Eq "maybe_apply_dynamic_route: timing phase=decode .*target=.*(attn=${expected_decode}|ffn=${expected_decode}|output=${expected_decode})" "${stderr_file}"; then
            route_ok=1
        fi
        if grep -Eq 'backend-unavailable|failed to allocate graph|rejecting hetero plan update|KV migration failed' "${stderr_file}"; then
            route_ok=0
        fi
    fi

    prompt_ms="$(sed -n 's/^common_perf_print:[[:space:]]*prompt eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_file}" | head -n 1)"
    eval_ms="$(sed -n 's/^common_perf_print:[[:space:]]*eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_file}" | head -n 1)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${name}" "${category}" "${backend}" "${prefill}" "${decode}" "${expected_decode}" \
        "${rc}" "${semantic_ok}" "${route_ok}" "${prompt_ms}" "${eval_ms}" "${excerpt}" \
        "${stdout_file}" "${stderr_file}" "${exit_file}" "${command_file}" "${env_file}" "${response_file}" >> "${CASES}"

    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"

    if [ "${semantic_ok}" != "1" ] || [ "${route_ok}" != "1" ] || [ "${rc}" != "0" ]; then
        FAILURES=$((FAILURES + 1))
        printf '[%s] fail %s rc=%s semantic_ok=%s route_ok=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${rc}" "${semantic_ok}" "${route_ok}" >&2
    else
        printf '[%s] pass %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" >&2
    fi

    if [ "${COOLDOWN_SEC}" -gt 0 ]; then
        sleep "${COOLDOWN_SEC}"
    fi
}

completion_cmd() {
    local args=$1
    local mask=$2
    local quoted_prompt
    local verbose_flag=""
    quoted_prompt="$(shell_quote "${PROMPT}")"
    if [ "${COMPLETION_VERBOSE}" = "1" ]; then
        verbose_flag="-v"
    fi

    printf ' && taskset %s ./llama-completion %s --simple-io -no-cnv -st --temp 0 -m %s %s -t %s -c %s -b %s -ub %s -p %s -n %s -s %s --no-warmup --no-mmap' \
        "${mask}" "${verbose_flag}" "${MODEL_PATH}" "${args}" "${THREADS}" "${CTX_SIZE}" "${BATCH_SIZE}" "${UBATCH_SIZE}" \
        "${quoted_prompt}" "${N_GEN}" "${SEED}"
}

run_single_backend() {
    local backend=$1
    local args
    local mask
    local cmd
    args="$(backend_args "${backend}")"
    mask="$(taskset_for_backend "${backend}")"
    cmd="$(remote_base_env)"
    if [ "${backend}" = qnn ]; then
        cmd="${cmd}$(qnn_env)"
    else
        cmd="${cmd}$(disable_qnn_env)"
    fi
    cmd="${cmd}$(completion_cmd "${args}" "${mask}")"
    run_remote_case "single_${backend}" "single_backend" "${backend}" "" "" "$(route_name "${backend}")" "${cmd}"
}

run_switch() {
    local prefill_backend=$1
    local decode_backend=$2
    local prefill_route
    local decode_route
    local args
    local mask
    local cmd

    prefill_route="$(route_name "${prefill_backend}")"
    decode_route="$(route_name "${decode_backend}")"
    args="$(route_args "${prefill_route}" "${decode_route}")"
    mask="$(taskset_for_route "${prefill_backend}" "${decode_backend}")"
    cmd="$(remote_base_env)"

    if [ "${prefill_backend}" = qnn ] || [ "${decode_backend}" = qnn ]; then
        cmd="${cmd}$(qnn_env)"
    else
        cmd="${cmd}$(disable_qnn_env)"
    fi
    if [ "${prefill_backend}" = qnn ] && [ "${decode_backend}" = opencl ]; then
        cmd="${cmd} && export GGML_HETERO_QNN_SHARED_HOST=1 && export GGML_OPENCL_EXPERIMENTAL_QNN_DIRECT_HOST_PTR=1"
    fi
    if { [ "${prefill_backend}" = cpu ] && [ "${decode_backend}" = opencl ]; } ||
       { [ "${prefill_backend}" = opencl ] && [ "${decode_backend}" = cpu ]; }; then
        cmd="${cmd} && export GGML_HETERO_ENABLE_OPENCL_CPU_EXTRA_CPU_COPY=1 && export GGML_HETERO_DISABLE_CPU_OPENCL_SHARED_HOST=1"
    fi

    cmd="${cmd}$(dynamic_env "${prefill_route}" "${decode_route}")"
    cmd="${cmd}$(completion_cmd "${args}" "${mask}")"
    run_remote_case "switch_${prefill_backend}_to_${decode_backend}" "switch" "" "${prefill_route}" "${decode_route}" "${decode_route}" "${cmd}"
}

preflight() {
    adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-bench"
    adb -s "${DEVICE}" shell "test -x ${REMOTE_BIN_DIR}/llama-completion"
    adb -s "${DEVICE}" shell "test -f ${MODEL_PATH}"
    adb -s "${DEVICE}" shell "test -f ${QNN_DIR}/config.json"

    local cmd
    cmd="$(remote_base_env) && export GGML_HEXAGON_EXPERIMENTAL=1 && ./llama-bench --list-devices"
    save_command_and_env "${cmd}" "${RAW_DIR}/list-devices.command" "${RAW_DIR}/list-devices.env"

    set +e
    adb -s "${DEVICE}" shell "${cmd}" > "${RAW_DIR}/list-devices.stdout" 2> "${RAW_DIR}/list-devices.stderr"
    local rc=$?
    set -e
    printf '%s\n' "${rc}" > "${RAW_DIR}/list-devices.exit"
    cp "${RAW_DIR}/list-devices.stdout" "${LOCAL_ROOT}/list-devices.stdout"
    cp "${RAW_DIR}/list-devices.stderr" "${LOCAL_ROOT}/list-devices.stderr"

    if [ "${rc}" != "0" ]; then
        printf 'error: --list-devices failed; see %s\n' "${RAW_DIR}/list-devices.stderr" >&2
        exit 1
    fi
    if ! grep -q 'GPUOpenCL' "${RAW_DIR}/list-devices.stdout"; then
        printf 'error: fd device list does not include GPUOpenCL; see %s\n' "${RAW_DIR}/list-devices.stdout" >&2
        exit 1
    fi
    if ! grep -q 'qnn-npu' "${RAW_DIR}/list-devices.stdout"; then
        printf 'error: fd device list does not include qnn-npu; see %s\n' "${RAW_DIR}/list-devices.stdout" >&2
        exit 1
    fi

    adb -s "${DEVICE}" shell "cd ${REMOTE_BIN_DIR} && sha256sum llama-bench llama-completion libllama.so libggml.so libggml-opencl.so libggml-qnn.so 2>/dev/null" \
        > "${LOCAL_ROOT}/remote-sha256.txt" 2> "${LOCAL_ROOT}/remote-sha256.stderr" || true

    {
        printf 'device=%s\n' "${DEVICE}"
        printf 'remote_bin_dir=%s\n' "${REMOTE_BIN_DIR}"
        printf 'model_path=%s\n' "${MODEL_PATH}"
        printf 'qnn_dir=%s\n' "${QNN_DIR}"
        printf 'local_root=%s\n' "${LOCAL_ROOT}"
        printf 'case_timeout_sec=%s\n' "${CASE_TIMEOUT_SEC}"
        printf 'cooldown_sec=%s\n' "${COOLDOWN_SEC}"
        printf 'n_gen=%s\n' "${N_GEN}"
        printf 'seed=%s\n' "${SEED}"
        printf 'ctx_size=%s\n' "${CTX_SIZE}"
        printf 'batch_size=%s\n' "${BATCH_SIZE}"
        printf 'ubatch_size=%s\n' "${UBATCH_SIZE}"
        printf 'threads=%s\n' "${THREADS}"
        printf 'qnn_trace=%s\n' "${QNN_TRACE}"
        printf 'completion_verbose=%s\n' "${COMPLETION_VERBOSE}"
        printf 'prompt=%s\n' "${PROMPT}"
        printf 'formal_backends=cpu opencl qnn-npu\n'
        printf 'fastrpc_policy=not tested by this runner\n'
    } > "${MANIFEST}"
}

main() {
    preflight

    local backends=(cpu opencl qnn)
    local backend
    for backend in "${backends[@]}"; do
        run_single_backend "${backend}"
    done

    local prefill
    local decode
    for prefill in "${backends[@]}"; do
        for decode in "${backends[@]}"; do
            if [ "${prefill}" = "${decode}" ]; then
                continue
            fi
            run_switch "${prefill}" "${decode}"
        done
    done

    write_summary > "${SUMMARY_DIR}/last_summary_path.txt"
    printf 'LOCAL_ROOT=%s\n' "${LOCAL_ROOT}"
    if [ "${FAILURES}" -ne 0 ]; then
        exit 1
    fi
}

main "$@"
