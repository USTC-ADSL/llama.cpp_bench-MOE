#!/usr/bin/env bash

set -euo pipefail

: "${DEVICE:?set DEVICE to an adb serial}"
: "${MODEL_PATH:?set MODEL_PATH to the device-side GGUF path}"
: "${QNN_DIR:?set QNN_DIR to the device-side QNN AoT directory}"
: "${REMOTE_BIN_DIR:?set REMOTE_BIN_DIR to the device-side binary directory}"

prompt="${PROMPT:-Write two concise sentences explaining why the sky looks blue during the day.}"

stdout_logs=()
stderr_logs=()

cleanup() {
    rm -f "${stdout_logs[@]}" "${stderr_logs[@]}"
}
trap cleanup EXIT

run_case() {
    local case_name="$1"
    local prefill_route="$2"
    local decode_route="$3"
    local decode_backend_label="$4"
    local primary_device="$5"

    local stdout_log
    local stderr_log
    stdout_log="$(mktemp)"
    stderr_log="$(mktemp)"
    stdout_logs+=("${stdout_log}")
    stderr_logs+=("${stderr_log}")

    adb -s "${DEVICE}" shell "
cd ${REMOTE_BIN_DIR} &&
export LD_LIBRARY_PATH=${REMOTE_BIN_DIR}:\$LD_LIBRARY_PATH &&
export ADSP_LIBRARY_PATH=${REMOTE_BIN_DIR} &&
export GGML_HEXAGON_EXPERIMENTAL=1 &&
export GGML_QNN_AOT_CONFIG=${QNN_DIR}/config.json &&
export GGML_QNN_AOT_MODEL_DIR=${QNN_DIR} &&
export GGML_QNN_AOT_WRITE_GENERIC_KV=1 &&
export GGML_QNN_AOT_DISABLE_SEED_KV=1 &&
export GGML_HETERO_DYNAMIC_MODE=phase &&
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=${prefill_route} &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=${decode_route} &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev ${primary_device} -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${prompt}' \
  -n 24 -s 123 --no-warmup" \
      >"${stdout_log}" 2>"${stderr_log}"

    local stdout_text
    local response
    local prompt_ms
    local eval_ms
    local max_kv_migration_us

    stdout_text="$(tr -d '\r' < "${stdout_log}")"
    response="${stdout_text#${prompt} }"
    prompt_ms="$(sed -n 's/^common_perf_print:[[:space:]]*prompt eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_log}" | head -n 1)"
    eval_ms="$(sed -n 's/^common_perf_print:[[:space:]]*eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_log}" | head -n 1)"
    max_kv_migration_us="$(sed -n 's/.*kv_migration_us=\([0-9][0-9]*\).*/\1/p' "${stderr_log}" | sort -nr | head -n 1)"

    if [[ -z "${max_kv_migration_us}" || "${max_kv_migration_us}" == "0" ]]; then
        echo "${case_name}: expected non-zero switch overhead accounting" >&2
        rg -n "apply_hetero_plan|maybe_apply_dynamic_route|replay_dynamic_qnn_prefix|kv_migration_us=|common_perf_print" "${stderr_log}" >&2 || true
        exit 1
    fi

    if ! rg -q "apply_hetero_plan: updated hetero plan via decode: backend=${decode_backend_label}" "${stderr_log}"; then
        echo "${case_name}: expected decode route switch into ${decode_backend_label}" >&2
        rg -n "apply_hetero_plan|maybe_apply_dynamic_route" "${stderr_log}" >&2 || true
        exit 1
    fi

    if ! printf '%s\n' "${response}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'because|scatter|atmosphere|wavelength|rayleigh|molecules'; then
        echo "${case_name}: expected semantic sky-blue explanation, got: ${response}" >&2
        exit 1
    fi

    printf '%s device=%s prompt_ms=%s eval_ms=%s max_kv_migration_us=%s\n' \
        "${case_name}" "${DEVICE}" "${prompt_ms}" "${eval_ms}" "${max_kv_migration_us}"
    printf '%s\n' "${response}"
}

run_case "opencl-main opencl->qnn" "opencl" "qnn-npu" "qnn-npu" "GPUOpenCL"
run_case "opencl-main qnn->opencl" "qnn-npu" "opencl" "opencl" "GPUOpenCL"
run_case "qnn-main qnn->opencl" "qnn-npu" "opencl" "opencl" "qnn-npu"
