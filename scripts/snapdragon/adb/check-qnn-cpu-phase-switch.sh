#!/usr/bin/env bash

set -euo pipefail

: "${DEVICE:?set DEVICE to an adb serial}"
: "${MODEL_PATH:?set MODEL_PATH to the device-side GGUF path}"
: "${QNN_DIR:?set QNN_DIR to the device-side QNN AoT directory}"
: "${REMOTE_BIN_DIR:?set REMOTE_BIN_DIR to the device-side binary directory}"

prompt="${PROMPT:-Write two concise sentences explaining why the sky looks blue during the day.}"

stdout_log="$(mktemp)"
stderr_log="$(mktemp)"

cleanup() {
    rm -f "${stdout_log}" "${stderr_log}"
}
trap cleanup EXIT

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
export GGML_HETERO_DYNAMIC_PREFILL_ROUTE=qnn-npu &&
export GGML_HETERO_DYNAMIC_DECODE_ROUTE=cpu &&
export GGML_HETERO_DYNAMIC_TRACE_TIMING=1 &&
taskset 80 ./llama-completion --simple-io -no-cnv -st --temp 0 \
  -m ${MODEL_PATH} \
  -ngl 99 -dev qnn-npu -t 1 -c 2048 -b 2048 -ub 512 \
  -p '${prompt}' \
  -n 16 -s 123 --no-warmup" \
  >"${stdout_log}" 2>"${stderr_log}"

stdout_text="$(tr -d '\r' < "${stdout_log}")"
response="${stdout_text#${prompt} }"

if ! sed -n 's/.*kv_migration_us=\([0-9][0-9]*\).*/\1/p' "${stderr_log}" | awk '$1 > 0 { found = 1 } END { exit(found ? 0 : 1) }'; then
    echo "expected non-zero switch overhead accounting when switching qnn-npu -> cpu" >&2
    rg -n "starting KV migration|replaying .*prefix token|kv_migration_us=|phase-level KV placement|common_perf_print" "${stderr_log}" >&2 || true
    exit 1
fi

if ! printf '%s\n' "${response}" | tr '[:upper:]' '[:lower:]' | grep -Eq 'because|scattering|atmosphere|wavelength|rayleigh'; then
    echo "expected semantic sky-blue explanation, got: ${response}" >&2
    exit 1
fi

prompt_ms="$(sed -n 's/^common_perf_print:[[:space:]]*prompt eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_log}" | head -n 1)"
eval_ms="$(sed -n 's/^common_perf_print:[[:space:]]*eval time = *\([0-9.][0-9.]*\) ms.*/\1/p' "${stderr_log}" | head -n 1)"

printf 'device=%s prompt_ms=%s eval_ms=%s\n' "${DEVICE}" "${prompt_ms}" "${eval_ms}"
printf '%s\n' "${response}"
