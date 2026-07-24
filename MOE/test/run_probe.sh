#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TEST_DIR="$ROOT/MOE/test"
BUILD_DIR=${BUILD_DIR:-"$TEST_DIR/build-android"}
ADB_PORT=5038
DEVICE=fd8657d6
REMOTE=/data/local/tmp/buffer-capacity-probe
RESULT_ROOT="$TEST_DIR/results"
MODES=all
DO_BUILD=0
SMOKE=0
CONFIRMED=0
SWAP_ASSISTED=0
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: run_probe.sh [options] [-- native-probe-options]

  --device SERIAL                 default: fd8657d6
  --adb-port PORT                 default: 5038
  --remote-dir PATH               default: /data/local/tmp/buffer-capacity-probe
  --result-root PATH              default: MOE/test/results
  --modes LIST                    cpu,opencl,rpcmem,htp,combined or all
  --build                         build Android executable and v79 skel first
  --smoke                         safe 64 MiB/backend, 2 second validation
  --allow-swap                    swap-assisted policy: keep 1024 MiB MemAvailable
                                  and 256 MiB SwapFree while approaching ZRAM limit
  --confirm-capacity-test         required for automatic near-limit search

The full search deliberately creates memory pressure. It stops at the configured
MemAvailable/Swap guards, but LMKD or a reboot remains possible on a busy device.
EOF
}

while (($#)); do
    case "$1" in
        --device) DEVICE=$2; shift 2 ;;
        --adb-port) ADB_PORT=$2; shift 2 ;;
        --remote-dir) REMOTE=$2; shift 2 ;;
        --result-root) RESULT_ROOT=$2; shift 2 ;;
        --modes) MODES=$2; shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --allow-swap) SWAP_ASSISTED=1; shift ;;
        --confirm-capacity-test) CONFIRMED=1; shift ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if (( SWAP_ASSISTED )); then
    MEMORY_POLICY=swap-assisted
    SWAP_POLICY=allow
    RESERVE_MIB=1024
    SWAP_FLOOR_MIB=256
else
    MEMORY_POLICY=physical-resident
    SWAP_POLICY=reject
    RESERVE_MIB=512
    SWAP_FLOOR_MIB=512
fi

if (( ! SMOKE && ! CONFIRMED )); then
    echo "refusing a near-limit run without --confirm-capacity-test; use --smoke first" >&2
    exit 2
fi

if (( DO_BUILD )); then
    "$TEST_DIR/build_probe.sh"
fi

PROBE="$BUILD_DIR/buffer-capacity-probe"
SKEL="$BUILD_DIR/libbuffer-capacity-htp-v79.so"
LIBCXX="$BUILD_DIR/ship/libc++_shared.so"
LAUNCHER="$TEST_DIR/device_probe_launcher.sh"
for required in "$PROBE" "$SKEL"; do
    if [[ ! -f "$required" ]]; then
        echo "missing $required; run with --build" >&2
        exit 3
    fi
done
if [[ ! -f "$LAUNCHER" ]]; then
    echo "missing $LAUNCHER" >&2
    exit 3
fi

ADB=(adb -P "$ADB_PORT" -s "$DEVICE")
wait_for_device() {
    local attempt
    for attempt in $(seq 1 30); do
        if "${ADB[@]}" get-state >/dev/null 2>&1; then return 0; fi
        if (( attempt == 1 )); then adb -P "$ADB_PORT" reconnect offline >/dev/null 2>&1 || true; fi
        sleep 2
    done
    return 1
}

adb_retry() {
    local attempt
    for attempt in 1 2 3; do
        if "$@"; then return 0; fi
        echo "ADB command failed (attempt $attempt/3), retrying: $*" >&2
        sleep 2
    done
    return 1
}

push_retry() {
    local local_file=$1
    local remote_file=$2
    local remote_tmp="${remote_file}.tmp.$$"
    local local_size remote_size attempt
    local_size=$(stat -c %s "$local_file")
    for attempt in 1 2; do
        if "${ADB[@]}" push -Z "$local_file" "$remote_tmp"; then
            adb_retry "${ADB[@]}" shell "mv -f '$remote_tmp' '$remote_file'"
            return
        fi
        echo "ADB push response failed (attempt $attempt/2); checking whether the complete file arrived" >&2
        wait_for_device || continue
        remote_size=$("${ADB[@]}" shell "stat -c %s '$remote_tmp' 2>/dev/null" | tr -d '\r' || true)
        if [[ "$remote_size" == "$local_size" ]]; then
            echo "remote size matches after push EOF; accepting $remote_tmp" >&2
            adb_retry "${ADB[@]}" shell "mv -f '$remote_tmp' '$remote_file'"
            return
        fi
    done

    echo "falling back to 512 KiB chunked ADB push for $local_file" >&2
    wait_for_device || return 1
    local chunk_dir
    chunk_dir=$(mktemp -d)
    split -b 512K -d -a 3 "$local_file" "$chunk_dir/part."
    adb_retry "${ADB[@]}" shell "rm -f '${remote_tmp}.part.'* '$remote_tmp'"
    local chunk part_name remote_part chunk_size chunk_remote_size pushed
    for chunk in "$chunk_dir"/part.*; do
        part_name=${chunk##*/part.}
        remote_part="${remote_tmp}.part.${part_name}"
        chunk_size=$(stat -c %s "$chunk")
        pushed=0
        for attempt in 1 2 3; do
            if "${ADB[@]}" push -Z "$chunk" "$remote_part"; then pushed=1; break; fi
            wait_for_device || continue
            chunk_remote_size=$("${ADB[@]}" shell "stat -c %s '$remote_part' 2>/dev/null" | tr -d '\r' || true)
            if [[ "$chunk_remote_size" == "$chunk_size" ]]; then pushed=1; break; fi
        done
        if (( ! pushed )); then
            rm -rf "$chunk_dir"
            return 1
        fi
    done
    rm -rf "$chunk_dir"
    adb_retry "${ADB[@]}" shell "cat '${remote_tmp}.part.'* > '$remote_tmp' && rm -f '${remote_tmp}.part.'*"
    remote_size=$("${ADB[@]}" shell "stat -c %s '$remote_tmp' 2>/dev/null" | tr -d '\r' || true)
    if [[ "$remote_size" != "$local_size" ]]; then
        echo "chunked ADB push size mismatch for $remote_tmp: local=$local_size remote=$remote_size" >&2
        return 1
    fi
    adb_retry "${ADB[@]}" shell "mv -f '$remote_tmp' '$remote_file'"
}

if ! "${ADB[@]}" get-state >/dev/null 2>&1; then
    if ! wait_for_device; then
        echo "device $DEVICE is not reachable through adb server port $ADB_PORT" >&2
        adb -P "$ADB_PORT" devices -l >&2 || true
        exit 4
    fi
fi

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RESULT_DIR="$RESULT_ROOT/${STAMP}-${DEVICE}-${MEMORY_POLICY}"
mkdir -p "$RESULT_DIR/jsonl" "$RESULT_DIR/diagnostics" "$RESULT_DIR/remote-logs"
RUNTIME_LOG="$RESULT_DIR/probe-runtime.log"
LOGCAT_LOG="$RESULT_DIR/logcat-all.log"
RUNNER_EVENTS="$RESULT_DIR/runner-events.jsonl"
LOGCAT_PID=

append_runner_event() {
    local event=$1
    local mode=${2:-}
    local exit_code=${3:-0}
    local detail=${4:-}
    python3 - "$RUNNER_EVENTS" "$event" "$mode" "$exit_code" "$detail" "$MEMORY_POLICY" "$SWAP_POLICY" <<'PY'
import datetime, json, os, sys
path, event, mode, exit_code, detail, memory_policy, swap_policy = sys.argv[1:]
record = {
    "record_type": "runner_event",
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "event": event,
    "mode": mode,
    "exit_code": int(exit_code),
    "detail": detail,
    "memory_policy": memory_policy,
    "swap_policy": swap_policy,
}
with open(path, "a", encoding="utf-8") as out:
    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    out.flush()
    os.fsync(out.fileno())
PY
}

snapshot() {
    local name=$1
    {
        echo "timestamp=$(date -u +%FT%TZ)"
        "${ADB[@]}" shell 'getprop ro.product.manufacturer; getprop ro.product.model; getprop ro.product.device; getprop ro.build.fingerprint; getprop ro.soc.model' 2>&1 || true
        "${ADB[@]}" shell 'cat /proc/sys/kernel/random/boot_id; cat /proc/meminfo; cat /proc/swaps; cat /proc/pressure/memory' 2>&1 || true
        "${ADB[@]}" shell 'cat /sys/kernel/debug/dma_buf/bufinfo' 2>&1 || true
        "${ADB[@]}" shell 'cat /sys/class/kgsl/kgsl-3d0/gpu_available_frequencies 2>/dev/null' 2>&1 || true
    } >"$RESULT_DIR/diagnostics/$name.txt"
}

collect_post_failure() {
    local old_boot=$1
    local mode=$2
    local connected=0
    for _ in $(seq 1 30); do
        if "${ADB[@]}" get-state >/dev/null 2>&1; then connected=1; break; fi
        sleep 2
    done
    if (( ! connected )); then
        append_runner_event disconnect "$mode" 255 "device did not reconnect within 60 seconds"
        return
    fi
    local new_boot
    new_boot=$("${ADB[@]}" shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r')
    if [[ -n "$old_boot" && "$new_boot" != "$old_boot" ]]; then
        append_runner_event reboot "$mode" 255 "boot_id changed from $old_boot to $new_boot"
    else
        append_runner_event reconnect "$mode" 0 "boot_id=$new_boot"
    fi
    "${ADB[@]}" shell 'for f in /sys/fs/pstore/* /proc/last_kmsg; do [ -r "$f" ] && { echo "===== $f"; cat "$f"; }; done' \
        >"$RESULT_DIR/diagnostics/${mode}-pstore-last-kmsg.txt" 2>&1 || true
    "${ADB[@]}" shell dmesg >"$RESULT_DIR/diagnostics/${mode}-dmesg.txt" 2>&1 || true
    "${ADB[@]}" logcat -b all -d >"$RESULT_DIR/diagnostics/${mode}-logcat-after.txt" 2>&1 || true
}

pull_remote_file() {
    local remote_file=$1
    local local_file=$2
    mkdir -p "$(dirname "$local_file")"
    "${ADB[@]}" pull "$remote_file" "$local_file" >/dev/null 2>&1 || true
}

ensure_logcat() {
    if [[ -n "$LOGCAT_PID" ]] && kill -0 "$LOGCAT_PID" >/dev/null 2>&1; then
        return
    fi
    if [[ -n "$LOGCAT_PID" ]]; then
        wait "$LOGCAT_PID" >/dev/null 2>&1 || true
        append_runner_event logcat-restart "" 0 "continuous logcat transport ended; restarting after reconnect"
    fi
    "${ADB[@]}" logcat -b all -v threadtime -T 1 >>"$LOGCAT_LOG" 2>&1 &
    LOGCAT_PID=$!
}

run_detached_probe() {
    local mode=$1
    local remote_json=$2
    shift 2
    local -a command=("$REMOTE/buffer-capacity-probe" "$@")
    local prefix="$REMOTE/run-${mode}-${MEMORY_POLICY}"
    local remote_pid="${prefix}.pid"
    local remote_status="${prefix}.status"
    local remote_start="${prefix}.start"
    local remote_stdout="${prefix}.stdout.log"
    local remote_stderr="${prefix}.stderr.log"
    local remote_guard="${prefix}.launching"
    local local_stdout="$RESULT_DIR/remote-logs/${mode}.stdout.log"
    local local_stderr="$RESULT_DIR/remote-logs/${mode}.stderr.log"
    local local_status="$RESULT_DIR/remote-logs/${mode}.status"
    local local_start="$RESULT_DIR/remote-logs/${mode}.start"
    local launch_args

    adb_retry "${ADB[@]}" shell \
        "rm -rf '$remote_guard'; rm -f '$remote_pid' '$remote_status' '$remote_start' '$remote_stdout' '$remote_stderr'"
    printf -v launch_args '%q ' \
        "$REMOTE/device_probe_launcher.sh" "$REMOTE" "$remote_pid" "$remote_status" \
        "$remote_stdout" "$remote_stderr" "$remote_start" "${command[@]}"

    # The mkdir guard makes a retry idempotent if adb loses the successful
    # launch response. nohup ignores SIGHUP; setsid removes the process from the
    # adb shell process group; all descriptors are redirected before return.
    adb_retry "${ADB[@]}" shell \
        "cd '$REMOTE' && if mkdir '$remote_guard' 2>/dev/null; then nohup setsid $launch_args </dev/null >/dev/null 2>&1 & fi"

    local state state_raw state_rc new_boot rc pid
    local disconnected=0
    local poll_count=0
    local start_waits=0
    while :; do
        set +e
        state_raw=$("${ADB[@]}" shell \
            "if [ -f '$remote_status' ]; then echo STATUS; cat '$remote_status'; elif [ -s '$remote_pid' ]; then p=\$(cat '$remote_pid'); if kill -0 \"\$p\" 2>/dev/null; then echo RUNNING \"\$p\"; else echo MISSING \"\$p\"; fi; else echo STARTING; fi" \
            2>/dev/null)
        state_rc=$?
        set -e
        state=${state_raw//$'\r'/}

        if (( state_rc != 0 )); then
            if (( ! disconnected )); then
                append_runner_event transport-disconnect "$mode" 255 \
                    "adb transport became unavailable; detached probe may still be running"
                disconnected=1
            fi
            if ! wait_for_device; then
                append_runner_event disconnect "$mode" 255 "device did not reconnect within 60 seconds"
                return 255
            fi
            new_boot=$("${ADB[@]}" shell cat /proc/sys/kernel/random/boot_id 2>/dev/null | tr -d '\r' || true)
            if [[ -n "$BOOT_ID" && -n "$new_boot" && "$new_boot" != "$BOOT_ID" ]]; then
                append_runner_event reboot "$mode" 255 "boot_id changed from $BOOT_ID to $new_boot"
                return 255
            fi
            append_runner_event transport-reconnect "$mode" 0 \
                "boot_id=$new_boot; resuming detached probe polling"
            ensure_logcat
            disconnected=0
            continue
        fi

        case "$state" in
            STATUS$'\n'*)
                rc=$(sed -n 's/^exit_code=//p' <<<"$state")
                if [[ ! "$rc" =~ ^[0-9]+$ ]]; then
                    append_runner_event remote-state-error "$mode" 255 \
                        "atomic status file did not contain a numeric exit_code"
                    rc=255
                fi
                pull_remote_file "$remote_stdout" "$local_stdout"
                pull_remote_file "$remote_stderr" "$local_stderr"
                pull_remote_file "$remote_status" "$local_status"
                pull_remote_file "$remote_start" "$local_start"
                pull_remote_file "$remote_json" "$RESULT_DIR/jsonl/${mode}-${MEMORY_POLICY}.jsonl"
                {
                    echo "===== mode=$mode device-stdout ====="
                    [[ -f "$local_stdout" ]] && sed 's/^/[stdout] /' "$local_stdout"
                    echo "===== mode=$mode device-stderr ====="
                    [[ -f "$local_stderr" ]] && sed 's/^/[stderr] /' "$local_stderr"
                } >>"$RUNTIME_LOG"
                return "$rc"
                ;;
            RUNNING\ *)
                start_waits=0
                ;;
            MISSING\ *)
                pid=${state#MISSING }
                sleep 1
                pull_remote_file "$remote_stdout" "$local_stdout"
                pull_remote_file "$remote_stderr" "$local_stderr"
                pull_remote_file "$remote_start" "$local_start"
                pull_remote_file "$remote_json" "$RESULT_DIR/jsonl/${mode}-${MEMORY_POLICY}.jsonl"
                append_runner_event remote-process-missing "$mode" 255 \
                    "launcher pid=$pid disappeared without an atomic status file; possible signal/LMKD kill"
                return 255
                ;;
            STARTING)
                ((start_waits += 1))
                if (( start_waits >= 4 )); then
                    append_runner_event remote-launch-timeout "$mode" 255 \
                        "detached launcher did not create its pid file within 15 seconds"
                    return 255
                fi
                ;;
            *)
                append_runner_event remote-state-error "$mode" 255 "unexpected remote state: $state"
                return 255
                ;;
        esac

        ((poll_count += 1))
        if (( poll_count % 6 == 0 )); then
            # Native records are fsynced. Periodic pulls retain partial results
            # even if a later reconnect fails.
            pull_remote_file "$remote_json" "$RESULT_DIR/jsonl/${mode}-${MEMORY_POLICY}.jsonl"
        fi
        sleep 5
    done
}

cleanup() {
    if [[ -n "$LOGCAT_PID" ]]; then
        kill "$LOGCAT_PID" >/dev/null 2>&1 || true
        wait "$LOGCAT_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

adb_retry "${ADB[@]}" shell "mkdir -p '$REMOTE' && chmod 755 '$REMOTE'"
push_retry "$PROBE" "$REMOTE/buffer-capacity-probe"
push_retry "$SKEL" "$REMOTE/libbuffer-capacity-htp-v79.so"
push_retry "$LAUNCHER" "$REMOTE/device_probe_launcher.sh"
if [[ -f "$LIBCXX" ]]; then
    push_retry "$LIBCXX" "$REMOTE/libc++_shared.so"
fi
adb_retry "${ADB[@]}" shell "chmod 755 '$REMOTE/buffer-capacity-probe' '$REMOTE/libbuffer-capacity-htp-v79.so' '$REMOTE/device_probe_launcher.sh'"

snapshot before
BOOT_ID=$("${ADB[@]}" shell cat /proc/sys/kernel/random/boot_id | tr -d '\r')
"${ADB[@]}" logcat -b all -v threadtime -T 1 >"$LOGCAT_LOG" 2>&1 &
LOGCAT_PID=$!
append_runner_event start "" 0 "boot_id=$BOOT_ID device=$DEVICE adb_port=$ADB_PORT reserve_mib=$RESERVE_MIB swap_floor_mib=$SWAP_FLOOR_MIB"

if [[ "$MODES" == all ]]; then
    MODE_LIST=(cpu opencl rpcmem htp combined)
else
    IFS=',' read -r -a MODE_LIST <<<"$MODES"
fi

for mode in "${MODE_LIST[@]}"; do
    case "$mode" in cpu|opencl|rpcmem|htp|combined) ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac
    remote_json="$REMOTE/${mode}-${MEMORY_POLICY}.jsonl"
    "${ADB[@]}" shell "rm -f '$remote_json'"
    probe_args=(
        --mode "$mode"
        --reserve-mib "$RESERVE_MIB"
        --swap-tolerance-mib 64
        --swap-policy "$SWAP_POLICY"
        --swap-floor-mib "$SWAP_FLOOR_MIB"
        --recovery-timeout-sec 30
        --recovery-tolerance-mib 256
        --coarse-step-mib 256
        --fine-step-mib 64
        --final-step-mib 16
        --output-jsonl "$remote_json"
    )
    if (( SMOKE )); then
        smoke_target=64
        smoke_extra=()
        if [[ "$mode" == combined ]]; then
            smoke_target=192
            smoke_extra=(--ratios 1:1:1)
        fi
        probe_args+=(
            --target-mib "$smoke_target"
            --search-hold-sec 1
            --final-hold-sec 2
            --repeat 1
            --allocation-order 'cpu>gpu>htp'
            "${smoke_extra[@]}"
        )
    else
        probe_args+=(
            --target-mib auto
            --search-hold-sec 30
            --final-hold-sec 60
        )
        if [[ "$mode" == combined ]]; then
            probe_args+=(--repeat 2)
        else
            # Allocation order only matters when multiple backends coexist.
            # A single-backend capacity point uses one order and two repeats unless
            # the caller explicitly overrides these native options after --.
            probe_args+=(--repeat 2 --allocation-order 'cpu>gpu>htp')
        fi
    fi
    probe_args+=("${EXTRA_ARGS[@]}")
    append_runner_event mode-start "$mode" 0 "${probe_args[*]}"
    set +e
    run_detached_probe "$mode" "$remote_json" "${probe_args[@]}"
    rc=$?
    set -e
    append_runner_event mode-exit "$mode" "$rc" "detached native probe exited"
    adb_retry "${ADB[@]}" pull "$remote_json" "$RESULT_DIR/jsonl/${mode}-${MEMORY_POLICY}.jsonl" >/dev/null 2>&1 || true
    if (( rc != 0 )); then
        collect_post_failure "$BOOT_ID" "$mode"
        break
    fi
    if ! "${ADB[@]}" get-state >/dev/null 2>&1; then
        collect_post_failure "$BOOT_ID" "$mode"
        break
    fi
done

snapshot after
cleanup
LOGCAT_PID=

mapfile -t JSONL_FILES < <(find "$RESULT_DIR/jsonl" -type f -name '*.jsonl' -size +0c | sort)
if ((${#JSONL_FILES[@]})); then
    python3 "$TEST_DIR/analyze_results.py" \
        --input "${JSONL_FILES[@]}" "$RUNNER_EVENTS" \
        --output-dir "$RESULT_DIR" \
        --runtime-log "$RUNTIME_LOG" \
        --logcat-log "$LOGCAT_LOG"
else
    echo "no native JSONL was recovered; inspect $RUNNER_EVENTS and $RUNTIME_LOG" >&2
fi

echo "results: $RESULT_DIR"
