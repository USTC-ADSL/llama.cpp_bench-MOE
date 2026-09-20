#!/system/bin/sh

# Keep the native probe independent from the lifetime of the host adb shell.
# The host runner starts this through nohup+setsid and polls the files after a
# temporarily offline adb transport reconnects.

if [ "$#" -lt 7 ]; then
    echo "usage: device_probe_launcher.sh WORK_DIR PID_FILE STATUS_FILE STDOUT_FILE STDERR_FILE START_FILE COMMAND [ARGS...]" >&2
    exit 2
fi

work_dir=$1
pid_file=$2
status_file=$3
stdout_file=$4
stderr_file=$5
start_file=$6
shift 6

write_atomic() {
    destination=$1
    shift
    temporary="${destination}.tmp.$$"
    printf '%s\n' "$@" >"$temporary" && mv -f "$temporary" "$destination"
}

cd "$work_dir" || exit 125
export LD_LIBRARY_PATH="$work_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$work_dir"

rm -f "$status_file"
write_atomic "$pid_file" "$$"
write_atomic "$start_file" \
    "pid=$$" \
    "started_epoch=$(date +%s)" \
    "command=$*"

"$@" >"$stdout_file" 2>"$stderr_file"
rc=$?

write_atomic "$status_file" \
    "exit_code=$rc" \
    "pid=$$" \
    "finished_epoch=$(date +%s)"
exit "$rc"
