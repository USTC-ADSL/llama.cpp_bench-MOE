#!/system/bin/sh

pid="$1"
out="$2"
cgroup="$3"

if [ -z "$pid" ] || [ -z "$out" ]; then
    echo "usage: $0 <pid> <output.csv> [memory-cgroup]" >&2
    exit 2
fi

echo "timestamp_ms,pss_kb,memavailable_kb,majflt,rss_kb,cgroup_usage_bytes,cgroup_max_usage_bytes,cgroup_failcnt" > "$out"

while [ -d "/proc/$pid" ]; do
    timestamp_ms=$(date +%s%3N 2>/dev/null)
    pss_kb=$(awk '/^Pss:/ { print $2; exit }' "/proc/$pid/smaps_rollup" 2>/dev/null)
    memavailable_kb=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null)
    rss_kb=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status" 2>/dev/null)
    cgroup_usage_bytes=-1
    cgroup_max_usage_bytes=-1
    cgroup_failcnt=-1
    if [ -n "$cgroup" ]; then
        cgroup_usage_bytes=$(cat "$cgroup/memory.usage_in_bytes" 2>/dev/null)
        cgroup_max_usage_bytes=$(cat "$cgroup/memory.max_usage_in_bytes" 2>/dev/null)
        cgroup_failcnt=$(cat "$cgroup/memory.failcnt" 2>/dev/null)
    fi

    majflt=-1
    if [ -r "/proc/$pid/stat" ]; then
        stat_line=$(cat "/proc/$pid/stat" 2>/dev/null)
        stat_rest=${stat_line##*) }
        set -- $stat_rest
        process_majflt=${10}
        case "$process_majflt" in
            ''|*[!0-9]*) ;;
            *) majflt=$process_majflt ;;
        esac
    fi

    [ -n "$timestamp_ms" ] || timestamp_ms=0
    [ -n "$pss_kb" ] || pss_kb=-1
    [ -n "$memavailable_kb" ] || memavailable_kb=-1
    [ -n "$rss_kb" ] || rss_kb=-1
    [ -n "$cgroup_usage_bytes" ] || cgroup_usage_bytes=-1
    [ -n "$cgroup_max_usage_bytes" ] || cgroup_max_usage_bytes=-1
    [ -n "$cgroup_failcnt" ] || cgroup_failcnt=-1
    echo "$timestamp_ms,$pss_kb,$memavailable_kb,$majflt,$rss_kb,$cgroup_usage_bytes,$cgroup_max_usage_bytes,$cgroup_failcnt" >> "$out"
    sleep 0.2
done
