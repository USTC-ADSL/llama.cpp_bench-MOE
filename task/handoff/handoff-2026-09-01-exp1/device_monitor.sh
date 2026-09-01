#!/system/bin/sh

pid="$1"
out="$2"

if [ -z "$pid" ] || [ -z "$out" ]; then
    echo "usage: $0 <pid> <output.csv>" >&2
    exit 2
fi

echo "timestamp_ms,pss_kb,memavailable_kb,majflt,rss_kb" > "$out"

while [ -d "/proc/$pid" ]; do
    timestamp_ms=$(date +%s%3N 2>/dev/null)
    pss_kb=$(awk '/^Pss:/ { print $2; exit }' "/proc/$pid/smaps_rollup" 2>/dev/null)
    memavailable_kb=$(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo 2>/dev/null)
    rss_kb=$(awk '/^VmRSS:/ { print $2; exit }' "/proc/$pid/status" 2>/dev/null)

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
    echo "$timestamp_ms,$pss_kb,$memavailable_kb,$majflt,$rss_kb" >> "$out"
    sleep 0.2
done
