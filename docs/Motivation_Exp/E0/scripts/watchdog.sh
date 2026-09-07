#!/system/bin/sh
# Same safety monitor in both sampling arms. No cgroup writes or policy changes.
pid="$1"
dir="$2"
low_since=0
phase_prev=""
phase_start=0
echo 'uptime_s,phase,memavailable_kb,thermal_status,pss_kb,cgroup_usage_bytes,cgroup_historical_peak_bytes' > "$dir/memory.csv"
cg_path=$(awk -F: '$2 == "memory" {print $3}' "/proc/$pid/cgroup")
cg_mount=$(awk '$3 == "cgroup" && $4 ~ /(^|,)memory(,|$)/ {print $2; exit}' /proc/mounts)
cg="$cg_mount$cg_path"
while kill -0 "$pid" 2>/dev/null; do
    read uptime rest < /proc/uptime
    milliseconds=$(awk -v t="$uptime" 'BEGIN {printf "%.0f",t*1000}')
    phase=$(cat "$dir/phase" 2>/dev/null)
    if [ "$phase" != "$phase_prev" ]; then phase_start=$milliseconds; phase_prev="$phase"; fi
    available=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    thermal=$(dumpsys thermalservice 2>/dev/null | awk '
      /^Thermal Status:/ {status=$3; found=1}
      /^Current temperatures from HAL:/ {current=1; next}
      /^Current cooling devices/ {current=0}
      current && /mStatus=/ {s=$0; sub(/^.*mStatus=/,"",s); sub(/[^0-9].*$/,"",s); if(s+0>status) status=s+0; found=1}
      END {if(found) print status; else print -1}')
    pss=-1
    if [ "$(cat "$dir/sample" 2>/dev/null)" = 1 ]; then
        pss=$(awk '/^Pss:/ {print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null)
    fi
    usage=-1; peak=-1
    if [ -n "$cg_mount" ]; then
        usage=$(cat "$cg/memory.usage_in_bytes" 2>/dev/null)
        peak=$(cat "$cg/memory.max_usage_in_bytes" 2>/dev/null)
    fi
    echo "$uptime,$phase,$available,$thermal,${pss:--1},${usage:--1},${peak:--1}" >> "$dir/memory.csv"
    reason=""
    if [ -z "$available" ] || [ "$thermal" -lt 0 ]; then reason=watchdog_measurement_unavailable; fi
    if [ -n "$available" ] && [ "$available" -lt 1048576 ]; then
        if [ "$low_since" -eq 0 ]; then low_since=$milliseconds; fi
        if [ $((milliseconds-low_since)) -ge 3000 ]; then reason=safety_low_memavailable; fi
    else low_since=0; fi
    if [ "$thermal" -ge 3 ]; then reason=safety_severe_thermal; fi
    if [ "$phase" = loading ] && [ $((milliseconds-phase_start)) -ge 300000 ]; then reason=safety_load_timeout; fi
    if [ "$phase" = request ] && [ $((milliseconds-phase_start)) -ge 180000 ]; then reason=safety_request_timeout; fi
    if [ -n "$reason" ]; then
        echo "$reason" > "$dir/stop_reason"
        kill -TERM "$pid"
        sleep 2
        kill -KILL "$pid" 2>/dev/null
        exit 0
    fi
    read finish rest < /proc/uptime
    delay=$(awk -v start="$uptime" -v finish="$finish" 'BEGIN {d=1-(finish-start); print (d>0?d:0)}') || {
        echo watchdog_interval_error > "$dir/stop_reason"; kill -TERM "$pid"; exit 1;
    }
    sleep "$delay" || {
        echo watchdog_sleep_error > "$dir/stop_reason"; kill -TERM "$pid"; exit 1;
    }
done
