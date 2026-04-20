#!/bin/bash
# Thermal gate: wait until all cpuss* + gpuss* + cpu-*-usr + gpu*-usr zones <= 38C (38000 milliC)
set -e
SERIAL="${1:-R3CY408S5SB}"
THRESHOLD_MC=38000
SLEEP_S=120

check_once() {
    # returns max temp (milliC) across selected zones
    adb -s "$SERIAL" shell '
        max=0
        maxname=""
        for z in /sys/class/thermal/thermal_zone*/type; do
            t=$(cat "$z")
            case "$t" in
                cpuss*|gpuss*|cpu-*-usr|gpu*-usr)
                    temp=$(cat "${z%type}temp" 2>/dev/null || echo 0)
                    if [ "$temp" -gt "$max" ]; then
                        max=$temp
                        maxname=$t
                    fi
                    ;;
            esac
        done
        echo "$max $maxname"
    '
}

attempts=0
while :; do
    attempts=$((attempts+1))
    out=$(check_once)
    max=$(echo "$out" | awk '{print $1}')
    name=$(echo "$out" | awk '{print $2}')
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    echo "[$ts] attempt=$attempts max=${max}mC (zone=$name) threshold=${THRESHOLD_MC}mC"
    if [ "$max" -le "$THRESHOLD_MC" ]; then
        echo "[$ts] PASS max=${max}mC <= ${THRESHOLD_MC}mC"
        exit 0
    fi
    if [ "$attempts" -ge 12 ]; then
        echo "[$ts] TIMEOUT after $attempts attempts (60min+)"
        exit 2
    fi
    echo "[$ts] Sleeping ${SLEEP_S}s..."
    sleep "$SLEEP_S"
done
