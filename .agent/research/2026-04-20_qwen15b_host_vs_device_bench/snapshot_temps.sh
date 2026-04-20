#!/bin/bash
# Snapshot CPUSS/GPUSS temps in a machine-readable form (one timestamp per line, zones: name=temp_mC).
SERIAL="${1:-R3CY408S5SB}"
adb -s "$SERIAL" shell '
ts=$(date +%Y-%m-%dT%H:%M:%S)
line="$ts"
for z in /sys/class/thermal/thermal_zone*/type; do
    t=$(cat "$z")
    case "$t" in
        cpuss*|gpuss*|cpu-*-usr|gpu*-usr)
            temp=$(cat "${z%type}temp" 2>/dev/null || echo 0)
            line="$line ${t}=${temp}"
            ;;
    esac
done
echo "$line"
'
