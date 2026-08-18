#!/usr/bin/env bash
# Monitor the CUDA process using the most memory and report only new peaks.
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"
INTERVAL="${INTERVAL:-1}"
TARGET_PID="${1:-}"

usage() {
    cat <<'EOF'
Usage: ./monitor_gpu_highwatermark.sh [PID]

Without PID, waits for the CUDA process using the most memory on GPU_INDEX
(default: 0), then monitors that process until it exits.

Examples:
  ./monitor_gpu_highwatermark.sh
  ./monitor_gpu_highwatermark.sh 334453
  GPU_INDEX=1 INTERVAL=0.5 ./monitor_gpu_highwatermark.sh
EOF
}

if [[ "${TARGET_PID}" == "-h" || "${TARGET_PID}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi was not found in PATH." >&2
    exit 1
fi

if ! awk -v interval="$INTERVAL" 'BEGIN { exit !(interval > 0) }'; then
    echo "INTERVAL must be greater than zero, got: $INTERVAL" >&2
    exit 1
fi

query_compute_processes() {
    nvidia-smi -i "$GPU_INDEX" \
        --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -F ',' '
            NF >= 2 {
                gsub(/[[:space:]]/, "", $1)
                gsub(/[[:space:]]/, "", $2)
                if ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/) {
                    print $1, $2
                }
            }
        '
}

get_process_memory() {
    local pid="$1"
    query_compute_processes | awk -v target_pid="$pid" '$1 == target_pid { print $2; exit }'
}

get_largest_process() {
    query_compute_processes | awk '
        !found || $2 > largest_memory {
            pid = $1
            largest_memory = $2
            found = 1
        }
        END {
            if (found) {
                print pid, largest_memory
            }
        }
    '
}

if [[ -z "$TARGET_PID" ]]; then
    echo "Waiting for a CUDA compute process on GPU $GPU_INDEX..."
    while true; do
        largest_process="$(get_largest_process || true)"
        if [[ -n "$largest_process" ]]; then
            read -r TARGET_PID PEAK_MEMORY <<< "$largest_process"
            break
        fi
        sleep "$INTERVAL"
    done
else
    PEAK_MEMORY="$(get_process_memory "$TARGET_PID" || true)"
    if [[ -z "$PEAK_MEMORY" ]]; then
        echo "PID $TARGET_PID is not a CUDA compute process on GPU $GPU_INDEX." >&2
        exit 1
    fi
fi

echo "Monitoring PID $TARGET_PID on GPU $GPU_INDEX every ${INTERVAL}s. Initial high-water mark: ${PEAK_MEMORY} MiB"

while true; do
    sleep "$INTERVAL"
    current_memory="$(get_process_memory "$TARGET_PID" || true)"

    if [[ -z "$current_memory" ]]; then
        echo "[$(date '+%F %T')] PID=$TARGET_PID exited or no longer uses GPU $GPU_INDEX."
        exit 0
    fi

    if (( current_memory > PEAK_MEMORY )); then
        PEAK_MEMORY="$current_memory"
        echo "[$(date '+%F %T')] PID=$TARGET_PID new_high_watermark=${PEAK_MEMORY}MiB"
    fi
done
