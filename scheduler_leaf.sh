#!/bin/bash
# =============================================================================
# LEAF Scheduler — GPU-aware parallel task dispatcher
# =============================================================================
# Usage:
#   ./scheduler_leaf.sh
#
# This script discovers MLEBench competition tasks, manages GPU/CPU resource
# pools via /dev/shm lock files, and dispatches tasks to task_wrapper_leaf.sh.
#
# Configuration: Edit the variables below before running.
# =============================================================================

set -euo pipefail

# ========================= CONFIGURATION =====================================
# Path to MLEBench data root (contains competition directories)
DATA_ROOT="${LEAF_DATA_ROOT:-/path/to/mle-bench/data}"

# Log directory for scheduler output
LEAF_RUN_LOG_DIR="${SCRIPT_DIR:-.}/logs_leaf/run"

# GPU configuration
GPU_IDS=(0)                    # GPU device IDs to use
MAX_PARALLEL=2                 # Maximum concurrent tasks
TOTAL_CPUS=20                  # Total CPU cores to distribute
CPU_OFFSET=0                   # Starting CPU core ID

# API configuration (pass to LEAF via environment)
export LEAF_API_KEY="${LEAF_API_KEY:-}"
export LEAF_BASE_URL="${LEAF_BASE_URL:-}"

# ========================= TASK LIST =========================================
# Add your competition IDs here (must match directory names under DATA_ROOT)
SPECIFIC_TASKS=(
    "spaceship-titanic"
    # Add more competitions:
    # "aerial-cactus-identification"
    # "stanford-covid-vaccine"
    # "tgs-salt-identification-challenge"
)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$LEAF_RUN_LOG_DIR"

NUM_GPUS=${#GPU_IDS[@]}
TASKS_PER_GPU=$(( (MAX_PARALLEL + NUM_GPUS - 1) / NUM_GPUS ))
TOTAL_SLOTS=$((NUM_GPUS * TASKS_PER_GPU))
CPUS_PER_TASK=$((TOTAL_CPUS / TOTAL_SLOTS))

echo "=== LEAF Scheduler ==="
echo "GPUs: ${GPU_IDS[*]}"
echo "Max parallel: $MAX_PARALLEL"
echo "Total slots: $TOTAL_SLOTS"
echo "CPUs per task: $CPUS_PER_TASK"
echo "Data root: $DATA_ROOT"

# ========================= RESOURCE POOL =====================================
SHM_DIR="/dev/shm/leaf_scheduler_$$_$(whoami)"
mkdir -p "$SHM_DIR"

# Create lock files for each slot
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    for task_idx in $(seq 0 $((TASKS_PER_GPU - 1))); do
        SLOT_ID="${gpu_idx}_${task_idx}"
        touch "${SHM_DIR}/slot_${SLOT_ID}.free"
    done
done

# ========================= CLEANUP ===========================================
cleanup() {
    echo "[$(date)] Cleaning up..."
    # Kill all child processes
    if [ ${#active_pids[@]} -gt 0 ]; then
        for pid in "${active_pids[@]}"; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    pkill -9 -P $$ 2>/dev/null || true
    rm -rf "$SHM_DIR"
    echo "[$(date)] Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ========================= TASK DISCOVERY ====================================
TASK_DIRS=()
for TASK_NAME in "${SPECIFIC_TASKS[@]}"; do
    TASK_DIR="${DATA_ROOT}/${TASK_NAME}/prepared/public"
    if [ -d "$TASK_DIR" ]; then
        # Skip if log already exists (idempotent resume)
        if [ -d "${LEAF_RUN_LOG_DIR}/${TASK_NAME}" ]; then
            echo "[SKIP] $TASK_NAME (log exists)"
            continue
        fi
        TASK_DIRS+=("$TASK_DIR")
        echo "[QUEUED] $TASK_NAME"
    else
        echo "[MISSING] $TASK_NAME (data dir not found: $TASK_DIR)"
    fi
done

TOTAL_TASKS=${#TASK_DIRS[@]}
echo "Total tasks to run: $TOTAL_TASKS"

if [ "$TOTAL_TASKS" -eq 0 ]; then
    echo "No tasks to run. Exiting."
    rm -rf "$SHM_DIR"
    exit 0
fi

# ========================= MAIN DISPATCH LOOP ================================
declare -A active_tasks
task_index=0

while [ "$task_index" -lt "$TOTAL_TASKS" ] || [ ${#active_tasks[@]} -gt 0 ]; do
    # Reclaim finished slots
    for pid in "${!active_tasks[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            TASK_NAME="${active_tasks[$pid]}"
            wait "$pid" 2>/dev/null || true
            unset "active_tasks[$pid]"
            echo "[$(date)] ✅ Finished: $TASK_NAME (PID $pid)"
        fi
    done

    # Find free slot
    FREE_SLOT=$(ls "${SHM_DIR}"/*.free 2>/dev/null | head -1)

    if [ -z "$FREE_SLOT" ]; then
        # No free slots, wait
        sleep 2
        continue
    fi

    # Check if we have tasks to dispatch
    if [ "$task_index" -ge "$TOTAL_TASKS" ]; then
        sleep 2
        continue
    fi

    # Parse slot info
    SLOT_FILE=$(basename "$FREE_SLOT")
    SLOT_INFO="${SLOT_FILE#slot_}"
    SLOT_INFO="${SLOT_INFO%.free}"
    GPU_IDX="${SLOT_INFO%%_*}"
    TASK_IDX="${SLOT_INFO##*_}"
    REAL_GPU_ID="${GPU_IDS[$GPU_IDX]}"

    # Calculate CPU range
    SLOT_NUM=$((GPU_IDX * TASKS_PER_GPU + TASK_IDX))
    CPU_START=$((CPU_OFFSET + SLOT_NUM * CPUS_PER_TASK))
    CPU_END=$((CPU_START + CPUS_PER_TASK - 1))
    CURRENT_CPU_RANGE="${CPU_START}-${CPU_END}"

    # Mark slot as busy
    BUSY_TICKET="${FREE_SLOT%.free}.busy"
    mv "$FREE_SLOT" "$BUSY_TICKET"

    # Get task
    TASK_DIR="${TASK_DIRS[$task_index]}"
    TASK_NAME=$(basename "$(dirname "$(dirname "$TASK_DIR")")")
    task_index=$((task_index + 1))

    # Dispatch
    echo "[$(date)] 🚀 Launching: $TASK_NAME → GPU $REAL_GPU_ID, CPUs $CURRENT_CPU_RANGE"
    "${SCRIPT_DIR}/task_wrapper_leaf.sh" \
        "$TASK_DIR" \
        "$REAL_GPU_ID" \
        "$BUSY_TICKET" \
        "$CURRENT_CPU_RANGE" &

    LAUNCH_PID=$!
    active_tasks[$LAUNCH_PID]="$TASK_NAME"

    # Brief pause between launches to avoid IO spikes
    sleep 0.5
done

# Wait for all remaining tasks
echo "[$(date)] All tasks dispatched. Waiting for completion..."
wait

echo "[$(date)] 🎉 All tasks completed!"
rm -rf "$SHM_DIR"
