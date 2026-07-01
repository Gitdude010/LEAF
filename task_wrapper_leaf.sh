#!/bin/bash
# =============================================================================
# LEAF Task Wrapper — Single competition runner
# =============================================================================
# Usage:
#   ./task_wrapper_leaf.sh <DATA_DIR> <GPU_ID> <LOCK_FILE_BUSY> <CPU_RANGE>
#
# This script is called by scheduler_leaf.sh for each competition task.
# It sets up the environment, pins CPUs, and launches the LEAF agent.
#
# Arguments:
#   $1 - DATA_DIR: Path to competition's prepared/public directory
#   $2 - GPU_ID: CUDA device ID to use
#   $3 - LOCK_FILE_BUSY: Path to the .busy lock file in /dev/shm
#   $4 - CPU_RANGE: CPU core range (e.g. "0-9") for taskset pinning
# =============================================================================

set -euo pipefail

DATA_DIR="$1"
GPU_ID="$2"
LOCK_FILE_BUSY="$3"
CPU_RANGE="$4"

# Extract task name from data path
TASK_NAME=$(basename "$(dirname "$(dirname "$DATA_DIR")")")

# Calculate number of CPU cores
IFS='-' read -ra CPU_PARTS <<< "$CPU_RANGE"
NUM_CORES=$(( ${CPU_PARTS[1]} - ${CPU_PARTS[0]} + 1 ))

# Set environment
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export OMP_NUM_THREADS="$((NUM_CORES > 8 ? 8 : NUM_CORES))"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Ensure leaf package is importable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH:-}"

# Create log directory
LOG_DIR="${SCRIPT_DIR}/run_logs_leaf"
mkdir -p "$LOG_DIR"

echo "[$(date)] Starting task: $TASK_NAME on GPU $GPU_ID, CPUs $CPU_RANGE"
echo "[$(date)] Data dir: $DATA_DIR"

# Run LEAF agent with CPU pinning and timeout (12 hours)
taskset -c "$CPU_RANGE" timeout 43200 python -m leaf.run \
    data_dir="$DATA_DIR" \
    desc_file="$DATA_DIR/description.md" \
    exp_name="${TASK_NAME}" \
    cpu_number="${NUM_CORES}" \
    > "${LOG_DIR}/${TASK_NAME}.log" 2>&1

EXIT_CODE=$?

# Return the lock file (mark slot as free)
if [ -f "$LOCK_FILE_BUSY" ]; then
    FREE_FILE="${LOCK_FILE_BUSY%.busy}.free"
    mv "$LOCK_FILE_BUSY" "$FREE_FILE"
fi

echo "[$(date)] Task $TASK_NAME finished with exit code $EXIT_CODE"
exit $EXIT_CODE
