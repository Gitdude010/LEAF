#!/bin/bash
# =============================================================================
# LEAF — Simple single-task runner
# =============================================================================
# Usage:
#   ./run_leaf.sh <DATA_DIR> [extra args...]
#
# Examples:
#   ./run_leaf.sh ./data/spaceship-titanic/prepared/public
#   ./run_leaf.sh ./data/spaceship-titanic/prepared/public agent.steps=100
#   LEAF_API_KEY=sk-xxx ./run_leaf.sh ./data/...
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <DATA_DIR> [extra config args...]"
    echo ""
    echo "Environment variables:"
    echo "  LEAF_API_KEY     - LLM API key (or pass as agent.api_key=...)"
    echo "  LEAF_BASE_URL    - API base URL (or pass as agent.base_url=...)"
    echo "  LEAF_FAISS_DIR   - Path to gte-small embedding model"
    echo "  GRADING_SERVER_PORT - Grade server port (default: 5005)"
    echo ""
    echo "Examples:"
    echo "  LEAF_API_KEY=sk-xxx $0 ./data/competition/prepared/public"
    echo "  $0 ./data/competition/prepared/public agent.steps=50 agent.draft.model=gpt-4"
    exit 1
fi

DATA_DIR="$1"
shift

# Extract task name from path
TASK_NAME=$(basename "$(dirname "$(dirname "$DATA_DIR")")")

# Set up paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${SCRIPT_DIR}:${PARENT_DIR}:${PYTHONPATH:-}"

# Defaults
DESC_FILE="${DATA_DIR}/description.md"
EXP_NAME="${EXP_NAME:-$TASK_NAME}"

echo "=== LEAF Runner ==="
echo "Task: $TASK_NAME"
echo "Data: $DATA_DIR"
echo "Description: $DESC_FILE"
echo "Experiment: $EXP_NAME"
echo "API Key: ${LEAF_API_KEY:+set (${#LEAF_API_KEY} chars)}"
echo ""

# Run
exec python -m leaf.run \
    data_dir="$DATA_DIR" \
    desc_file="$DESC_FILE" \
    exp_name="$EXP_NAME" \
    "$@"
