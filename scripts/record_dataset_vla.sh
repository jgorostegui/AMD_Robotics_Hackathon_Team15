#!/bin/bash
################################################################################
# Script: record_dataset_vla.sh
# Purpose: Record a multi-task dataset for SmolVLA / VLA (task prompt varies)
#
# Usage:
#   ./scripts/record_dataset_vla.sh [task_id[:episodes] ...]
#
# Examples:
#   ./scripts/record_dataset_vla.sh                      # all tasks, EPISODES_PER_TASK each
#   ./scripts/record_dataset_vla.sh task1:5 task2:5      # custom plan
#   ./scripts/record_dataset_vla.sh task1:5 task2:5 task1:10
#
# Key env vars (optional):
#   - TASKS_JSON:         Path to tasks JSON (default: mission2/tasks_smolvla.json)
#   - EPISODES_PER_TASK:  Episodes per task prompt (default: 4)
#   - DATASET_RESUME:     true/false (default: false) - append into existing dataset dir
#   - DATASET_NAME:       Local folder name + default repo suffix
#
# Notes:
#   - Uses camera keys camera1/camera2/camera3 by default (VLA-friendly).
#   - Resume logic is deterministic: it resumes based on meta/info.json total_episodes
#     and the (task order, episodes-per-task) plan from your tasks JSON.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/record_dataset_vla_${TIMESTAMP}.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"
}

echo ""
echo "=============================================="
echo "  Record Multi-Task Dataset (SmolVLA/VLA)"
echo "  AMD Robotics Hackathon 2025"
echo "=============================================="
echo ""
log "Starting VLA dataset recording..."
log "Log file: $LOG_FILE"
echo ""

if [ -f "$SCRIPT_DIR/.env" ]; then
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.env"
fi

PYTHON=(uv run python3)

parse_video_index() {
    local value="$1"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value"
        return 0
    fi

    local idx
    idx="$(echo "$value" | grep -o '[0-9]*$' || true)"
    if [ -z "$idx" ]; then
        return 1
    fi
    echo "$idx"
}

get_existing_episodes() {
    local dataset_dir="$1"
    local info_json="${dataset_dir}/meta/info.json"
    if [ ! -f "$info_json" ]; then
        echo 0
        return 0
    fi
    "${PYTHON[@]}" -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("total_episodes", 0)))' "$info_json" 2>/dev/null || echo 0
}

parse_tasks_tsv() {
    local tasks_json="$1"
    "${PYTHON[@]}" - "$tasks_json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "tasks" in data:
    data = data["tasks"]

tasks = []

def add(task_id, prompt):
    if task_id is None:
        return
    if prompt is None:
        return
    prompt = str(prompt).strip()
    if not prompt:
        return
    tasks.append((str(task_id), prompt))

if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, dict):
            add(k, v.get("prompt") or v.get("instruction") or v.get("task"))
        else:
            add(k, v)
elif isinstance(data, list):
    for i, item in enumerate(data):
        if isinstance(item, dict):
            task_id = item.get("id") or item.get("name") or item.get("task_id") or f"task{i+1}"
            add(task_id, item.get("prompt") or item.get("instruction") or item.get("task"))
        elif isinstance(item, str):
            add(f"task{i+1}", item)
        else:
            raise SystemExit(f"Unsupported task entry at index {i}: {type(item).__name__}")
else:
    raise SystemExit(f"Unsupported tasks schema: {type(data).__name__}")

if not tasks:
    raise SystemExit("No tasks found in JSON.")

for task_id, prompt in tasks:
    print(f"{task_id}\t{prompt}")
PY
}

# Robot configuration
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-leader_arm}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
LEADER_TYPE="so101_leader"
FOLLOWER_TYPE="so101_follower"

# Camera configuration (defaults match record_dataset.sh)
TOP_CAMERA="${TOP_CAMERA:-/dev/video4}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-/dev/video6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

TOP_INDEX="$(parse_video_index "$TOP_CAMERA" || true)"
SIDE_INDEX="$(parse_video_index "$SIDE_CAMERA" || true)"
GRIPPER_INDEX="$(parse_video_index "$GRIPPER_CAMERA" || true)"

if [ -z "$TOP_INDEX" ] || [ -z "$SIDE_INDEX" ] || [ -z "$GRIPPER_INDEX" ]; then
    log_error "Could not parse camera indices."
    log_info "  TOP_CAMERA=$TOP_CAMERA"
    log_info "  SIDE_CAMERA=$SIDE_CAMERA"
    log_info "  GRIPPER_CAMERA=$GRIPPER_CAMERA"
    log_info "Expected '/dev/videoN' or a numeric index like '0'."
    exit 1
fi

CAMERA_CONFIG="{camera1: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, camera2: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, camera3: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

# Dataset configuration
DATASET_NAME="${DATASET_NAME:-mission2_smolvla_multitask}"
DATASET_ROOT_BASE="${DATASET_ROOT_BASE:-${DATASET_ROOT:-${HOME}/so101_datasets}}"
HF_USER="${HF_USER:-your_username}"
DATASET_RESUME="${DATASET_RESUME:-false}"

DATASET_EPISODE_TIME="${DATASET_EPISODE_TIME:-30}"
DATASET_RESET_TIME="${DATASET_RESET_TIME:-10}"

TASKS_JSON="${TASKS_JSON:-${SCRIPT_DIR}/mission2/tasks_smolvla.json}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-4}"

usage() {
    cat >&2 <<EOF
Usage: $0 [--resume] [task1 task2 ...]

Examples:
  $0                        # all tasks
  $0 task1                  # only task1
  $0 task1 task2            # task1 and task2
  $0 --resume task1         # resume and record task1

Environment:
  EPISODES_PER_TASK=5       # episodes per task (default: 4)
  DATASET_NAME=my_dataset   # dataset folder name
EOF
}

PLAN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --resume)
            DATASET_RESUME=true
            shift
            ;;
        *)
            PLAN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "$DATASET_RESUME" != "true" && "$DATASET_RESUME" != "false" ]]; then
    log_error "DATASET_RESUME must be 'true' or 'false' (got: $DATASET_RESUME)"
    exit 2
fi

if [ ! -f "$TASKS_JSON" ]; then
    log_error "Tasks JSON not found: $TASKS_JSON"
    log_info "Set TASKS_JSON to your tasks file path."
    exit 1
fi

if ! [[ "$EPISODES_PER_TASK" =~ ^[0-9]+$ ]] || [ "$EPISODES_PER_TASK" -le 0 ]; then
    log_error "EPISODES_PER_TASK must be a positive integer (got: $EPISODES_PER_TASK)"
    exit 2
fi

# Create dataset directory (versioned unless resuming)
DATASET_DIR_NAME="$DATASET_NAME"
DATASET_DIR="${DATASET_ROOT_BASE}/${DATASET_DIR_NAME}"
if [ "$DATASET_RESUME" = "true" ]; then
    if [ ! -d "$DATASET_DIR" ]; then
        shopt -s nullglob
        candidates=("${DATASET_ROOT_BASE}/${DATASET_NAME}_v"*)
        shopt -u nullglob

        best=""
        best_v=-1
        for path in "${candidates[@]}"; do
            base="$(basename "$path")"
            if [[ "$base" =~ ^${DATASET_NAME}_v([0-9]+)$ ]]; then
                v="${BASH_REMATCH[1]}"
                if [ "$v" -gt "$best_v" ]; then
                    best="$path"
                    best_v="$v"
                fi
            fi
        done

        if [ -n "$best" ]; then
            DATASET_DIR="$best"
            DATASET_DIR_NAME="$(basename "$best")"
        fi
    fi

    if [ ! -d "$DATASET_DIR" ]; then
        log_error "DATASET_RESUME=true but dataset directory was not found."
        log_info "Tried: $DATASET_DIR"
        log_info "Also searched for: ${DATASET_ROOT_BASE}/${DATASET_NAME}_vN"
        log_info "If this is a new dataset, run once without --resume first."
        exit 2
    fi
else
    if [ -d "$DATASET_DIR" ]; then
        i=1
        while [ -d "${DATASET_ROOT_BASE}/${DATASET_NAME}_v${i}" ]; do
            i=$((i + 1))
        done
        DATASET_DIR_NAME="${DATASET_NAME}_v${i}"
        DATASET_DIR="${DATASET_ROOT_BASE}/${DATASET_DIR_NAME}"
    fi
fi

DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/${DATASET_NAME}}"

# Parse tasks file into an id -> prompt mapping (order preserved separately).
declare -A TASK_PROMPT_BY_ID
declare -a TASK_IDS
while IFS=$'\t' read -r task_id prompt; do
    TASK_IDS+=("$task_id")
    TASK_PROMPT_BY_ID["$task_id"]="$prompt"
done < <(parse_tasks_tsv "$TASKS_JSON")

# Build the recording plan
declare -a PLAN_TASK_IDS

if [ "${#PLAN_ARGS[@]}" -gt 0 ]; then
    for task_id in "${PLAN_ARGS[@]}"; do
        if [ -z "${TASK_PROMPT_BY_ID[$task_id]:-}" ]; then
            log_error "Unknown task: '$task_id'"
            log_info "Available: ${TASK_IDS[*]}"
            exit 2
        fi
        PLAN_TASK_IDS+=("$task_id")
    done
else
    for task_id in "${TASK_IDS[@]}"; do
        PLAN_TASK_IDS+=("$task_id")
    done
fi

TOTAL_PLANNED=$((${#PLAN_TASK_IDS[@]} * EPISODES_PER_TASK))

EXISTING_EPISODES="$(get_existing_episodes "$DATASET_DIR")"

echo ""
log_info "Dataset Configuration:"
log "  Name: $DATASET_DIR_NAME"
log "  Local storage: $DATASET_DIR"
log "  Repository: $DATASET_REPO_ID"
log "  Episode duration: ${DATASET_EPISODE_TIME}s"
log "  Reset time: ${DATASET_RESET_TIME}s"
log "  Tasks file: $TASKS_JSON"
log "  Tasks: ${#TASK_IDS[@]} | Plan steps: ${#PLAN_TASK_IDS[@]} | Planned episodes: $TOTAL_PLANNED"
log "  Resume dataset dir: $DATASET_RESUME (existing episodes: $EXISTING_EPISODES)"
echo ""

log_info "Tasks: ${PLAN_TASK_IDS[*]}"
log_info "Episodes per task: $EPISODES_PER_TASK"
echo ""

log_warning "Make sure your workspace is set up and ready!"
echo ""
read -p "Press Enter to start recording..." </dev/tty
echo ""

# Basic device checks
log_info "Checking robot ports..."
if [ ! -e "$LEADER_PORT" ]; then
    log_error "Leader port not found: $LEADER_PORT"
    exit 1
fi
if [ ! -e "$FOLLOWER_PORT" ]; then
    log_error "Follower port not found: $FOLLOWER_PORT"
    exit 1
fi
log_success "Robot ports found"
echo ""

# Resume on the first call only if the dataset already exists on disk.
RECORD_RESUME=false
if [ -f "$DATASET_DIR/meta/info.json" ]; then
    RECORD_RESUME=true
fi

# Record each task
for task_id in "${PLAN_TASK_IDS[@]}"; do
    task_prompt="${TASK_PROMPT_BY_ID[$task_id]}"

    echo ""
    echo "=============================================="
    log_info "Task: $task_id"
    log "Prompt: $task_prompt"
    echo "=============================================="
    echo ""

    uv run lerobot-record \
        --robot.type="$FOLLOWER_TYPE" \
        --robot.port="$FOLLOWER_PORT" \
        --robot.id="$FOLLOWER_ID" \
        --robot.cameras="$CAMERA_CONFIG" \
        --teleop.type="$LEADER_TYPE" \
        --teleop.port="$LEADER_PORT" \
        --teleop.id="$LEADER_ID" \
        --display_data=true \
        --dataset.repo_id="$DATASET_REPO_ID" \
        --dataset.num_episodes="$EPISODES_PER_TASK" \
        --dataset.episode_time_s="$DATASET_EPISODE_TIME" \
        --dataset.reset_time_s="$DATASET_RESET_TIME" \
        --dataset.single_task="$task_prompt" \
        --dataset.push_to_hub=false \
        --dataset.root="$DATASET_DIR" \
        --resume="$RECORD_RESUME" 2>&1 | tee -a "$LOG_FILE"

    RECORD_RESUME=true
done

echo ""
echo "=============================================="
log_success "VLA dataset recording complete!"
echo ""
log_info "Dataset saved to: $DATASET_DIR"
log_info "Next step: Upload with ./scripts/upload_dataset.sh"
echo "=============================================="
echo ""

exit 0
