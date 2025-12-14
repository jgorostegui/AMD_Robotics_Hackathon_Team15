#!/bin/bash
################################################################################
# Script: run_inference_vla.sh
# Purpose: Run policy inference on the real robot (3-camera setup, task prompts from JSON)
#
# Usage:
#   ./scripts/run_inference_vla.sh <POLICY_PATH_OR_REPO_ID> [task_id[:episodes] ...]
#
# Examples:
#   ./scripts/run_inference_vla.sh jlamperez/smolvla_mission2              # all tasks
#   ./scripts/run_inference_vla.sh jlamperez/smolvla_mission2 task1        # only task1
#   ./scripts/run_inference_vla.sh jlamperez/smolvla_mission2 task1:5 task2:3
#
# Notes:
#   - Reads robot + camera settings from .env (same keys as record_dataset.sh)
#   - Requires TASKS_JSON (default: mission2/tasks_smolvla.json). Aborts if missing.
#   - Runs tasks sequentially and records EPISODES_PER_TASK per prompt into a single eval dataset (resume mode).
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
fi

usage() {
  cat >&2 <<EOF
Usage: $0 <POLICY_PATH> [task1 task2 ...]

Examples:
  $0 jlamperez/smolvla_mission2                # all tasks
  $0 jlamperez/smolvla_mission2 task1          # only task1
  $0 jlamperez/smolvla_mission2 task1 task2    # task1 and task2
  $0 --resume jlamperez/smolvla_mission2 task1 # resume eval dataset

Environment:
  EPISODES_PER_TASK=5  # episodes per task (default: 4)
EOF
}

# Parse arguments
POLICY_PATH=""
EVAL_RESUME=false
PLAN_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --resume)
      EVAL_RESUME=true
      shift
      ;;
    *)
      if [ -z "$POLICY_PATH" ]; then
        POLICY_PATH="$1"
      else
        PLAN_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [ -z "$POLICY_PATH" ]; then
  usage
  exit 2
fi

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
FOLLOWER_TYPE="so101_follower"

PYTHON=(uv run python3)
LEROBOT_RECORD=(uv run lerobot-record)

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
    if task_id is None or prompt is None:
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
  echo "ERROR: Could not parse camera indices from TOP_CAMERA/SIDE_CAMERA/GRIPPER_CAMERA" >&2
  echo "  TOP_CAMERA=$TOP_CAMERA" >&2
  echo "  SIDE_CAMERA=$SIDE_CAMERA" >&2
  echo "  GRIPPER_CAMERA=$GRIPPER_CAMERA" >&2
  exit 1
fi

# VLA-friendly camera keys (avoid rename-map churn downstream).
CAMERA_CONFIG="{camera1: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, camera2: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, camera3: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

HF_USER="${HF_USER:-localuser}"

TASKS_JSON="${TASKS_JSON:-${SCRIPT_DIR}/mission2/tasks_smolvla.json}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-${EVAL_EPISODES_PER_TASK:-${EVAL_NUM_EPISODES:-4}}}"

if ! [[ "$EPISODES_PER_TASK" =~ ^[0-9]+$ ]] || [ "$EPISODES_PER_TASK" -le 0 ]; then
  echo "ERROR: EPISODES_PER_TASK must be a positive integer (got: $EPISODES_PER_TASK)" >&2
  exit 2
fi

EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-eval_vla_tasks}"
EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-${HF_USER}/${EVAL_DATASET_NAME}}"
EVAL_EPISODE_TIME="${EVAL_EPISODE_TIME:-30}"
EVAL_RESET_TIME="${EVAL_RESET_TIME:-10}"
EVAL_PUSH_TO_HUB="${EVAL_PUSH_TO_HUB:-false}"
POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-1}"

EVAL_DATASET_ROOT_BASE="${EVAL_DATASET_ROOT_BASE:-${HOME}/so101_datasets}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT_BASE}/${EVAL_DATASET_NAME}"

# Handle resume vs new dataset
if [ "$EVAL_RESUME" = "true" ]; then
  # Find existing dataset to resume
  if [ ! -d "$EVAL_DATASET_ROOT" ]; then
    shopt -s nullglob
    candidates=("${EVAL_DATASET_ROOT_BASE}/${EVAL_DATASET_NAME}_v"*)
    shopt -u nullglob
    best=""
    best_v=-1
    for path in "${candidates[@]}"; do
      base="$(basename "$path")"
      if [[ "$base" =~ ^${EVAL_DATASET_NAME}_v([0-9]+)$ ]]; then
        v="${BASH_REMATCH[1]}"
        if [ "$v" -gt "$best_v" ]; then
          best="$path"
          best_v="$v"
        fi
      fi
    done
    if [ -n "$best" ]; then
      EVAL_DATASET_ROOT="$best"
    fi
  fi
  if [ ! -d "$EVAL_DATASET_ROOT" ]; then
    echo "ERROR: --resume specified but no existing eval dataset found" >&2
    exit 2
  fi
else
  # Create new versioned dataset if exists
  if [ -d "$EVAL_DATASET_ROOT" ]; then
    i=1
    while [ -d "${EVAL_DATASET_ROOT}_v${i}" ]; do
      i=$((i + 1))
    done
    EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT}_v${i}"
  fi
fi

if [ ! -f "$TASKS_JSON" ]; then
  echo "ERROR: TASKS_JSON not found: $TASKS_JSON" >&2
  echo "  Set TASKS_JSON=/path/to/tasks.json" >&2
  exit 1
fi

# Parse tasks into id -> prompt mapping
declare -A TASK_PROMPT_BY_ID
declare -a TASK_IDS
while IFS=$'\t' read -r task_id prompt; do
  [ -z "${task_id:-}" ] && continue
  [ -z "${prompt:-}" ] && continue
  TASK_IDS+=("$task_id")
  TASK_PROMPT_BY_ID["$task_id"]="$prompt"
done < <(parse_tasks_tsv "$TASKS_JSON")

if [ "${#TASK_IDS[@]}" -eq 0 ]; then
  echo "ERROR: No tasks parsed from TASKS_JSON: $TASKS_JSON" >&2
  exit 1
fi

# Build the execution plan
declare -a PLAN_TASK_IDS

if [ "${#PLAN_ARGS[@]}" -gt 0 ]; then
  for task_id in "${PLAN_ARGS[@]}"; do
    if [ -z "${TASK_PROMPT_BY_ID[$task_id]:-}" ]; then
      echo "ERROR: Unknown task: '$task_id'" >&2
      echo "Available: ${TASK_IDS[*]}" >&2
      exit 2
    fi
    PLAN_TASK_IDS+=("$task_id")
  done
else
  # No task args: run all tasks
  for task_id in "${TASK_IDS[@]}"; do
    PLAN_TASK_IDS+=("$task_id")
  done
fi

echo "" >&2
echo "============================================" >&2
echo "  VLA Inference - $(basename "$POLICY_PATH")" >&2
echo "============================================" >&2
echo "Eval dataset: $EVAL_DATASET_ROOT" >&2
echo "Episodes per task: $EPISODES_PER_TASK" >&2
echo "Tasks: ${PLAN_TASK_IDS[*]}" >&2
echo "" >&2

# Resume on the first call only if dataset already exists
RESUME=false
if [ -f "$EVAL_DATASET_ROOT/meta/info.json" ]; then
  RESUME=true
fi

for task_id in "${PLAN_TASK_IDS[@]}"; do
  task_prompt="${TASK_PROMPT_BY_ID[$task_id]}"

  echo "" >&2
  echo "=== Task: $task_id ===" >&2
  echo "Prompt: $task_prompt" >&2

  uv run lerobot-record \
    --robot.type="$FOLLOWER_TYPE" \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="$CAMERA_CONFIG" \
    --display_data=true \
    --dataset.repo_id="$EVAL_DATASET_REPO_ID" \
    --dataset.num_episodes="$EPISODES_PER_TASK" \
    --dataset.episode_time_s="$EVAL_EPISODE_TIME" \
    --dataset.reset_time_s="$EVAL_RESET_TIME" \
    --dataset.single_task="$task_prompt" \
    --dataset.push_to_hub="$EVAL_PUSH_TO_HUB" \
    --dataset.root="$EVAL_DATASET_ROOT" \
    --policy.path="$POLICY_PATH" \
    --policy.device="cuda" \
    --resume="$RESUME"

  RESUME=true
done

echo "" >&2
echo "============================================" >&2
echo "Inference complete!" >&2
echo "Eval dataset: $EVAL_DATASET_ROOT" >&2
echo "============================================" >&2
