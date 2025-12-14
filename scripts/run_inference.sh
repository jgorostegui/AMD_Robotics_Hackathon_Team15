#!/bin/bash
################################################################################
# Script: run_inference.sh
# Purpose: Run policy inference on the real robot (3-camera setup)
#
# Usage:
#   ./scripts/run_inference.sh <POLICY_PATH_OR_REPO_ID>
#
# Examples:
#   ./scripts/run_inference.sh models/act_mission1_pick_place
#   ./scripts/run_inference.sh jlamperez/act_mission1_pick_place
#
# Notes:
#   - Reads robot + camera settings from .env (same keys as record_dataset.sh)
#   - Runs `lerobot-record` with a policy (evaluation mode)
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/.env"
fi

POLICY_PATH="${1:-${POLICY_PATH:-}}"
if [ -z "$POLICY_PATH" ]; then
  echo "Usage: $0 <POLICY_PATH_OR_REPO_ID>" >&2
  exit 2
fi

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
FOLLOWER_TYPE="so101_follower"

TOP_CAMERA="${TOP_CAMERA:-/dev/video4}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-/dev/video6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

TOP_INDEX="$(echo "$TOP_CAMERA" | grep -o '[0-9]*$' || true)"
SIDE_INDEX="$(echo "$SIDE_CAMERA" | grep -o '[0-9]*$' || true)"
GRIPPER_INDEX="$(echo "$GRIPPER_CAMERA" | grep -o '[0-9]*$' || true)"

if [ -z "$TOP_INDEX" ] || [ -z "$SIDE_INDEX" ] || [ -z "$GRIPPER_INDEX" ]; then
  echo "ERROR: Could not parse /dev/video* indices from TOP_CAMERA/SIDE_CAMERA/GRIPPER_CAMERA" >&2
  echo "  TOP_CAMERA=$TOP_CAMERA" >&2
  echo "  SIDE_CAMERA=$SIDE_CAMERA" >&2
  echo "  GRIPPER_CAMERA=$GRIPPER_CAMERA" >&2
  exit 1
fi

CAMERA_CONFIG="{top: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, side: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

DATASET_TASK="${DATASET_TASK:-Pick and place task}"
HF_USER="${HF_USER:-localuser}"

EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-eval_mission1_pick_place}"
EVAL_DATASET_REPO_ID="${EVAL_DATASET_REPO_ID:-${HF_USER}/${EVAL_DATASET_NAME}}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-10}"
EVAL_EPISODE_TIME="${EVAL_EPISODE_TIME:-30}"
EVAL_RESET_TIME="${EVAL_RESET_TIME:-10}"
EVAL_PUSH_TO_HUB="${EVAL_PUSH_TO_HUB:-false}"

EVAL_DATASET_ROOT_BASE="${EVAL_DATASET_ROOT_BASE:-${HOME}/so101_datasets}"
EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT_BASE}/${EVAL_DATASET_NAME}"
if [ -d "$EVAL_DATASET_ROOT" ]; then
  i=1
  while [ -d "${EVAL_DATASET_ROOT}_v${i}" ]; do
    i=$((i + 1))
  done
  EVAL_DATASET_ROOT="${EVAL_DATASET_ROOT}_v${i}"
fi

RUN_PREFIX=()
if command -v uv >/dev/null 2>&1; then
  RUN_PREFIX=(uv run)
fi

set -x
"${RUN_PREFIX[@]}" lerobot-record \
  --robot.type="$FOLLOWER_TYPE" \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.cameras="$CAMERA_CONFIG" \
  --display_data=true \
  --dataset.repo_id="$EVAL_DATASET_REPO_ID" \
  --dataset.num_episodes="$EVAL_NUM_EPISODES" \
  --dataset.episode_time_s="$EVAL_EPISODE_TIME" \
  --dataset.reset_time_s="$EVAL_RESET_TIME" \
  --dataset.single_task="$DATASET_TASK" \
  --dataset.push_to_hub="$EVAL_PUSH_TO_HUB" \
  --dataset.root="$EVAL_DATASET_ROOT" \
  --policy.path="$POLICY_PATH" \
  --policy.device="cuda"
