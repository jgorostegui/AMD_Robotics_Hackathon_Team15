#!/bin/bash
################################################################################
# Script: record_dataset_interactive.sh
# Purpose: Interactive dataset recording with per-episode confirmation
#
# This script records episodes ONE AT A TIME with manual confirmation.
# After each episode, you decide: Keep, Retry, or Stop
#
# Usage: ./scripts/record_dataset_interactive.sh
################################################################################

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Print header
echo ""
echo "=============================================="
echo "  Interactive Dataset Recording"
echo "  AMD Robotics Hackathon 2025"
echo "=============================================="
echo ""

# Load configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

# Configuration with defaults
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-leader_arm}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
LEADER_TYPE="so101_leader"
FOLLOWER_TYPE="so101_follower"

# Camera configuration
TOP_CAMERA="${TOP_CAMERA:-/dev/video4}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-/dev/video6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# Dataset configuration
DATASET_NAME="${DATASET_NAME:-my_dataset}"
DATASET_TASK="${DATASET_TASK:-Pick and place task}"
TARGET_EPISODES="${DATASET_NUM_EPISODES:-50}"
DATASET_EPISODE_TIME="${DATASET_EPISODE_TIME:-30}"
DATASET_RESET_TIME="${DATASET_RESET_TIME:-10}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/so101_datasets}"
HF_USER="${HF_USER:-your_username}"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

# Build camera config
TOP_INDEX=$(echo $TOP_CAMERA | grep -o '[0-9]*$')
SIDE_INDEX=$(echo $SIDE_CAMERA | grep -o '[0-9]*$')
GRIPPER_INDEX=$(echo $GRIPPER_CAMERA | grep -o '[0-9]*$')

CAMERA_CONFIG="{top: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, side: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

# Create dataset directory
EFFECTIVE_DATASET_ROOT="${DATASET_ROOT}/${DATASET_NAME}"
if [ -d "${EFFECTIVE_DATASET_ROOT}" ]; then
    i=1
    while [ -d "${EFFECTIVE_DATASET_ROOT}_v${i}" ]; do
        i=$((i + 1))
    done
    EFFECTIVE_DATASET_ROOT="${EFFECTIVE_DATASET_ROOT}_v${i}"
fi
mkdir -p "${EFFECTIVE_DATASET_ROOT}"
export DATASET_ROOT="${EFFECTIVE_DATASET_ROOT}"

# Display configuration
log_info "Configuration:"
echo "  Dataset: $DATASET_NAME"
echo "  Task: $DATASET_TASK"
echo "  Target Episodes: $TARGET_EPISODES"
echo "  Episode Time: ${DATASET_EPISODE_TIME}s"
echo "  Storage: $DATASET_ROOT"
echo ""

# Count existing episodes
CURRENT_EPISODE=0
if [ -d "$DATASET_ROOT" ]; then
    EXISTING=$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "episode_*" | wc -l)
    CURRENT_EPISODE=$EXISTING
    if [ $EXISTING -gt 0 ]; then
        log_info "Found $EXISTING existing episodes"
        log_info "Continuing from episode $CURRENT_EPISODE"
        echo ""
    fi
fi

# Instructions
log_info "Interactive Recording Mode:"
echo ""
echo "  After each episode, you will be asked:"
echo "    [K] Keep episode and continue"
echo "    [R] Retry episode (discard and redo)"
echo "    [S] Stop recording"
echo ""
log_warning "Make sure your workspace is ready!"
echo ""

read -p "Press Enter to start..."
echo ""

# Recording loop
EPISODES_RECORDED=0

while [ $CURRENT_EPISODE -lt $TARGET_EPISODES ]; do
    # Get actual latest episode number from directory
    LATEST_EPISODE=$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | sed 's/.*episode_//' | sort -n | tail -1)
    if [ -z "$LATEST_EPISODE" ]; then
        NEXT_EPISODE=0
    else
        NEXT_EPISODE=$((LATEST_EPISODE + 1))
    fi

    echo ""
    echo "=============================================="
    echo "  Recording Episode $NEXT_EPISODE / $TARGET_EPISODES"
    echo "=============================================="
    echo ""

    # Record single episode
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
        --dataset.num_episodes=1 \
        --dataset.episode_time_s="$DATASET_EPISODE_TIME" \
        --dataset.reset_time_s="$DATASET_RESET_TIME" \
        --dataset.single_task="$DATASET_TASK" \
        --dataset.push_to_hub=false \
        --dataset.root="$DATASET_ROOT"

    RECORD_EXIT=$?

    if [ $RECORD_EXIT -ne 0 ]; then
        log_error "Recording failed"
        read -p "Try again? (y/n): " retry
        if [[ ! $retry =~ ^[Yy]$ ]]; then
            break
        fi
        continue
    fi

    # Find the episode that was just created (highest numbered)
    JUST_RECORDED=$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | sed 's/.*episode_//' | sort -n | tail -1)

    echo ""
    echo "=============================================="
    log_info "Episode $JUST_RECORDED complete!"
    echo "=============================================="
    echo ""

    # Decision
    while true; do
        read -p "Keep [K], Retry [R], or Stop [S]? " -n 1 -r decision
        echo ""

        case "$decision" in
            [Kk])
                log_success "Episode $JUST_RECORDED kept"
                ((CURRENT_EPISODE++))
                ((EPISODES_RECORDED++))
                break
                ;;
            [Rr])
                log_warning "Discarding episode $JUST_RECORDED"
                rm -rf "$DATASET_ROOT/episode_$JUST_RECORDED"
                log_info "Ready to retry"
                # Don't increment CURRENT_EPISODE - we'll retry
                break
                ;;
            [Ss])
                log_info "Stopping recording"
                echo ""
                TOTAL_NOW=$(find "$DATASET_ROOT" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | wc -l)
                log_success "Total episodes in dataset: $TOTAL_NOW"
                echo "  Dataset: $DATASET_ROOT"
                echo ""
                exit 0
                ;;
            *)
                echo "Invalid choice. Use K, R, or S."
                ;;
        esac
    done

    # Reset prompt
    if [ $CURRENT_EPISODE -lt $TARGET_EPISODES ]; then
        echo ""
        log_info "Reset workspace for next episode"
        read -p "Press Enter when ready..."
    fi
done

echo ""
echo "=============================================="
log_success "Recording complete!"
echo ""
log_info "Total episodes: $EPISODES_RECORDED"
log_info "Dataset: $DATASET_ROOT"
log_info "Next step: Review with ./scripts/review_dataset.sh"
echo "=============================================="
echo ""

exit 0
