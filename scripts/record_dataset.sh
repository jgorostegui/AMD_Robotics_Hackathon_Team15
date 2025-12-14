#!/bin/bash
################################################################################
# Script: record_dataset.sh
# Purpose: Record training dataset with teleoperation
#
# Origin: Adapted from AMD_Hackathon_Dekuran official template
# https://github.com/ROCm/AMD_Hackathon
#
# Usage: ./scripts/record_dataset.sh
#
# Prerequisites:
#   - SO-101 robot connected (leader + follower arms)
#   - Cameras connected and configured
#   - .env file with HF_TOKEN set
################################################################################

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/record_dataset_${TIMESTAMP}.log"

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

# Print header
echo ""
echo "=============================================="
echo "  Record Training Dataset"
echo "  AMD Robotics Hackathon 2025"
echo "=============================================="
echo ""
log "Starting dataset recording..."
log "Log file: $LOG_FILE"
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

# Camera configuration (3 cameras: top, side, gripper)
TOP_CAMERA="${TOP_CAMERA:-/dev/video4}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-/dev/video6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# Dataset configuration
DATASET_NAME="${DATASET_NAME:-my_dataset}"
DATASET_TASK="${DATASET_TASK:-Pick and place task}"
DATASET_NUM_EPISODES="${DATASET_NUM_EPISODES:-50}"
DATASET_EPISODE_TIME="${DATASET_EPISODE_TIME:-30}"
DATASET_RESET_TIME="${DATASET_RESET_TIME:-10}"
DATASET_ROOT="${DATASET_ROOT:-${HOME}/so101_datasets}"
HF_USER="${HF_USER:-your_username}"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

# Build camera config string for lerobot (3 cameras)
# Extract video device numbers from paths
TOP_INDEX=$(echo $TOP_CAMERA | grep -o '[0-9]*$')
SIDE_INDEX=$(echo $SIDE_CAMERA | grep -o '[0-9]*$')
GRIPPER_INDEX=$(echo $GRIPPER_CAMERA | grep -o '[0-9]*$')

# Use YAML-style config (matches your working command)
CAMERA_CONFIG="{top: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, side: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

# Check HuggingFace token
log_info "Checking HuggingFace configuration..."
if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    log_warning "HF_TOKEN not configured in .env file"
    log_info "  Set HF_TOKEN in .env for automatic authentication"
    log_info "  (You can continue, but upload may fail later)"
else
    log_success "HF_TOKEN is configured"
fi
echo ""

# Check devices
log_info "Checking devices..."
if [ ! -e "$LEADER_PORT" ]; then
    log_error "Leader port not found: $LEADER_PORT"
    log_info "Available serial ports:"
    ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  None found"
    exit 1
fi

if [ ! -e "$FOLLOWER_PORT" ]; then
    log_error "Follower port not found: $FOLLOWER_PORT"
    exit 1
fi

log_success "Robot ports found"
echo ""

# Display configuration
log_info "Dataset Configuration:"
log "  Name: $DATASET_NAME"
log "  Task: $DATASET_TASK"
log "  Episodes: $DATASET_NUM_EPISODES"
log "  Episode duration: ${DATASET_EPISODE_TIME}s"
log "  Reset time: ${DATASET_RESET_TIME}s"
log "  Repository: $DATASET_REPO_ID"

# Create unique dataset directory
EFFECTIVE_DATASET_ROOT="${DATASET_ROOT}/${DATASET_NAME}"
if [ -d "${EFFECTIVE_DATASET_ROOT}" ]; then
    i=1
    while [ -d "${EFFECTIVE_DATASET_ROOT}_v${i}" ]; do
        i=$((i + 1))
    done
    EFFECTIVE_DATASET_ROOT="${EFFECTIVE_DATASET_ROOT}_v${i}"
fi
export DATASET_ROOT="${EFFECTIVE_DATASET_ROOT}"
log "  Local storage: $DATASET_ROOT"
echo ""

# Instructions
log_info "Recording Instructions:"
echo ""
echo "  1. Each episode lasts ${DATASET_EPISODE_TIME} seconds"
echo "  2. Perform your task: $DATASET_TASK"
echo "  3. After each episode, you have ${DATASET_RESET_TIME} seconds to reset"
echo ""
log_info "Keyboard Controls During Recording:"
echo "  • Left Arrow (←):  Cancel current episode and re-record it"
echo "  • Right Arrow (→): Early stop episode/reset and move to next"
echo "  • Escape (ESC):    Stop session immediately"
echo ""
echo "  More info: https://huggingface.co/docs/lerobot/il_robots#record-a-dataset"
echo ""
log_warning "Make sure your workspace is set up and ready!"
echo ""

read -p "Press Enter to start recording..."
echo ""

# Run dataset recording
log_info "Starting dataset recording..."
log "This will record $DATASET_NUM_EPISODES episodes"
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
    --dataset.num_episodes="$DATASET_NUM_EPISODES" \
    --dataset.episode_time_s="$DATASET_EPISODE_TIME" \
    --dataset.reset_time_s="$DATASET_RESET_TIME" \
    --dataset.single_task="$DATASET_TASK" \
    --dataset.push_to_hub=false \
    --dataset.root="$DATASET_ROOT" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "=============================================="
log_success "Dataset recording complete!"
echo ""
log_info "Dataset saved to: $DATASET_ROOT"
log_info "Next step: Upload with ./scripts/upload_dataset.sh"
echo "=============================================="
echo ""

exit 0
