#!/bin/bash
################################################################################
# Script: test_teleoperate.sh
# Purpose: Test teleoperation with SO-101 robot (3 cameras)
#
# Usage: ./scripts/test_teleoperate.sh
#
# Prerequisites:
#   - Robot arms calibrated
#   - All cameras connected
#   - .env file configured
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

# Print header
echo ""
echo "=============================================="
echo "  Test Teleoperation"
echo "  AMD Robotics Hackathon 2025"
echo "=============================================="
echo ""

# Load configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    log_error ".env file not found"
    log_info "Copy from template: cp .env.example .env"
    exit 1
fi

# Configuration
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-leader_arm}"
FOLLOWER_ID="${FOLLOWER_ID:-follower_arm}"
LEADER_TYPE="so101_leader"
FOLLOWER_TYPE="so101_follower"

# Camera configuration (3 cameras)
TOP_CAMERA="${TOP_CAMERA:-/dev/video4}"
SIDE_CAMERA="${SIDE_CAMERA:-/dev/video2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-/dev/video6}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

# Extract video device numbers
TOP_INDEX=$(echo $TOP_CAMERA | grep -o '[0-9]*$')
SIDE_INDEX=$(echo $SIDE_CAMERA | grep -o '[0-9]*$')
GRIPPER_INDEX=$(echo $GRIPPER_CAMERA | grep -o '[0-9]*$')

# Build camera config
CAMERA_CONFIG="{top: {type: opencv, index_or_path: ${TOP_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, side: {type: opencv, index_or_path: ${SIDE_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}, gripper: {type: opencv, index_or_path: ${GRIPPER_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}}}"

# Display configuration
log_info "Configuration:"
echo "  Leader Port:    $LEADER_PORT"
echo "  Follower Port:  $FOLLOWER_PORT"
echo "  Leader ID:      $LEADER_ID"
echo "  Follower ID:    $FOLLOWER_ID"
echo ""
echo "  Top Camera:     $TOP_CAMERA (index $TOP_INDEX)"
echo "  Side Camera:    $SIDE_CAMERA (index $SIDE_INDEX)"
echo "  Gripper Camera: $GRIPPER_CAMERA (index $GRIPPER_INDEX)"
echo ""

# Check devices
log_info "Checking devices..."
CHECKS_PASSED=true

if [ ! -e "$LEADER_PORT" ]; then
    log_error "Leader port not found: $LEADER_PORT"
    CHECKS_PASSED=false
fi

if [ ! -e "$FOLLOWER_PORT" ]; then
    log_error "Follower port not found: $FOLLOWER_PORT"
    CHECKS_PASSED=false
fi

if [ ! -e "$TOP_CAMERA" ]; then
    log_error "Top camera not found: $TOP_CAMERA"
    CHECKS_PASSED=false
fi

if [ ! -e "$SIDE_CAMERA" ]; then
    log_error "Side camera not found: $SIDE_CAMERA"
    CHECKS_PASSED=false
fi

if [ ! -e "$GRIPPER_CAMERA" ]; then
    log_error "Gripper camera not found: $GRIPPER_CAMERA"
    CHECKS_PASSED=false
fi

if [ "$CHECKS_PASSED" = false ]; then
    echo ""
    log_error "Some devices not found. Please check connections."
    log_info "Run diagnostics: ./scripts/diagnose.sh"
    exit 1
fi

log_success "All devices found"
echo ""

# Instructions
log_info "Teleoperation Instructions:"
echo ""
echo "  1. Control the FOLLOWER arm by moving the LEADER arm"
echo "  2. Camera feeds will display in real-time"
echo "  3. Press Ctrl+C to stop"
echo ""
echo "  Safety: Make sure both arms have clearance to move!"
echo ""

read -p "Press Enter to start teleoperation..."
echo ""

# Run teleoperation
log_info "Starting teleoperation..."
echo ""

uv run lerobot-teleoperate \
    --robot.type="$FOLLOWER_TYPE" \
    --robot.port="$FOLLOWER_PORT" \
    --robot.id="$FOLLOWER_ID" \
    --robot.cameras="$CAMERA_CONFIG" \
    --teleop.type="$LEADER_TYPE" \
    --teleop.port="$LEADER_PORT" \
    --teleop.id="$LEADER_ID" \
    --display_data=true

echo ""
echo "=============================================="
log_success "Teleoperation stopped"
echo "=============================================="
echo ""

exit 0
