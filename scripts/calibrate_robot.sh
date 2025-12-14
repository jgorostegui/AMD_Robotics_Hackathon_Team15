#!/bin/bash
################################################################################
# Script: calibrate_robot.sh
# Purpose: Calibrate SO-101 robot arms (leader and follower)
#
# Origin: Adapted from AMD_Hackathon_Dekuran official template
# https://github.com/ROCm/AMD_Hackathon
#
# Usage:
#   ./scripts/calibrate_robot.sh leader   # Calibrate leader arm
#   ./scripts/calibrate_robot.sh follower # Calibrate follower arm
#   ./scripts/calibrate_robot.sh both     # Calibrate both arms
#
# Prerequisites:
#   - SO-101 robot connected
#   - LeRobot installed
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
echo "  SO-101 Robot Calibration"
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

# Get calibration target
TARGET="${1:-both}"

calibrate_arm() {
    local ARM_TYPE="$1"
    local ARM_PORT="$2"
    local ARM_ID="$3"

    log_info "Calibrating $ARM_TYPE arm..."
    log_info "Port: $ARM_PORT"
    log_info "ID: $ARM_ID"
    echo ""

    # Check port exists
    if [ ! -e "$ARM_PORT" ]; then
        log_error "Port not found: $ARM_PORT"
        log_info "Available ports:"
        ls -la /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  None found"
        return 1
    fi

    echo "Calibration Instructions:"
    if [ "$ARM_TYPE" = "follower" ]; then
        echo "  1. This is the FOLLOWER arm (with GRIPPER)"
    else
        echo "  1. This is the LEADER arm (with HANDLE - you control it)"
    fi
    echo "  2. Move the arm to the HOME position"
    echo "  3. All joints should be at neutral/zero position"
    if [ "$ARM_TYPE" = "follower" ]; then
        echo "  4. The gripper should be fully open"
    fi
    echo ""

    read -p "Press Enter when arm is in HOME position..."
    echo ""

    # Run calibration with correct argument prefix
    if [ "$ARM_TYPE" = "follower" ]; then
        # Follower uses --robot.* arguments
        uv run lerobot-calibrate \
            --robot.type="so101_${ARM_TYPE}" \
            --robot.port="$ARM_PORT" \
            --robot.id="$ARM_ID"
    else
        # Leader uses --teleop.* arguments
        uv run lerobot-calibrate \
            --teleop.type="so101_${ARM_TYPE}" \
            --teleop.port="$ARM_PORT" \
            --teleop.id="$ARM_ID"
    fi

    log_success "$ARM_TYPE arm calibration complete!"
    echo ""
}

case "$TARGET" in
    leader)
        calibrate_arm "leader" "$LEADER_PORT" "$LEADER_ID"
        ;;
    follower)
        calibrate_arm "follower" "$FOLLOWER_PORT" "$FOLLOWER_ID"
        ;;
    both)
        log_info "Calibrating both arms..."
        echo ""
        calibrate_arm "follower" "$FOLLOWER_PORT" "$FOLLOWER_ID"
        calibrate_arm "leader" "$LEADER_PORT" "$LEADER_ID"
        ;;
    *)
        log_error "Unknown target: $TARGET"
        echo "Usage: $0 [leader|follower|both]"
        exit 1
        ;;
esac

echo "=============================================="
log_success "Calibration complete!"
echo ""
log_info "Calibration files saved to: ~/.cache/huggingface/lerobot/calibration/"
log_info "Next step: Test teleoperation with ./scripts/test_teleoperate.sh"
echo "=============================================="
echo ""

exit 0
