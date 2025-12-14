#!/bin/bash
################################################################################
# Script: review_dataset.sh
# Purpose: Review/visualize recorded datasets before uploading
#
# Usage:
#   ./scripts/review_dataset.sh                    # Interactive selection
#   ./scripts/review_dataset.sh /path/to/dataset   # Specific dataset
#
# This script helps you:
#   - List all recorded datasets
#   - Show dataset statistics
#   - Visualize episodes (camera feeds + robot state)
#   - Check data quality before uploading
################################################################################

set -euo pipefail

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
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

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}================================================${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}================================================${NC}"
    echo ""
}

# Load configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
fi

DATASET_ROOT="${DATASET_ROOT:-${HOME}/so101_datasets}"

log_header "Dataset Review Tool"

# Get dataset path
if [ $# -ge 1 ]; then
    DATASET_PATH="$1"
else
    # Interactive selection
    echo "Available datasets in: $DATASET_ROOT"
    echo ""

    if [ ! -d "$DATASET_ROOT" ] || [ -z "$(ls -A "$DATASET_ROOT" 2>/dev/null)" ]; then
        log_error "No datasets found in $DATASET_ROOT"
        log_info "Record a dataset first: ./scripts/record_dataset.sh"
        exit 1
    fi

    # List datasets with episode counts
    idx=1
    declare -a DATASETS
    for dir in "$DATASET_ROOT"/*; do
        if [ -d "$dir" ]; then
            DATASET_NAME=$(basename "$dir")
            EPISODE_COUNT=$(find "$dir" -maxdepth 1 -type d -name "episode_*" 2>/dev/null | wc -l)
            echo "  [$idx] $DATASET_NAME ($EPISODE_COUNT episodes)"
            DATASETS[$idx]="$dir"
            ((idx++))
        fi
    done

    if [ ${#DATASETS[@]} -eq 0 ]; then
        log_error "No valid datasets found"
        exit 1
    fi

    echo ""
    read -p "Select dataset number (or 'q' to quit): " selection

    if [ "$selection" = "q" ]; then
        echo "Cancelled"
        exit 0
    fi

    if [ -z "${DATASETS[$selection]:-}" ]; then
        log_error "Invalid selection"
        exit 1
    fi

    DATASET_PATH="${DATASETS[$selection]}"
fi

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    log_error "Dataset not found: $DATASET_PATH"
    exit 1
fi

DATASET_NAME=$(basename "$DATASET_PATH")

log_header "Dataset: $DATASET_NAME"

echo "Path: $DATASET_PATH"
echo ""

# Count episodes
EPISODE_DIRS=$(find "$DATASET_PATH" -maxdepth 1 -type d -name "episode_*" | sort)
EPISODE_COUNT=$(echo "$EPISODE_DIRS" | wc -l)

log_info "Total episodes: $EPISODE_COUNT"
echo ""

# Show episode list
log_info "Episodes:"
for ep_dir in $EPISODE_DIRS; do
    EP_NAME=$(basename "$ep_dir")
    EP_SIZE=$(du -sh "$ep_dir" 2>/dev/null | cut -f1)

    # Count frames (look for camera images)
    FRAME_COUNT=0
    for cam_dir in "$ep_dir"/observation.images.*; do
        if [ -d "$cam_dir" ]; then
            FRAMES=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
            if [ $FRAMES -gt $FRAME_COUNT ]; then
                FRAME_COUNT=$FRAMES
            fi
        fi
    done

    echo "  - $EP_NAME: $FRAME_COUNT frames, $EP_SIZE"
done
echo ""

# Check for metadata
if [ -f "$DATASET_PATH/meta/info.json" ]; then
    log_success "Metadata found"

    if command -v jq &> /dev/null; then
        echo ""
        log_info "Dataset info:"
        cat "$DATASET_PATH/meta/info.json" | jq '.'
    else
        log_info "Install 'jq' for formatted metadata: sudo apt install jq"
    fi
else
    log_info "No metadata file found (meta/info.json)"
fi

echo ""

# Action menu
log_header "Review Options"

echo "What would you like to do?"
echo ""
echo "  [1] Visualize dataset (play episodes with camera feeds)"
echo "  [2] Show dataset statistics"
echo "  [3] Check for issues (incomplete episodes, corrupted data)"
echo "  [4] Export episode list to file"
echo "  [5] Exit"
echo ""

read -p "Select option: " option

case "$option" in
    1)
        log_header "Visualizing Dataset"

        log_info "This will open a window showing:"
        echo "  - Camera feeds (all 3 cameras)"
        echo "  - Robot joint positions"
        echo "  - Actions taken"
        echo ""
        log_info "Use arrow keys to navigate between frames"
        log_info "Press 'q' to quit"
        echo ""

        read -p "Press Enter to start visualization..."

        # Run LeRobot visualizer
        uv run python -m lerobot.scripts.visualize_dataset \
            --root "$DATASET_PATH" \
            --repo-id "$DATASET_NAME"
        ;;

    2)
        log_header "Dataset Statistics"

        # Calculate statistics
        TOTAL_FRAMES=0
        TOTAL_SIZE=0

        for ep_dir in $EPISODE_DIRS; do
            # Count frames
            for cam_dir in "$ep_dir"/observation.images.*; do
                if [ -d "$cam_dir" ]; then
                    FRAMES=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
                    if [ $FRAMES -gt 0 ]; then
                        TOTAL_FRAMES=$((TOTAL_FRAMES + FRAMES))
                        break  # Only count once per episode
                    fi
                fi
            done

            # Sum size
            SIZE_KB=$(du -sk "$ep_dir" 2>/dev/null | cut -f1)
            TOTAL_SIZE=$((TOTAL_SIZE + SIZE_KB))
        done

        TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024))
        AVG_FRAMES_PER_EP=$((TOTAL_FRAMES / EPISODE_COUNT))

        echo "Episodes:           $EPISODE_COUNT"
        echo "Total frames:       $TOTAL_FRAMES"
        echo "Avg frames/episode: $AVG_FRAMES_PER_EP"
        echo "Total size:         ${TOTAL_SIZE_MB} MB"
        echo "Avg size/episode:   $((TOTAL_SIZE_MB / EPISODE_COUNT)) MB"
        echo ""

        # Estimate upload time (assuming 10 Mbps)
        UPLOAD_TIME_SEC=$((TOTAL_SIZE_MB * 8 / 10))
        UPLOAD_TIME_MIN=$((UPLOAD_TIME_SEC / 60))

        echo "Estimated upload time: ~${UPLOAD_TIME_MIN} minutes (at 10 Mbps)"
        echo ""
        ;;

    3)
        log_header "Checking for Issues"

        ISSUES_FOUND=0

        # Check each episode
        for ep_dir in $EPISODE_DIRS; do
            EP_NAME=$(basename "$ep_dir")

            # Check for state file
            if [ ! -f "$ep_dir/observation.state.npy" ]; then
                log_error "$EP_NAME: Missing observation.state.npy"
                ((ISSUES_FOUND++))
            fi

            # Check for action file
            if [ ! -f "$ep_dir/action.npy" ]; then
                log_error "$EP_NAME: Missing action.npy"
                ((ISSUES_FOUND++))
            fi

            # Check cameras have same frame count
            FRAME_COUNTS=()
            for cam_dir in "$ep_dir"/observation.images.*; do
                if [ -d "$cam_dir" ]; then
                    CAM_NAME=$(basename "$cam_dir" | sed 's/observation.images.//')
                    FRAMES=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
                    FRAME_COUNTS+=("$CAM_NAME:$FRAMES")
                fi
            done

            # Verify all cameras have same count
            if [ ${#FRAME_COUNTS[@]} -gt 1 ]; then
                FIRST_COUNT=$(echo "${FRAME_COUNTS[0]}" | cut -d: -f2)
                for fc in "${FRAME_COUNTS[@]}"; do
                    COUNT=$(echo "$fc" | cut -d: -f2)
                    if [ "$COUNT" != "$FIRST_COUNT" ]; then
                        CAM=$(echo "$fc" | cut -d: -f1)
                        log_error "$EP_NAME: Camera '$CAM' has $COUNT frames, expected $FIRST_COUNT"
                        ((ISSUES_FOUND++))
                    fi
                done
            fi
        done

        echo ""
        if [ $ISSUES_FOUND -eq 0 ]; then
            log_success "No issues found! Dataset looks good."
        else
            log_error "Found $ISSUES_FOUND issues"
            log_info "Consider re-recording problematic episodes"
        fi
        echo ""
        ;;

    4)
        OUTPUT_FILE="$SCRIPT_DIR/dataset_${DATASET_NAME}_episodes.txt"

        echo "Dataset: $DATASET_NAME" > "$OUTPUT_FILE"
        echo "Path: $DATASET_PATH" >> "$OUTPUT_FILE"
        echo "Generated: $(date)" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "Episodes ($EPISODE_COUNT total):" >> "$OUTPUT_FILE"
        echo "======================================" >> "$OUTPUT_FILE"

        for ep_dir in $EPISODE_DIRS; do
            EP_NAME=$(basename "$ep_dir")

            # Get frame count
            FRAME_COUNT=0
            for cam_dir in "$ep_dir"/observation.images.*; do
                if [ -d "$cam_dir" ]; then
                    FRAMES=$(ls "$cam_dir"/*.png 2>/dev/null | wc -l)
                    if [ $FRAMES -gt $FRAME_COUNT ]; then
                        FRAME_COUNT=$FRAMES
                    fi
                fi
            done

            echo "$EP_NAME: $FRAME_COUNT frames" >> "$OUTPUT_FILE"
        done

        log_success "Episode list exported to: $OUTPUT_FILE"
        ;;

    5|q|Q)
        echo "Exiting"
        exit 0
        ;;

    *)
        log_error "Invalid option"
        exit 1
        ;;
esac

echo ""
log_header "Review Complete"

echo "Dataset: $DATASET_NAME"
echo "Episodes: $EPISODE_COUNT"
echo ""
log_info "Next steps:"
echo "  - Review more: ./scripts/review_dataset.sh"
echo "  - Record more: ./scripts/record_dataset.sh (resumes automatically)"
echo "  - Upload: ./scripts/upload_dataset.sh $DATASET_PATH"
echo ""

exit 0
