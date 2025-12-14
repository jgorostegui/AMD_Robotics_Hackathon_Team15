#!/bin/bash
################################################################################
# Script: upload_dataset.sh
# Purpose: Upload recorded dataset to HuggingFace Hub
#
# Origin: Adapted from AMD_Hackathon_Dekuran official template
# https://github.com/ROCm/AMD_Hackathon
#
# Usage: ./scripts/upload_dataset.sh [DATASET_PATH]
#
# Prerequisites:
#   - .env file with HF_TOKEN and HF_USER set
#   - Dataset recorded with record_dataset.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Print header
echo ""
echo "=============================================="
echo "  Upload Dataset to HuggingFace Hub"
echo "  AMD Robotics Hackathon 2025"
echo "=============================================="
echo ""

# Load configuration
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Get dataset path from argument or prompt
if [ $# -ge 1 ]; then
    DATASET_PATH="$1"
else
    DATASET_ROOT="${DATASET_ROOT:-${HOME}/so101_datasets}"
    echo "Available datasets:"
    ls -d "${DATASET_ROOT}"/*/ 2>/dev/null || echo "  No datasets found in $DATASET_ROOT"
    echo ""
    read -p "Enter dataset path: " DATASET_PATH
fi

# Verify dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset not found: $DATASET_PATH" >&2
    exit 1
fi

# Normalize to absolute path (script `cd`s later; relative paths would break)
DATASET_PATH="$(cd "$DATASET_PATH" && pwd)"

# Extract dataset name from path
DATASET_NAME=$(basename "$DATASET_PATH")
HF_USER="${HF_USER:-}"
DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"

echo "Dataset: $DATASET_NAME"
echo "Path: $DATASET_PATH"
echo "Target repo: $DATASET_REPO_ID"
echo ""

# Check HuggingFace token
if [ -z "${HF_TOKEN:-}" ] || [ "$HF_TOKEN" = "your_huggingface_token_here" ]; then
    echo "ERROR: HF_TOKEN not configured in $SCRIPT_DIR/.env" >&2
    echo "Get a token from: https://huggingface.co/settings/tokens" >&2
    exit 1
fi
if [ -z "$HF_USER" ] || [ "$HF_USER" = "your_username" ]; then
    echo "ERROR: HF_USER not configured in $SCRIPT_DIR/.env" >&2
    exit 1
fi
export HF_TOKEN

# Prefer the repo venv's huggingface-cli (avoids uv/cache issues); fall back to PATH or uv.
HF_CLI=()
if [ -x "$SCRIPT_DIR/.venv/bin/huggingface-cli" ]; then
    HF_CLI=("$SCRIPT_DIR/.venv/bin/huggingface-cli")
elif command -v huggingface-cli &>/dev/null; then
    HF_CLI=(huggingface-cli)
elif command -v uv &>/dev/null; then
    HF_CLI=(uv run huggingface-cli)
else
    echo "ERROR: huggingface-cli not found (install deps with 'uv sync' or 'pip install -U \"huggingface_hub[cli]\"')" >&2
    exit 1
fi

# Confirm upload
echo "This will upload the dataset to: https://huggingface.co/datasets/${DATASET_REPO_ID}"
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Upload dataset
echo "Uploading dataset..."
echo ""

cd "$SCRIPT_DIR"
"${HF_CLI[@]}" upload "$DATASET_REPO_ID" "$DATASET_PATH" --repo-type dataset

# LeRobot training requires datasets to be tagged with their `codebase_version` (from meta/info.json).
INFO_JSON="${DATASET_PATH}/meta/info.json"
if [ -f "$INFO_JSON" ]; then
    PY=()
    if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
        PY=("$SCRIPT_DIR/.venv/bin/python")
    else
        PY=(python3)
    fi
    CODEBASE_VERSION="$("${PY[@]}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["codebase_version"])' "$INFO_JSON" 2>/dev/null || true)"
    if [ -n "${CODEBASE_VERSION:-}" ]; then
        echo ""
        echo "Ensuring dataset tag exists: ${CODEBASE_VERSION}"
        if ! "${HF_CLI[@]}" tag "$DATASET_REPO_ID" "$CODEBASE_VERSION" --repo-type dataset -y >/dev/null 2>&1; then
            echo "WARNING: Could not create tag '${CODEBASE_VERSION}' for ${DATASET_REPO_ID}." >&2
            echo "  LeRobot training may fail until you tag the dataset:" >&2
            echo "  ${HF_CLI[*]} tag \"$DATASET_REPO_ID\" \"$CODEBASE_VERSION\" --repo-type dataset -y" >&2
        fi
    fi
fi

echo ""
echo "=============================================="
echo "Dataset uploaded successfully!"
echo ""
echo "View at: https://huggingface.co/datasets/${DATASET_REPO_ID}"
echo "=============================================="
echo ""

exit 0
