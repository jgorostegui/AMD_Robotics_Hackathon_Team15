#!/usr/bin/env python3
"""
Check dataset integrity for VLA/SmolVLA training.

Usage:
    python scripts/check_dataset.py --input ~/so101_datasets/mission2_smolvla_multitask
    python scripts/check_dataset.py --input ~/so101_datasets/mission2_smolvla_multitask --verbose
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("Error: pandas not installed. Run: uv pip install pandas")
    sys.exit(1)


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg: str) -> str:
    return f"{Colors.GREEN}✓{Colors.END} {msg}"


def fail(msg: str) -> str:
    return f"{Colors.RED}✗{Colors.END} {msg}"


def warn(msg: str) -> str:
    return f"{Colors.YELLOW}!{Colors.END} {msg}"


def info(msg: str) -> str:
    return f"{Colors.BLUE}→{Colors.END} {msg}"


def header(msg: str) -> str:
    return f"\n{Colors.BOLD}{'=' * 50}\n{msg}\n{'=' * 50}{Colors.END}"


def check_dataset(input_path: Path, verbose: bool = False) -> bool:
    """Check dataset integrity. Returns True if all checks pass."""

    errors = []
    warnings = []

    print(header(f"Dataset Check: {input_path.name}"))
    print(f"Path: {input_path}\n")

    # 1. Check directory exists
    if not input_path.exists():
        print(fail(f"Dataset directory not found: {input_path}"))
        return False

    # 2. Check meta/info.json
    print(f"{Colors.BOLD}[1/5] Checking meta/info.json{Colors.END}")
    info_path = input_path / "meta" / "info.json"
    if not info_path.exists():
        print(fail("meta/info.json not found"))
        errors.append("Missing meta/info.json")
    else:
        with open(info_path) as f:
            info_data = json.load(f)

        total_episodes = info_data.get("total_episodes", 0)
        total_tasks = info_data.get("total_tasks", 0)
        total_frames = info_data.get("total_frames", 0)
        fps = info_data.get("fps", 0)
        robot_type = info_data.get("robot_type", "unknown")

        print(ok(f"Total episodes: {total_episodes}"))
        print(ok(f"Total tasks: {total_tasks}"))
        print(ok(f"Total frames: {total_frames}"))
        print(ok(f"FPS: {fps}"))
        print(ok(f"Robot type: {robot_type}"))

        # Check camera naming (VLA-friendly)
        features = info_data.get("features", {})
        camera_keys = [k for k in features.keys() if k.startswith("observation.images.")]

        vla_cameras = ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"]
        legacy_cameras = ["observation.images.top", "observation.images.side", "observation.images.gripper"]

        if all(cam in camera_keys for cam in vla_cameras):
            print(ok(f"Camera naming: VLA-friendly (camera1/camera2/camera3)"))
        elif any(cam in camera_keys for cam in legacy_cameras):
            print(warn(f"Camera naming: Legacy (top/side/gripper) - may need renaming for VLA"))
            warnings.append("Legacy camera naming detected")
        else:
            print(info(f"Cameras: {[k.split('.')[-1] for k in camera_keys]}"))

        if verbose:
            for cam_key in camera_keys:
                cam_info = features[cam_key].get("info", {})
                res = f"{cam_info.get('video.width', '?')}x{cam_info.get('video.height', '?')}"
                codec = cam_info.get("video.codec", "?")
                print(f"      {cam_key.split('.')[-1]}: {res} @ {cam_info.get('video.fps', '?')}fps ({codec})")

    # 3. Check meta/tasks.parquet
    print(f"\n{Colors.BOLD}[2/5] Checking meta/tasks.parquet{Colors.END}")
    tasks_path = input_path / "meta" / "tasks.parquet"
    if not tasks_path.exists():
        print(fail("meta/tasks.parquet not found"))
        errors.append("Missing meta/tasks.parquet")
        tasks_df = None
    else:
        tasks_df = pd.read_parquet(tasks_path)
        print(ok(f"Found {len(tasks_df)} task(s)"))

        print(f"\n   {'Task Index':<12} {'Prompt'}")
        print(f"   {'-' * 12} {'-' * 50}")
        for prompt, row in tasks_df.iterrows():
            task_idx = row['task_index']
            prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"   {task_idx:<12} {prompt_short}")

    # 4. Check data parquet files
    print(f"\n{Colors.BOLD}[3/5] Checking data files{Colors.END}")
    data_dir = input_path / "data"
    if not data_dir.exists():
        print(fail("data/ directory not found"))
        errors.append("Missing data/ directory")
        data_df = None
    else:
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            print(fail("No parquet files found in data/"))
            errors.append("No data parquet files")
            data_df = None
        else:
            print(ok(f"Found {len(parquet_files)} data file(s)"))

            # Load all data
            data_df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

            # Episode-to-task mapping
            episode_task_map = data_df.groupby("episode_index")["task_index"].first()
            frames_per_episode = data_df.groupby("episode_index").size()

            print(f"\n   {'Episode':<10} {'Task':<8} {'Frames':<10} {'Duration'}")
            print(f"   {'-' * 10} {'-' * 8} {'-' * 10} {'-' * 10}")

            for ep_idx in sorted(episode_task_map.index):
                task_idx = episode_task_map[ep_idx]
                frames = frames_per_episode[ep_idx]
                duration = frames / fps if fps > 0 else 0
                print(f"   {ep_idx:<10} {task_idx:<8} {frames:<10} {duration:.1f}s")

            # Summary by task
            print(f"\n   {Colors.BOLD}Episodes per task:{Colors.END}")
            task_counts = episode_task_map.value_counts().sort_index()
            for task_idx, count in task_counts.items():
                print(f"      Task {task_idx}: {count} episodes")

    # 5. Check video files
    print(f"\n{Colors.BOLD}[4/5] Checking video files{Colors.END}")
    videos_dir = input_path / "videos"
    if not videos_dir.exists():
        print(fail("videos/ directory not found"))
        errors.append("Missing videos/ directory")
    else:
        video_cameras = [d.name for d in videos_dir.iterdir() if d.is_dir()]
        print(ok(f"Found {len(video_cameras)} camera stream(s)"))

        for cam_dir in sorted(videos_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            video_files = list(cam_dir.rglob("*.mp4"))
            total_size = sum(f.stat().st_size for f in video_files) / (1024 * 1024)
            print(f"      {cam_dir.name}: {len(video_files)} file(s), {total_size:.1f} MB")

    # 6. Check episodes metadata
    print(f"\n{Colors.BOLD}[5/5] Checking episodes metadata{Colors.END}")
    episodes_dir = input_path / "meta" / "episodes"
    if not episodes_dir.exists():
        print(fail("meta/episodes/ directory not found"))
        errors.append("Missing meta/episodes/ directory")
    else:
        episode_files = list(episodes_dir.rglob("*.parquet"))
        print(ok(f"Found {len(episode_files)} episode metadata file(s)"))

    # Summary
    print(header("Summary"))

    if errors:
        print(f"{Colors.RED}Errors ({len(errors)}):{Colors.END}")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"{Colors.YELLOW}Warnings ({len(warnings)}):{Colors.END}")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print(ok("All checks passed! Dataset is ready for training."))
    elif not errors:
        print(warn("Dataset OK with warnings."))
    else:
        print(fail("Dataset has errors that need to be fixed."))

    # Resume info
    print(f"\n{Colors.BOLD}Resume Info:{Colors.END}")
    print(f"  To add more episodes: ./scripts/record_dataset_vla.sh --resume taskN:X")
    print(f"  Dataset path for training: {input_path}")

    return len(errors) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset integrity for VLA/SmolVLA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/check_dataset.py --input ~/so101_datasets/mission2_smolvla_multitask
    python scripts/check_dataset.py --input ~/so101_datasets/mission2_smolvla_multitask --verbose
        """
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed information"
    )

    args = parser.parse_args()

    success = check_dataset(args.input, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
