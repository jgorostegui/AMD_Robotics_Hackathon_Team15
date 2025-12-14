#!/usr/bin/env python3
"""
Python equivalent of `scripts/run_inference_vla.sh`.

Runs `uv run lerobot-record ...` for one or more tasks, using the same `.env` keys
and dataset-root resume/versioning logic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

from mission2.robot.lerobot_record import (
    DEFAULT_CAMERA_FPS,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_EVAL_DATASET_NAME,
    DEFAULT_EVAL_EPISODE_TIME_S,
    DEFAULT_EVAL_RESET_TIME_S,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    DEFAULT_FOLLOWER_TYPE,
    DEFAULT_GRIPPER_CAMERA,
    DEFAULT_HF_USER,
    DEFAULT_SIDE_CAMERA,
    DEFAULT_TOP_CAMERA,
    build_vla_camera_config,
    load_dotenv,
    parse_video_index,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_tasks_json(path: str | None) -> Path:
    if path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else _project_root() / candidate

    env_value = os.environ.get("TASKS_JSON", "").strip()
    if env_value:
        candidate = Path(env_value)
        return candidate if candidate.is_absolute() else _project_root() / candidate

    return _project_root() / "mission2/tasks_smolvla.json"


def _load_tasks(tasks_json: Path) -> dict[str, str]:
    with open(tasks_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]

    tasks: list[tuple[str, str]] = []

    def add(task_id: object, prompt: object) -> None:
        if task_id is None or prompt is None:
            return
        task_id_str = str(task_id).strip()
        prompt_str = str(prompt).strip()
        if not task_id_str or not prompt_str:
            return
        tasks.append((task_id_str, prompt_str))

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                add(key, value.get("prompt") or value.get("instruction") or value.get("task"))
            else:
                add(key, value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                task_id = item.get("id") or item.get("name") or item.get("task_id") or f"task{i+1}"
                add(task_id, item.get("prompt") or item.get("instruction") or item.get("task"))
            elif isinstance(item, str):
                add(f"task{i+1}", item)
            else:
                raise ValueError(f"Unsupported task entry at index {i}: {type(item).__name__}")
    else:
        raise ValueError(f"Unsupported tasks schema: {type(data).__name__}")

    if not tasks:
        raise ValueError(f"No tasks found in {tasks_json}")

    return dict(tasks)


def _resolve_eval_dataset_root(*, base: Path, name: str, resume: bool) -> Path:
    root = base / name

    if resume:
        if root.exists():
            return root

        best: Path | None = None
        best_v = -1
        pattern = re.compile(rf"^{re.escape(name)}_v(\d+)$")
        for candidate in base.glob(f"{name}_v*"):
            match = pattern.match(candidate.name)
            if match is None:
                continue
            v = int(match.group(1))
            if v > best_v:
                best_v = v
                best = candidate

        if best is None:
            raise FileNotFoundError("--resume specified but no existing eval dataset found")
        return best

    if not root.exists():
        return root

    i = 1
    while (base / f"{name}_v{i}").exists():
        i += 1
    return base / f"{name}_v{i}"


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="simple_run_lerobot_record.py",
        description="Python equivalent of scripts/run_inference_vla.sh",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing eval dataset root")
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Control loop / dataset FPS. Lower = less compute. Default: 15",
    )
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Enable camera UI / rerun display (default: off).",
    )
    parser.add_argument(
        "--play-sounds",
        action="store_true",
        help="Enable sounds (default: off).",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="Episode time in seconds (overrides EVAL_EPISODE_TIME).",
    )
    parser.add_argument("policy_path", help="Policy path or HuggingFace repo id")
    parser.add_argument("tasks", nargs="*", help="Task ids to run (default: all)")
    args = parser.parse_args()

    project_root = _project_root()
    load_dotenv(project_root / ".env", override=False)

    follower_port = os.environ.get("FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT)
    follower_id = os.environ.get("FOLLOWER_ID", DEFAULT_FOLLOWER_ID)

    top = os.environ.get("TOP_CAMERA", DEFAULT_TOP_CAMERA)
    side = os.environ.get("SIDE_CAMERA", DEFAULT_SIDE_CAMERA)
    gripper = os.environ.get("GRIPPER_CAMERA", DEFAULT_GRIPPER_CAMERA)
    width = int(os.environ.get("CAMERA_WIDTH", str(DEFAULT_CAMERA_WIDTH)))
    height = int(os.environ.get("CAMERA_HEIGHT", str(DEFAULT_CAMERA_HEIGHT)))
    # NOTE: Some webcams/drivers ignore or reject FPS values like 15.
    # Keep camera capture FPS from env (usually 30) and only reduce the control/dataset loop FPS.
    camera_fps = int(os.environ.get("CAMERA_FPS", str(DEFAULT_CAMERA_FPS)))
    fps = int(args.fps)

    camera_config = build_vla_camera_config(
        top=top,
        side=side,
        gripper=gripper,
        width=width,
        height=height,
        fps=camera_fps,
    )

    tasks_json = _resolve_tasks_json(None)
    if not tasks_json.is_file():
        raise SystemExit(f"ERROR: TASKS_JSON not found: {tasks_json}")

    task_map = _load_tasks(tasks_json)

    if args.tasks:
        plan_task_ids = []
        for task_id in args.tasks:
            if task_id not in task_map:
                available = " ".join(task_map.keys())
                raise SystemExit(f"ERROR: Unknown task: '{task_id}'\nAvailable: {available}")
            plan_task_ids.append(task_id)
    else:
        plan_task_ids = list(task_map.keys())

    hf_user = os.environ.get("HF_USER", DEFAULT_HF_USER)
    dataset_name = os.environ.get("EVAL_DATASET_NAME", DEFAULT_EVAL_DATASET_NAME)
    dataset_repo_id = os.environ.get("EVAL_DATASET_REPO_ID", f"{hf_user}/{dataset_name}")

    episodes_per_task = int(os.environ.get("EPISODES_PER_TASK", os.environ.get("EVAL_EPISODES_PER_TASK", os.environ.get("EVAL_NUM_EPISODES", "4"))))
    episode_time_s = int(args.time) if args.time is not None else int(os.environ.get("EVAL_EPISODE_TIME", str(DEFAULT_EVAL_EPISODE_TIME_S)))
    reset_time_s = int(os.environ.get("EVAL_RESET_TIME", str(DEFAULT_EVAL_RESET_TIME_S)))
    push_to_hub = os.environ.get("EVAL_PUSH_TO_HUB", "false").strip().lower() == "true"

    base_root = Path(os.environ.get("EVAL_DATASET_ROOT_BASE", str(Path.home() / "so101_datasets"))).expanduser()
    eval_dataset_root = _resolve_eval_dataset_root(base=base_root, name=dataset_name, resume=bool(args.resume))

    print("")
    print("============================================")
    print(f"  VLA Inference - {Path(args.policy_path).name}")
    print("============================================")
    print(f"Eval dataset: {eval_dataset_root}")
    print(f"Episodes per task: {episodes_per_task}")
    print(f"Tasks: {' '.join(plan_task_ids)}")
    print(f"Dataset/control FPS: {fps} (camera FPS: {camera_fps})")
    print("")

    # Match the shell script behavior: resume=True if meta/info.json exists on first call.
    resume = (eval_dataset_root / "meta/info.json").is_file()

    for task_id in plan_task_ids:
        prompt = task_map[task_id]
        print("")
        print(f"=== Task: {task_id} ===")
        print(f"Prompt: {prompt}")

        cmd = [
            "uv",
            "run",
            "lerobot-record",
            f"--robot.type={DEFAULT_FOLLOWER_TYPE}",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.cameras={camera_config}",
            f"--display_data={'true' if args.display_data else 'false'}",
            f"--play_sounds={'true' if args.play_sounds else 'false'}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.fps={fps}",
            f"--dataset.num_episodes={episodes_per_task}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            f"--dataset.single_task={prompt}",
            f"--dataset.push_to_hub={'true' if push_to_hub else 'false'}",
            f"--dataset.root={eval_dataset_root}",
            f"--policy.path={args.policy_path}",
            "--policy.device=cuda",
            f"--resume={'true' if resume else 'false'}",
        ]

        subprocess.run(cmd, check=True, cwd=str(project_root))
        resume = True

    print("")
    print("============================================")
    print("Inference complete!")
    print(f"Eval dataset: {eval_dataset_root}")
    print("============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
