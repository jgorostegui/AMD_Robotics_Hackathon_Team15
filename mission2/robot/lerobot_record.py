"""Helpers for invoking `lerobot-record` consistently.

This mirrors the environment keys used by `scripts/run_inference_vla.sh`.

Behavior stays intentionally simple:
- one `lerobot-record` call per invocation
- dataset root/resume selection is handled by the caller
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections import deque
from datetime import datetime
from pathlib import Path


DEFAULT_FOLLOWER_TYPE = "so101_follower"
DEFAULT_FOLLOWER_PORT = "/dev/ttyACM1"
DEFAULT_FOLLOWER_ID = "follower_arm"

DEFAULT_TOP_CAMERA = "/dev/video4"
DEFAULT_SIDE_CAMERA = "/dev/video2"
DEFAULT_GRIPPER_CAMERA = "/dev/video6"
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480
DEFAULT_CAMERA_FPS = 30

DEFAULT_HF_USER = "localuser"
DEFAULT_EVAL_DATASET_NAME = "eval_vla_tasks"
DEFAULT_EVAL_EPISODE_TIME_S = 30
DEFAULT_EVAL_RESET_TIME_S = 10


def load_dotenv(path: Path, *, override: bool = False) -> None:
    """Load a simple `.env` file into `os.environ` (KEY=VALUE only)."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].lstrip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        # Handle quoted values
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            # Strip inline comments (only for unquoted values)
            # Match: value # comment  or  value#comment
            comment_match = re.match(r"^([^#]*?)(?:\s+#|\s*#\s).*$", value)
            if comment_match:
                value = comment_match.group(1).strip()

        if not override and key in os.environ:
            continue
        os.environ[key] = value


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def parse_video_index(value: str) -> str:
    """Parse a camera index from `/dev/videoN`, `videoN`, or `N`."""
    text = (value or "").strip()
    if not text:
        raise ValueError("Camera value is empty")

    if text.isdigit():
        return text

    match = re.search(r"(\d+)\s*$", text)
    if not match:
        raise ValueError(f"Could not parse camera index from '{value}'")
    return match.group(1)


def build_vla_camera_config(
    *,
    top: str,
    side: str,
    gripper: str,
    width: int,
    height: int,
    fps: int,
) -> str:
    """Build VLA-friendly camera config string (camera1/camera2/camera3)."""
    top_idx = parse_video_index(top)
    side_idx = parse_video_index(side)
    gripper_idx = parse_video_index(gripper)

    return (
        f"{{camera1: {{type: opencv, index_or_path: {top_idx}, width: {width}, height: {height}, fps: {fps}}}, "
        f"camera2: {{type: opencv, index_or_path: {side_idx}, width: {width}, height: {height}, fps: {fps}}}, "
        f"camera3: {{type: opencv, index_or_path: {gripper_idx}, width: {width}, height: {height}, fps: {fps}}}}}"
    )


def default_eval_dataset_root(*, suffix: str = "inference") -> Path:
    """Create a unique eval dataset root path (no resume)."""
    base = Path(os.environ.get("EVAL_DATASET_ROOT_BASE", str(Path.home() / "so101_datasets"))).expanduser()
    name = os.environ.get("EVAL_DATASET_NAME", DEFAULT_EVAL_DATASET_NAME)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / f"{name}_{suffix}_{ts}_{os.getpid()}"


def build_lerobot_record_command(
    *,
    policy_path: str,
    prompt: str,
    episodes: int,
    resume: bool = False,
    device: str = "cuda",
    follower_port: str | None = None,
    follower_id: str | None = None,
    dataset_root: Path | None = None,
    dataset_repo_id: str | None = None,
    dataset_fps: int | None = None,
    episode_time_s: int | None = None,
    reset_time_s: int | None = None,
    display_data: bool = True,
    play_sounds: bool | None = None,
) -> tuple[list[str], Path]:
    """Build a `lerobot-record` command.

    If `MISSION2_SUBPROCESS_BACKEND=direct-inference`, this instead builds a command that runs
    `simple_run.py` (direct in-process inference inside a subprocess, no dataset saving).
    """
    if episodes <= 0:
        raise ValueError(f"episodes must be positive (got {episodes})")

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env", override=False)

    port = follower_port or os.environ.get("FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT)
    robot_id = follower_id or os.environ.get("FOLLOWER_ID", DEFAULT_FOLLOWER_ID)

    top = os.environ.get("TOP_CAMERA", DEFAULT_TOP_CAMERA)
    side = os.environ.get("SIDE_CAMERA", DEFAULT_SIDE_CAMERA)
    gripper = os.environ.get("GRIPPER_CAMERA", DEFAULT_GRIPPER_CAMERA)
    width = _env_int("CAMERA_WIDTH", DEFAULT_CAMERA_WIDTH)
    height = _env_int("CAMERA_HEIGHT", DEFAULT_CAMERA_HEIGHT)
    fps = _env_int("CAMERA_FPS", DEFAULT_CAMERA_FPS)
    camera_config = build_vla_camera_config(
        top=top,
        side=side,
        gripper=gripper,
        width=width,
        height=height,
        fps=fps,
    )

    if dataset_root is None:
        dataset_root = default_eval_dataset_root()

    hf_user = os.environ.get("HF_USER", DEFAULT_HF_USER)
    default_repo_id = f"{hf_user}/{os.environ.get('EVAL_DATASET_NAME', DEFAULT_EVAL_DATASET_NAME)}"
    dataset_repo_id = dataset_repo_id or os.environ.get("EVAL_DATASET_REPO_ID", default_repo_id)

    push_to_hub = os.environ.get("EVAL_PUSH_TO_HUB", "false").strip().lower() == "true"
    episode_time = episode_time_s if episode_time_s is not None else _env_int("EVAL_EPISODE_TIME", DEFAULT_EVAL_EPISODE_TIME_S)
    reset_time = reset_time_s if reset_time_s is not None else _env_int("EVAL_RESET_TIME", DEFAULT_EVAL_RESET_TIME_S)

    # Backward compatible alias: MISSION2_SUBPROCESS_RUNNER
    runner = os.environ.get("MISSION2_SUBPROCESS_BACKEND", "").strip() or os.environ.get("MISSION2_SUBPROCESS_RUNNER", "lerobot-record").strip()
    runner = runner.lower()
    if runner in {
        "direct-inference",
        "direct_inference",
        "direct",
        "inprocess",
        "in-process",
        "in_process",
        # Backward compatible aliases
        "simple-run",
        "simple_run",
        "simple",
    }:
        cmd = [
            "uv",
            "run",
            "python",
            str(project_root / "simple_run.py"),
            policy_path,
            prompt,
            "--time",
            str(episode_time),
            "--fps",
            str(fps),
        ]
        if display_data:
            cmd.append("--display-data")
        return cmd, dataset_root

    cmd = [
        "uv",
        "run",
        "lerobot-record",
        f"--robot.type={DEFAULT_FOLLOWER_TYPE}",
        f"--robot.port={port}",
        f"--robot.id={robot_id}",
        f"--robot.cameras={camera_config}",
        f"--display_data={'true' if display_data else 'false'}",
        f"--dataset.repo_id={dataset_repo_id}",
    ]
    if dataset_fps is not None:
        cmd.append(f"--dataset.fps={dataset_fps}")
    cmd += [
        f"--dataset.num_episodes={episodes}",
        f"--dataset.episode_time_s={episode_time}",
        f"--dataset.reset_time_s={reset_time}",
        f"--dataset.single_task={prompt}",
        f"--dataset.push_to_hub={'true' if push_to_hub else 'false'}",
        f"--dataset.root={dataset_root!s}",
        f"--policy.path={policy_path}",
        f"--policy.device={device}",
        f"--resume={'true' if resume else 'false'}",
    ]
    if play_sounds is not None:
        cmd.append(f"--play_sounds={'true' if play_sounds else 'false'}")
    return cmd, dataset_root


def run_lerobot_record(
    *,
    policy_path: str,
    prompt: str,
    episodes: int,
    resume: bool = False,
    device: str = "cuda",
    follower_port: str | None = None,
    follower_id: str | None = None,
    dataset_root: Path | None = None,
    dataset_repo_id: str | None = None,
    dataset_fps: int | None = None,
    episode_time_s: int | None = None,
    reset_time_s: int | None = None,
    display_data: bool = True,
    play_sounds: bool | None = None,
) -> tuple[bool, Path, str]:
    """Run `lerobot-record` and return (ok, dataset_root, error_message)."""
    cmd, resolved_root = build_lerobot_record_command(
        policy_path=policy_path,
        prompt=prompt,
        episodes=episodes,
        resume=resume,
        device=device,
        follower_port=follower_port,
        follower_id=follower_id,
        dataset_root=dataset_root,
        dataset_repo_id=dataset_repo_id,
        dataset_fps=dataset_fps,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        display_data=display_data,
        play_sounds=play_sounds,
    )

    project_root = Path(__file__).resolve().parents[2]

    def run_with_output_tail() -> tuple[int, str]:
        echo = os.environ.get("MISSION2_SUBPROCESS_ECHO", "").strip().lower() in {"1", "true", "yes", "y"}
        tail: deque[str] = deque(maxlen=200)

        if echo:
            print("MISSION2 subprocess:", " ".join(shlex.quote(part) for part in cmd), flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                text_line = line.rstrip("\n")
                tail.append(text_line)
                if echo:
                    print(text_line, flush=True)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass

        rc = proc.wait()
        return rc, "\n".join(list(tail)[-25:])

    try:
        rc, tail = run_with_output_tail()
        if rc == 0:
            return True, resolved_root, ""
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        if tail.strip():
            return False, resolved_root, f"subprocess failed (exit {rc})\ncommand: {cmd_str}\n{tail}"
        return False, resolved_root, f"subprocess failed (exit {rc})\ncommand: {cmd_str}"
    except FileNotFoundError:
        return False, resolved_root, "command not found. Ensure 'uv' is installed and dependencies are synced."
