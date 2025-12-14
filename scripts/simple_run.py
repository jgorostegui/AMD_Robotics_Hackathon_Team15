#!/usr/bin/env python3

import argparse
import json
import os
import time
from pathlib import Path

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import build_dataset_frame, combine_feature_dicts
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import make_default_processors
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from mission2.robot.lerobot_record import DEFAULT_CAMERA_FPS, DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA_WIDTH
from mission2.robot.lerobot_record import DEFAULT_GRIPPER_CAMERA, DEFAULT_SIDE_CAMERA, DEFAULT_TOP_CAMERA
from mission2.robot.lerobot_record import DEFAULT_FOLLOWER_ID, DEFAULT_FOLLOWER_PORT
from mission2.robot.lerobot_record import DEFAULT_EVAL_EPISODE_TIME_S, load_dotenv, parse_video_index


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_tasks_json(tasks_json: str | None) -> Path:
    if tasks_json:
        path = Path(tasks_json)
        return path if path.is_absolute() else _project_root() / path

    env_value = os.environ.get("TASKS_JSON", "").strip()
    if env_value:
        path = Path(env_value)
        return path if path.is_absolute() else _project_root() / path

    return _project_root() / "mission2/tasks_smolvla.json"


def load_tasks(tasks_json: Path) -> dict[str, str]:
    with open(tasks_json, encoding="utf-8") as file:
        data = json.load(file)

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
        for index, item in enumerate(data):
            if isinstance(item, dict):
                task_id = item.get("id") or item.get("name") or item.get("task_id") or f"task{index+1}"
                add(task_id, item.get("prompt") or item.get("instruction") or item.get("task"))
            elif isinstance(item, str):
                add(f"task{index+1}", item)
            else:
                raise ValueError(f"Unsupported task entry at index {index}: {type(item).__name__}")
    else:
        raise ValueError(f"Unsupported tasks schema: {type(data).__name__}")

    if not tasks:
        raise ValueError(f"No tasks found in {tasks_json}")

    return dict(tasks)


def _resolve_prompt(task_or_prompt: str, *, tasks_json: Path) -> tuple[str, str]:
    task_map = load_tasks(tasks_json)
    key = (task_or_prompt or "").strip()
    if key in task_map:
        return key, task_map[key]
    return "custom", task_or_prompt


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-shot SmolVLA inference (no subprocess, no dataset saving)."
    )
    parser.add_argument("model_path", type=str, help="Local path or HF repo id")
    parser.add_argument("task", type=str, help="Task id (e.g. task4) or a raw prompt string")
    parser.add_argument("--tasks-json", type=str, default=None, help="Tasks JSON path (optional)")
    parser.add_argument("--time", type=float, default=None, help="Run time seconds (default: EVAL_EPISODE_TIME/30)")
    parser.add_argument("--fps", type=int, default=None, help="Control FPS (default: CAMERA_FPS/30)")
    parser.add_argument(
        "--display-data",
        action="store_true",
        help="Show rerun visualization (if supported by your environment).",
    )
    args = parser.parse_args()

    load_dotenv(_project_root() / ".env", override=False)

    tasks_json = _resolve_tasks_json(args.tasks_json)
    task_id, prompt = _resolve_prompt(args.task, tasks_json=tasks_json)

    follower_port = os.environ.get("FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT)
    follower_id = os.environ.get("FOLLOWER_ID", DEFAULT_FOLLOWER_ID)

    top = os.environ.get("TOP_CAMERA", DEFAULT_TOP_CAMERA)
    side = os.environ.get("SIDE_CAMERA", DEFAULT_SIDE_CAMERA)
    gripper = os.environ.get("GRIPPER_CAMERA", DEFAULT_GRIPPER_CAMERA)

    camera_width = int(os.environ.get("CAMERA_WIDTH", str(DEFAULT_CAMERA_WIDTH)))
    camera_height = int(os.environ.get("CAMERA_HEIGHT", str(DEFAULT_CAMERA_HEIGHT)))
    camera_fps = int(os.environ.get("CAMERA_FPS", str(DEFAULT_CAMERA_FPS)))

    fps = int(args.fps) if args.fps is not None else camera_fps
    run_time_s = float(args.time) if args.time is not None else float(os.environ.get("EVAL_EPISODE_TIME", DEFAULT_EVAL_EPISODE_TIME_S))

    metrics = {
        "total_infer_time_s": 0.0,
        "num_steps": 0,
    }

    cameras = {
        "camera1": OpenCVCameraConfig(
            index_or_path=int(parse_video_index(top)),
            width=camera_width,
            height=camera_height,
            fps=fps,
        ),
        "camera2": OpenCVCameraConfig(
            index_or_path=int(parse_video_index(side)),
            width=camera_width,
            height=camera_height,
            fps=fps,
        ),
        "camera3": OpenCVCameraConfig(
            index_or_path=int(parse_video_index(gripper)),
            width=camera_width,
            height=camera_height,
            fps=fps,
        ),
    }

    robot = SO101Follower(
        SO101FollowerConfig(
            port=follower_port,
            id=follower_id,
            cameras=cameras,
        )
    )

    cfg = PreTrainedConfig.from_pretrained(args.model_path)
    cfg.device = "cuda"
    model_path = Path(args.model_path)
    if model_path.exists():
        cfg.pretrained_path = model_path
    cfg.validate_features()

    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.model_path, config=cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    if args.display_data:
        init_rerun(session_name="inference")

    print(f"Model: {args.model_path}")
    print(f"Task: {task_id}")
    print(f"Prompt: {prompt}")
    print(f"Device: {cfg.device}")
    print(f"Robot port: {follower_port}")
    print(f"Cameras: top={top} side={side} gripper={gripper}")
    print(f"FPS: {fps}")
    print(f"Time: {run_time_s}")

    robot.connect()
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    device = get_safe_torch_device(cfg.device)

    def run_prompt(task_prompt: str, duration_s: float) -> None:
        
        if duration_s <= 0:
            return
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < duration_s:
            loop_start = time.perf_counter()

            obs_raw = robot.get_observation()
            obs_processed = robot_observation_processor(obs_raw)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)

            t_inf_start = time.perf_counter()
            action_tensor = predict_action(
                observation=observation_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=bool(getattr(cfg, "use_amp", False)),
                task=task_prompt,
                robot_type=robot.robot_type,
            )
            infer_dt = time.perf_counter() - t_inf_start

            metrics["total_infer_time_s"] += infer_dt
            metrics["num_steps"] += 1

            action_values = make_robot_action(action_tensor, dataset_features)
            action_to_send = robot_action_processor((action_values, obs_raw))
            robot.send_action(action_to_send)

            if args.display_data:
                log_rerun_data(observation=obs_processed, action=action_values)

            dt_s = time.perf_counter() - loop_start
            sleep_s = max(0.0, (1.0 / float(fps)) - dt_s)
            if sleep_s > 0:
                time.sleep(sleep_s)

    try:
        loop_start = time.perf_counter()
        run_prompt(prompt, run_time_s)
        loop_end = time.perf_counter()
        loop_dt = loop_end - loop_start
        print("\n[LOOP]")
        print(f"  Tiempo total run_prompt(): {loop_dt:.3f} s")
        print(f"  Tiempo objetivo (run_time_s): {run_time_s:.3f} s")
    finally:
        robot.disconnect()

    if metrics["num_steps"] > 0:
        total_infer_time_s = metrics["total_infer_time_s"]
        num_steps = metrics["num_steps"]
        avg_infer_ms = (total_infer_time_s / num_steps) * 1000.0
        fps_infer = num_steps / total_infer_time_s

        print(f"\n[METRICS]")
        print(f"  Steps de control: {num_steps}")
        print(f"  Tiempo total de inferencia: {total_infer_time_s:.3f} s")
        print(f"  Tiempo medio por inferencia: {avg_infer_ms:.2f} ms")
        print(f"  FPS efectivos de inferencia: {fps_infer:.2f} fps")
    else:
        print("\n[METRICS] No se ejecutó ningún step de inferencia.")

    return 0


# if __name__ == "__main__":
#     raise SystemExit(main())

if __name__ == "__main__":
    t0 = time.perf_counter()
    exit_code = main()
    total_s = time.perf_counter() - t0

    print("\n[GLOBAL]")
    print(f"  Tiempo total del programa (main): {total_s:.3f} s")

    raise SystemExit(exit_code)
