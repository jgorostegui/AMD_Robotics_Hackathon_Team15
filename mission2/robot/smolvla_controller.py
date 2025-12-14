"""Persistent SmolVLA controller (no per-move subprocess).

- Loads the VLA policy once
- Keeps the robot connection open (prevents torque drop)
- Runs multiple task prompts sequentially

Note: `lerobot`/`torch` are imported inside `connect()`/`run_prompt()` to avoid
paying their import/init cost unless robot mode is actually used.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from .lerobot_record import parse_video_index


@dataclass(frozen=True)
class SmolVLAControllerConfig:
    policy_path: str
    device: str = "cuda"
    follower_port: str = "/dev/ttyACM1"
    follower_id: str = "follower_arm"
    episode_time_s: float = 30.0
    fps: float = 10.0


class SmolVLAControllerError(RuntimeError):
    pass


class _RobotWrapper:
    def __init__(self, robot: Any):
        self.robot = robot
        self._lock = threading.Lock()

    def get_observation(self):
        with self._lock:
            return self.robot.get_observation()

    def send_action(self, action):
        with self._lock:
            self.robot.send_action(action)

    def observation_features(self) -> dict[str, type | tuple]:
        with self._lock:
            return dict(self.robot.observation_features)

    def action_features(self) -> dict[str, type]:
        with self._lock:
            return dict(self.robot.action_features)


class SmolVLAController:
    """Loads policy+robot once and runs prompts on demand."""

    def __init__(self, cfg: SmolVLAControllerConfig):
        self._cfg = cfg
        self._shutdown_event = threading.Event()
        self._policy = None
        self._policy_cfg = None
        self._robot_wrapper: _RobotWrapper | None = None
        self._robot_observation_processor = None
        self._robot_action_processor = None
        self._preprocessor = None
        self._postprocessor = None
        self._rtc_cfg = None

    def connect(self) -> None:
        """Load policy and connect to robot (heavy)."""
        if self._robot_wrapper is not None:
            return

        try:
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.processor.factory import (
                make_default_robot_action_processor,
                make_default_robot_observation_processor,
            )
            from lerobot.robots.so101_follower import SO101FollowerConfig
            from lerobot.robots.utils import make_robot_from_config
            from lerobot.utils.import_utils import register_third_party_devices
        except Exception as e:  # pragma: no cover
            raise SmolVLAControllerError(
                f"Missing runtime dependencies for SmolVLA control: {e}"
            ) from e
        robot = None
        try:
            register_third_party_devices()

            policy_cfg = PreTrainedConfig.from_pretrained(self._cfg.policy_path)
            policy_cfg.pretrained_path = self._cfg.policy_path
            policy_cfg.device = self._cfg.device
            policy_class = get_policy_class(policy_cfg.type)
            policy = policy_class.from_pretrained(self._cfg.policy_path, config=policy_cfg)

            rtc_cfg = RTCConfig(enabled=True)
            policy.config.rtc_config = rtc_cfg
            policy.init_rtc_processor()
            policy = policy.to(self._cfg.device)
            policy.eval()

            self._policy = policy
            self._policy_cfg = policy_cfg
            self._rtc_cfg = rtc_cfg

            cameras = _build_default_cameras(OpenCVCameraConfig)
            robot_cfg = SO101FollowerConfig(
                port=self._cfg.follower_port,
                id=self._cfg.follower_id,
                cameras=cameras,
            )
            robot = make_robot_from_config(robot_cfg)
            robot.connect()
            self._robot_wrapper = _RobotWrapper(robot)

            self._robot_observation_processor = make_default_robot_observation_processor()
            self._robot_action_processor = make_default_robot_action_processor()

            self._preprocessor, self._postprocessor = make_pre_post_processors(
                policy_cfg=policy_cfg,
                pretrained_path=policy_cfg.pretrained_path,
                dataset_stats=None,
                preprocessor_overrides={"device_processor": {"device": policy_cfg.device}},
            )
        except SmolVLAControllerError:
            raise
        except Exception as e:
            if robot is not None:
                try:
                    robot.disconnect()
                except Exception:
                    pass
            self._robot_wrapper = None
            self._policy = None
            self._policy_cfg = None
            self._robot_observation_processor = None
            self._robot_action_processor = None
            self._preprocessor = None
            self._postprocessor = None
            self._rtc_cfg = None
            raise SmolVLAControllerError(f"Failed to initialize controller: {e}") from e

    def disconnect(self) -> None:
        """Disconnect robot and release references."""
        self._shutdown_event.set()
        if self._robot_wrapper is not None:
            try:
                self._robot_wrapper.robot.disconnect()
            except Exception:
                pass
        self._robot_wrapper = None
        self._policy = None
        self._policy_cfg = None
        self._robot_observation_processor = None
        self._robot_action_processor = None
        self._preprocessor = None
        self._postprocessor = None
        self._rtc_cfg = None

    def run_prompt(self, prompt: str, *, duration_s: float | None = None) -> None:
        """Run one prompt/episode while keeping robot connected."""
        if self._robot_wrapper is None or self._policy is None:
            raise SmolVLAControllerError("Controller not connected")

        duration = float(duration_s if duration_s is not None else self._cfg.episode_time_s)
        fps = float(self._cfg.fps)
        if fps <= 0:
            raise SmolVLAControllerError(f"Invalid fps: {fps}")

        self._shutdown_event.clear()

        # Lazy imports for runtime-heavy deps.
        import torch  # type: ignore
        from lerobot.datasets.utils import (  # type: ignore
            build_dataset_frame,
            hw_to_dataset_features,
        )
        from lerobot.policies.rtc.action_queue import ActionQueue  # type: ignore
        from lerobot.policies.rtc.latency_tracker import LatencyTracker  # type: ignore

        robot_wrapper = self._robot_wrapper
        policy = self._policy
        policy_cfg = self._policy_cfg
        robot_observation_processor = self._robot_observation_processor
        robot_action_processor = self._robot_action_processor
        preprocessor = self._preprocessor
        postprocessor = self._postprocessor
        rtc_cfg = self._rtc_cfg

        if any(x is None for x in (policy_cfg, robot_observation_processor, robot_action_processor, preprocessor, postprocessor, rtc_cfg)):
            raise SmolVLAControllerError("Controller is not fully initialized")

        action_queue = ActionQueue(rtc_cfg)

        exceptions: list[BaseException] = []

        def get_actions_thread() -> None:
            try:
                latency_tracker = LatencyTracker()
                time_per_chunk = 1.0 / fps
                dataset_features = hw_to_dataset_features(robot_wrapper.observation_features(), "observation")
                policy_device = policy.config.device
                get_actions_threshold = 0 if not getattr(rtc_cfg, "enabled", True) else 30

                while not self._shutdown_event.is_set():
                    if action_queue.qsize() > get_actions_threshold:
                        time.sleep(0.01)
                        continue

                    current_time = time.perf_counter()
                    action_index_before_inference = action_queue.get_action_index()
                    prev_actions = action_queue.get_left_over()

                    inference_latency = latency_tracker.max()
                    inference_delay = math.ceil(inference_latency / time_per_chunk)

                    obs = robot_wrapper.get_observation()
                    obs_processed = robot_observation_processor(obs)
                    obs_features = build_dataset_frame(dataset_features, obs_processed, prefix="observation")

                    for name, tensor in list(obs_features.items()):
                        t = torch.from_numpy(tensor)
                        if "image" in name:
                            t = t.type(torch.float32) / 255
                            t = t.permute(2, 0, 1).contiguous()
                        obs_features[name] = t.unsqueeze(0).to(policy_device)

                    obs_features["task"] = [prompt]
                    obs_features["robot_type"] = getattr(robot_wrapper.robot, "name", "")

                    preprocessed_obs = preprocessor(obs_features)
                    actions = policy.predict_action_chunk(
                        preprocessed_obs,
                        inference_delay=inference_delay,
                        prev_chunk_left_over=prev_actions,
                    )

                    original_actions = actions.squeeze(0).clone()
                    postprocessed_actions = postprocessor(actions).squeeze(0)

                    new_latency = time.perf_counter() - current_time
                    new_delay = math.ceil(new_latency / time_per_chunk)
                    latency_tracker.add(new_latency)

                    action_queue.merge(
                        original_actions,
                        postprocessed_actions,
                        new_delay,
                        action_index_before_inference,
                    )
            except BaseException as e:
                exceptions.append(e)
                self._shutdown_event.set()

        def actor_thread() -> None:
            try:
                action_interval = 1.0 / fps
                while not self._shutdown_event.is_set():
                    start_time = time.perf_counter()
                    action = action_queue.get()
                    if action is not None:
                        action = action.cpu()
                        action_dict = {k: action[i].item() for i, k in enumerate(robot_wrapper.action_features())}
                        action_processed = robot_action_processor((action_dict, None))
                        robot_wrapper.send_action(action_processed)
                    dt_s = time.perf_counter() - start_time
                    time.sleep(max(0.0, (action_interval - dt_s) - 0.001))
            except BaseException as e:
                exceptions.append(e)
                self._shutdown_event.set()

        t_get = threading.Thread(target=get_actions_thread, daemon=True, name="SmolVLAGetActions")
        t_act = threading.Thread(target=actor_thread, daemon=True, name="SmolVLAActor")
        t_get.start()
        t_act.start()

        start = time.time()
        while not self._shutdown_event.is_set() and (time.time() - start) < duration:
            time.sleep(0.25)

        self._shutdown_event.set()
        t_get.join(timeout=5.0)
        t_act.join(timeout=5.0)

        if exceptions:
            raise SmolVLAControllerError(str(exceptions[0]))


def _build_default_cameras(OpenCVCameraConfig):
    top = os.environ.get("TOP_CAMERA", "/dev/video4")
    side = os.environ.get("SIDE_CAMERA", "/dev/video2")
    gripper = os.environ.get("GRIPPER_CAMERA", "/dev/video6")

    width = _env_int("CAMERA_WIDTH", 640)
    height = _env_int("CAMERA_HEIGHT", 480)
    fps = _env_int("CAMERA_FPS", 30)

    return {
        "camera1": OpenCVCameraConfig(index_or_path=int(parse_video_index(top)), width=width, height=height, fps=fps),
        "camera2": OpenCVCameraConfig(index_or_path=int(parse_video_index(side)), width=width, height=height, fps=fps),
        "camera3": OpenCVCameraConfig(index_or_path=int(parse_video_index(gripper)), width=width, height=height, fps=fps),
    }


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default
