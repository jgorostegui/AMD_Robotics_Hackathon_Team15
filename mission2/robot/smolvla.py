"""SmolVLA robot implementation using `lerobot-record` via subprocess.

This mirrors the proven behavior of `scripts/run_inference_vla.sh`:
- one `lerobot-record` invocation per move (task prompt)
- robot connects/disconnects inside that process
"""

import os
from pathlib import Path

from ..core.bus import get_event_bus
from ..core.events import Event, EventType
from ..core.types import ActionStatus, Move, RobotAction, RobotState
from .interface import RobotInterface
from .lerobot_record import (
    DEFAULT_EVAL_DATASET_NAME,
    DEFAULT_EVAL_EPISODE_TIME_S,
    DEFAULT_EVAL_RESET_TIME_S,
    run_lerobot_record,
)
from .positions import HOME_PROMPT, get_move_prompt


class SmolVLARobot(RobotInterface):
    """Real robot control via SmolVLA policy using `lerobot-record` subprocess."""

    def __init__(
        self,
        policy_path: str,
        follower_port: str | None = None,
        follower_id: str | None = None,
        episode_time: int | None = None,
        reset_time: int | None = None,
        device: str = "cuda",
        display_data: bool = True,
    ):
        """Initialize SmolVLA robot.

        Args:
            policy_path: Path to SmolVLA policy (HF repo or local path)
            follower_port: Robot serial port (default: from env or /dev/ttyACM1)
            follower_id: Robot ID (default: from env or follower_arm)
            episode_time: Max episode duration in seconds (default: from env `EVAL_EPISODE_TIME` or 30)
            reset_time: Reset time between episodes (default: from env `EVAL_RESET_TIME` or 10)
            device: Compute device (cuda/cpu)
            display_data: Show camera windows / visualization (subprocess-dependent)
        """
        self.policy_path = policy_path
        self.bus = get_event_bus()

        self.follower_port = follower_port or os.environ.get("FOLLOWER_PORT", "/dev/ttyACM1")
        self.follower_id = follower_id or os.environ.get("FOLLOWER_ID", "follower_arm")
        self.episode_time = int(
            episode_time
            if episode_time is not None
            else os.environ.get("EVAL_EPISODE_TIME", str(DEFAULT_EVAL_EPISODE_TIME_S))
        )
        self.reset_time = int(
            reset_time
            if reset_time is not None
            else os.environ.get("EVAL_RESET_TIME", str(DEFAULT_EVAL_RESET_TIME_S))
        )
        self.device = device
        self.display_data = bool(display_data)

        self._connected = False
        self._is_moving = False
        self._last_action: RobotAction | None = None
        self._last_instruction = ""
        self._eval_dataset_root: Path | None = None
        self._resume_dataset = False

    def connect(self) -> bool:
        """Prepare a session dataset root. Robot connects in the subprocess."""
        if self._connected and self._eval_dataset_root is not None:
            return True

        if not os.path.exists(self.follower_port):
            self._last_instruction = f"ERROR: Robot port not found: {self.follower_port}"
            return False

        self._eval_dataset_root = _default_session_eval_dataset_root()
        self._resume_dataset = (self._eval_dataset_root / "meta/info.json").is_file()

        self._connected = True
        self._last_instruction = f"Ready (policy: {self.policy_path})"
        self.bus.publish(Event(
            type=EventType.ROBOT_CONNECTED,
            data={"port": self.follower_port, "policy": self.policy_path},
            source="smolvla_robot"
        ))
        return True

    def disconnect(self) -> None:
        """Mark robot as disconnected (no persistent connection)."""
        self._eval_dataset_root = None
        self._resume_dataset = False
        self._connected = False
        self._last_instruction = "Robot disconnected"
        self.bus.publish(Event(
            type=EventType.ROBOT_DISCONNECTED,
            source="smolvla_robot"
        ))

    def get_state(self) -> RobotState:
        """Get current robot state."""
        return RobotState(
            connected=self._connected,
            is_moving=self._is_moving,
            at_home=not self._is_moving,
            last_action=self._last_action,
            last_instruction=self._last_instruction,
        )

    def go_home(self) -> bool:
        """Move robot to home position.

        Uses the home prompt to execute a single episode.
        """
        if not self._connected:
            return False

        self._last_instruction = HOME_PROMPT
        return self._execute_prompt(HOME_PROMPT)

    def execute_move(self, move: Move) -> RobotAction:
        """Execute a game move using SmolVLA policy.

        Runs one in-process episode with the column-specific prompt.
        """
        if not self._connected:
            action = RobotAction(
                move=move,
                status=ActionStatus.FAILED,
                error="Robot not connected",
                instruction="ERROR: Robot not connected"
            )
            self._last_action = action
            return action

        prompt = get_move_prompt(move.column)
        self._last_instruction = prompt

        action = RobotAction(
            move=move,
            status=ActionStatus.EXECUTING,
            instruction=prompt
        )

        self.bus.publish(Event(
            type=EventType.ROBOT_MOVING,
            data={"column": move.column, "prompt": prompt},
            source="smolvla_robot"
        ))

        self._is_moving = True
        success = self._execute_prompt(prompt)
        self._is_moving = False

        if success:
            action.status = ActionStatus.COMPLETED
            self.bus.publish(Event(
                type=EventType.ROBOT_MOVE_COMPLETE,
                data={"column": move.column},
                source="smolvla_robot"
            ))
        else:
            action.status = ActionStatus.FAILED
            action.error = self._last_instruction or "SmolVLA execution failed"

        self._last_action = action
        return action

    def _execute_prompt(self, prompt: str) -> bool:
        """Execute one prompt by spawning `lerobot-record`."""
        if self._eval_dataset_root is None:
            self._last_instruction = "ERROR: Robot not connected"
            return False

        self._last_instruction = f"RUNNING: {prompt}"
        ok, _, error = run_lerobot_record(
            policy_path=self.policy_path,
            prompt=prompt,
            episodes=1,
            resume=bool(self._resume_dataset),
            device=self.device,
            follower_port=self.follower_port,
            follower_id=self.follower_id,
            dataset_root=self._eval_dataset_root,
            episode_time_s=int(self.episode_time),
            reset_time_s=int(self.reset_time),
            display_data=self.display_data,
        )
        if ok:
            self._resume_dataset = True
            self._last_instruction = prompt
            return True
        self._last_instruction = f"ERROR: {error}"
        return False

    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected

    def get_last_instruction(self) -> str:
        """Get the last SmolVLA instruction."""
        return self._last_instruction


def _default_session_eval_dataset_root() -> Path:
    base = Path(os.environ.get("EVAL_DATASET_ROOT_BASE", str(Path.home() / "so101_datasets"))).expanduser()
    name = os.environ.get("EVAL_DATASET_NAME", DEFAULT_EVAL_DATASET_NAME)
    root = base / name

    if not root.exists():
        return root

    i = 1
    while (base / f"{name}_v{i}").exists():
        i += 1
    return base / f"{name}_v{i}"
