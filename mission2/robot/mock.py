"""Mock robot for testing without hardware.

Uses the same prompt format as real SmolVLA from tasks_smolvla.json.
"""

import time

from ..core.bus import get_event_bus
from ..core.events import Event, EventType
from ..core.types import ActionStatus, Move, RobotAction, RobotState
from .interface import RobotInterface
from .positions import HOME_PROMPT, get_move_prompt


class MockRobot(RobotInterface):
    """Mock robot that shows SmolVLA prompts without executing.
    
    Displays the exact prompt that would be sent to the real robot.
    """

    def __init__(self, move_delay: float = 0.3):
        """Initialize mock robot.
        
        Args:
            move_delay: Simulated delay (seconds)
        """
        self.move_delay = move_delay
        self._connected = False
        self._last_action: RobotAction | None = None
        self._last_instruction = ""
        self.bus = get_event_bus()

    def connect(self) -> bool:
        self._connected = True
        self._last_instruction = "Robot connected (mock mode)"
        self.bus.publish(Event(
            type=EventType.ROBOT_CONNECTED,
            data={"mode": "mock"},
            source="mock_robot"
        ))
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._last_instruction = "Robot disconnected"
        self.bus.publish(Event(
            type=EventType.ROBOT_DISCONNECTED,
            source="mock_robot"
        ))

    def get_state(self) -> RobotState:
        return RobotState(
            connected=self._connected,
            is_moving=False,
            at_home=True,
            last_action=self._last_action,
            last_instruction=self._last_instruction,
        )

    def go_home(self) -> bool:
        if not self._connected:
            return False
        self._last_instruction = HOME_PROMPT
        time.sleep(self.move_delay)
        self.bus.publish(Event(
            type=EventType.ROBOT_AT_HOME,
            source="mock_robot"
        ))
        return True

    def execute_move(self, move: Move) -> RobotAction:
        """Show the SmolVLA prompt that would be sent."""
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
            source="mock_robot"
        ))

        # Simulate execution time
        time.sleep(self.move_delay)

        action.status = ActionStatus.COMPLETED
        self._last_action = action

        self.bus.publish(Event(
            type=EventType.ROBOT_MOVE_COMPLETE,
            data={"column": move.column},
            source="mock_robot"
        ))

        return action

    def is_connected(self) -> bool:
        return self._connected

    def get_last_instruction(self) -> str:
        return self._last_instruction
