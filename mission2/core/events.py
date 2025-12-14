"""
Event definitions for the Connect4 TARS system.

Events enable loose coupling between modules.
Modules publish events without knowing who consumes them.
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    """Types of events in the system."""

    # Vision events
    FRAME_CAPTURED = auto()
    BOARD_DETECTED = auto()
    DETECTION_FAILED = auto()
    VISION_STATE_UPDATED = auto()  # Vision board accepted/applied to game state
    VISION_STATE_DESYNC = auto()  # Vision board disagrees with game state

    # Game events
    GAME_STARTED = auto()
    TURN_CHANGED = auto()
    MOVE_MADE = auto()
    INVALID_MOVE = auto()
    GAME_WON = auto()
    GAME_DRAW = auto()
    GAME_RESET = auto()

    # Robot events
    ROBOT_CONNECTED = auto()
    ROBOT_DISCONNECTED = auto()
    ROBOT_MOVING = auto()
    ROBOT_MOVE_COMPLETE = auto()
    ROBOT_ERROR = auto()
    ROBOT_AT_HOME = auto()

    # TARS events
    TARS_RESPONSE = auto()
    VOICE_COMMAND = auto()

    # System events
    SYSTEM_ERROR = auto()
    SYSTEM_SHUTDOWN = auto()


@dataclass
class Event:
    """
    Base event structure.

    Attributes:
        type: The type of event
        data: Event-specific payload
        timestamp: When the event was created
        source: Which module created the event
    """

    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"

    def __str__(self) -> str:
        return f"[{self.source}] {self.type.name}: {self.data}"
