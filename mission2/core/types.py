"""
Shared data types for Connect4 TARS system.

These types are the contracts between modules.
All modules communicate using these structures.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


# ─────────────────────────────────────────────────────────────
# PLAYER & GAME PHASE
# ─────────────────────────────────────────────────────────────


class Player(Enum):
    """Player identifier."""

    ORANGE = "orange"  # Robot
    YELLOW = "yellow"  # Human
    EMPTY = "empty"

    def __str__(self) -> str:
        return self.value

    @property
    def symbol(self) -> str:
        """Get emoji symbol for display."""
        return {"orange": "🟠", "yellow": "🟡", "empty": "⚪"}[self.value]


class GamePhase(Enum):
    """Current phase of the game."""

    WAITING_FOR_START = auto()  # Game not started
    ROBOT_TURN = auto()  # Robot's turn to play
    HUMAN_TURN = auto()  # Human's turn to play
    ROBOT_MOVING = auto()  # Robot is executing a move
    GAME_OVER = auto()  # Game finished
    ERROR = auto()  # Error state


# ─────────────────────────────────────────────────────────────
# BOARD REPRESENTATION
# ─────────────────────────────────────────────────────────────


@dataclass
class Position:
    """Grid position (0-indexed)."""

    row: int  # 0 = top, 4 = bottom
    col: int  # 0 = left, 4 = right

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return False
        return self.row == other.row and self.col == other.col


@dataclass
class BoardState:
    """
    Board state from vision or game engine.

    The grid is a 2D list where:
    - grid[0] is the top row
    - grid[4] is the bottom row
    - grid[row][col] contains a Player value
    """

    grid: list[list[Player]] = field(
        default_factory=lambda: [[Player.EMPTY] * 5 for _ in range(5)]
    )
    board_detected: bool = True
    confidence: float = 1.0
    timestamp: float = 0.0
    raw_image: np.ndarray | None = None
    annotated_image: np.ndarray | None = None
    corners: list[tuple[int, int]] = field(default_factory=list)

    @property
    def as_matrix(self) -> np.ndarray:
        """Convert to numpy matrix for AI processing.

        Returns:
            numpy array where ORANGE=1, YELLOW=-1, EMPTY=0
        """
        mapping = {Player.ORANGE: 1, Player.YELLOW: -1, Player.EMPTY: 0}
        return np.array([[mapping[cell] for cell in row] for row in self.grid])

    def copy(self) -> "BoardState":
        """Create a deep copy of the board state."""
        return BoardState(
            grid=[[cell for cell in row] for row in self.grid],
            board_detected=self.board_detected,
            confidence=self.confidence,
            timestamp=self.timestamp,
            corners=self.corners.copy(),
        )


# ─────────────────────────────────────────────────────────────
# MOVE & GAME STATE
# ─────────────────────────────────────────────────────────────


@dataclass
class Move:
    """A game move."""

    column: int  # 0-4 for 5-column board
    player: Player
    position: Position | None = None  # Filled after placement

    def __str__(self) -> str:
        return f"{self.player.symbol} → Column {self.column}"


@dataclass
class GameState:
    """Complete game state snapshot."""

    board: BoardState
    phase: GamePhase
    current_player: Player
    move_history: list[Move] = field(default_factory=list)
    winner: Player | None = None
    winning_positions: list[Position] = field(default_factory=list)
    legal_moves: list[int] = field(default_factory=list)
    turn_number: int = 0
    error_message: str | None = None

    def copy(self) -> "GameState":
        """Create a copy of the game state."""
        return GameState(
            board=self.board.copy(),
            phase=self.phase,
            current_player=self.current_player,
            move_history=self.move_history.copy(),
            winner=self.winner,
            winning_positions=self.winning_positions.copy(),
            legal_moves=self.legal_moves.copy(),
            turn_number=self.turn_number,
            error_message=self.error_message,
        )


# ─────────────────────────────────────────────────────────────
# ROBOT CONTROL
# ─────────────────────────────────────────────────────────────


class ActionStatus(Enum):
    """Status of a robot action."""

    PENDING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class RobotAction:
    """Command for robot arm."""

    move: Move
    status: ActionStatus = ActionStatus.PENDING
    error: str | None = None
    instruction: str = ""  # SmolVLA instruction text

    @property
    def is_complete(self) -> bool:
        return self.status in (ActionStatus.COMPLETED, ActionStatus.FAILED)


@dataclass
class RobotState:
    """Current robot status."""

    connected: bool = False
    is_moving: bool = False
    at_home: bool = True
    last_action: RobotAction | None = None
    last_instruction: str = ""  # Last SmolVLA instruction
    joint_positions: list[float] | None = None
