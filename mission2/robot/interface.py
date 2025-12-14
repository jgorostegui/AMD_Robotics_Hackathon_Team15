"""Abstract interface for robot control."""

from abc import ABC, abstractmethod

from ..core.types import Move, RobotAction, RobotState


class RobotInterface(ABC):
    """Abstract interface for robot control.
    
    Implementations can be mock (for testing) or real (SmolVLA).
    """

    @abstractmethod
    def connect(self) -> bool:
        """Connect to robot.
        
        Returns:
            True if connection succeeded
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from robot."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state."""
        pass

    @abstractmethod
    def go_home(self) -> bool:
        """Move robot to home position.
        
        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def execute_move(self, move: Move) -> RobotAction:
        """Execute a game move (pick piece, drop in column).
        
        Args:
            move: The game move to execute
            
        Returns:
            RobotAction with status and instruction
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        pass

    @abstractmethod
    def get_last_instruction(self) -> str:
        """Get the last SmolVLA instruction (for display)."""
        pass
