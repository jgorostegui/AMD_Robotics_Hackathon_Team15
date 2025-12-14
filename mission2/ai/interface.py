"""Abstract interface for AI players."""

from abc import ABC, abstractmethod

from ..core.types import GameState


class AIInterface(ABC):
    """Abstract interface for AI players.
    
    Implementations compute the best move given a game state.
    """

    @abstractmethod
    def get_move(self, state: GameState) -> int:
        """Compute the best move.
        
        Args:
            state: Current game state
            
        Returns:
            Column number (0-4) for the move
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get AI name for display."""
        pass

    def get_move_with_explanation(self, state: GameState) -> tuple[int, str]:
        """Get move with explanation (optional override).
        
        Args:
            state: Current game state
            
        Returns:
            Tuple of (column, explanation_string)
        """
        move = self.get_move(state)
        return move, f"Selected column {move}"
