"""Random AI for testing."""

import random

from ..core.types import GameState
from .interface import AIInterface


class RandomAI(AIInterface):
    """Random move AI for testing.
    
    Simply picks a random legal move. Useful for:
    - Testing game logic
    - Baseline comparison
    - Quick demos
    """

    def get_move(self, state: GameState) -> int:
        """Pick a random legal move."""
        if not state.legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(state.legal_moves)

    def get_name(self) -> str:
        return "Random AI"

    def get_move_with_explanation(self, state: GameState) -> tuple[int, str]:
        move = self.get_move(state)
        return move, f"Randomly selected column {move}"
