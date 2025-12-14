"""Minimax AI with alpha-beta pruning."""

import math

from ..core.types import BoardState, GameState, Player
from ..game.rules import Connect4Rules
from .interface import AIInterface


class MinimaxAI(AIInterface):
    """Minimax AI with alpha-beta pruning.
    
    Features:
    - Configurable search depth
    - Alpha-beta pruning for efficiency
    - Heuristic evaluation for non-terminal states
    - Center column preference
    - Threat detection
    """

    def __init__(self, depth: int = 5, player: Player = Player.ORANGE):
        """Initialize Minimax AI.
        
        Args:
            depth: Search depth (higher = stronger but slower)
            player: Which player this AI controls
        """
        self.depth = depth
        self.player = player
        self.opponent = Player.YELLOW if player == Player.ORANGE else Player.ORANGE
        self.rules = Connect4Rules()
        self._last_explanation = ""

    def get_move(self, state: GameState) -> int:
        """Find best move using minimax with alpha-beta pruning."""
        if not state.legal_moves:
            raise ValueError("No legal moves available")

        # Quick check for immediate wins/blocks
        immediate = self._check_immediate_moves(state)
        if immediate is not None:
            return immediate

        # Run minimax
        _, best_col = self._minimax(
            state.board,
            self.depth,
            -math.inf,
            math.inf,
            True  # Maximizing
        )

        # Fallback to center if minimax returns invalid
        if best_col not in state.legal_moves:
            best_col = state.legal_moves[len(state.legal_moves) // 2]

        return best_col

    def _check_immediate_moves(self, state: GameState) -> int | None:
        """Check for immediate winning moves or blocks."""
        # Check if we can win immediately
        for col in state.legal_moves:
            test_board = self._simulate_move(state.board, col, self.player)
            winner, _ = self.rules.check_winner(test_board)
            if winner == self.player:
                self._last_explanation = f"Winning move at column {col}!"
                return col

        # Check if opponent can win (must block)
        for col in state.legal_moves:
            test_board = self._simulate_move(state.board, col, self.opponent)
            winner, _ = self.rules.check_winner(test_board)
            if winner == self.opponent:
                self._last_explanation = f"Blocking opponent win at column {col}"
                return col

        return None

    def _minimax(
        self,
        board: BoardState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool
    ) -> tuple[float, int]:
        """Minimax with alpha-beta pruning.
        
        Returns:
            Tuple of (score, best_column)
        """
        # Check terminal states
        winner, _ = self.rules.check_winner(board)

        if winner == self.player:
            return 10000 + depth, -1  # Prefer faster wins
        elif winner == self.opponent:
            return -10000 - depth, -1  # Avoid losses
        elif depth == 0 or self.rules.is_draw(board):
            return self._evaluate(board), -1

        legal_moves = self.rules.get_legal_moves(board)
        if not legal_moves:
            return 0, -1  # Draw

        # Order moves (center first for better pruning)
        ordered_moves = self._order_moves(legal_moves)

        if maximizing:
            max_eval = -math.inf
            best_col = ordered_moves[0]

            for col in ordered_moves:
                new_board = self._simulate_move(board, col, self.player)
                eval_score, _ = self._minimax(new_board, depth - 1, alpha, beta, False)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col

                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval, best_col
        else:
            min_eval = math.inf
            best_col = ordered_moves[0]

            for col in ordered_moves:
                new_board = self._simulate_move(board, col, self.opponent)
                eval_score, _ = self._minimax(new_board, depth - 1, alpha, beta, True)

                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col

                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval, best_col

    def _order_moves(self, moves: list[int]) -> list[int]:
        """Order moves for better alpha-beta pruning (center first)."""
        center = 2
        return sorted(moves, key=lambda x: abs(x - center))

    def _simulate_move(self, board: BoardState, col: int, player: Player) -> BoardState:
        """Create new board state with move applied."""
        new_grid = [[cell for cell in row] for row in board.grid]
        row = self.rules.get_landing_row(board, col)
        if row >= 0:
            new_grid[row][col] = player
        return BoardState(grid=new_grid, board_detected=True)

    def _evaluate(self, board: BoardState) -> float:
        """Heuristic board evaluation.
        
        Considers:
        - Center column control
        - Number of 2-in-a-row and 3-in-a-row
        - Blocking opponent threats
        """
        score = 0.0

        # Center column preference (column 2)
        center_col = 2
        for row in range(5):
            cell = board.grid[row][center_col]
            if cell == self.player:
                score += 3
            elif cell == self.opponent:
                score -= 3

        # Evaluate windows (groups of 4)
        score += self._evaluate_windows(board)

        return score

    def _evaluate_windows(self, board: BoardState) -> float:
        """Evaluate all possible 4-in-a-row windows."""
        score = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(5):
            for col in range(5):
                for dr, dc in directions:
                    window = self._get_window(board, row, col, dr, dc)
                    if window:
                        score += self._score_window(window)

        return score

    def _get_window(
        self, board: BoardState, row: int, col: int, dr: int, dc: int
    ) -> list[Player] | None:
        """Get a window of 4 cells in a direction."""
        window = []
        for i in range(4):
            r, c = row + i * dr, col + i * dc
            if 0 <= r < 5 and 0 <= c < 5:
                window.append(board.grid[r][c])
            else:
                return None
        return window

    def _score_window(self, window: list[Player]) -> float:
        """Score a window of 4 cells."""
        my_count = window.count(self.player)
        opp_count = window.count(self.opponent)
        empty_count = window.count(Player.EMPTY)

        # Can't score if both players in window
        if my_count > 0 and opp_count > 0:
            return 0

        # My pieces
        if my_count == 4:
            return 1000
        elif my_count == 3 and empty_count == 1:
            return 50  # One away from winning
        elif my_count == 2 and empty_count == 2:
            return 10

        # Opponent pieces (negative = bad for us)
        if opp_count == 4:
            return -1000
        elif opp_count == 3 and empty_count == 1:
            return -80  # Must block!
        elif opp_count == 2 and empty_count == 2:
            return -8

        return 0

    def get_name(self) -> str:
        return f"Minimax (depth={self.depth})"

    def get_move_with_explanation(self, state: GameState) -> tuple[int, str]:
        """Get move with explanation."""
        self._last_explanation = ""
        move = self.get_move(state)

        if not self._last_explanation:
            # Generate explanation based on evaluation
            self._last_explanation = self._generate_explanation(state, move)

        return move, self._last_explanation

    def _generate_explanation(self, state: GameState, move: int) -> str:
        """Generate explanation for a move."""
        # Check if it's a center move
        if move == 2:
            return f"Column {move} (center control)"

        # Check if it creates a threat
        test_board = self._simulate_move(state.board, move, self.player)
        my_threats = self._count_threats(test_board, self.player)

        if my_threats > 0:
            return f"Column {move} (creates {my_threats} threat(s))"

        return f"Column {move} (best evaluated move)"

    def _count_threats(self, board: BoardState, player: Player) -> int:
        """Count 3-in-a-row threats for a player."""
        threats = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(5):
            for col in range(5):
                for dr, dc in directions:
                    window = self._get_window(board, row, col, dr, dc)
                    if window:
                        if window.count(player) == 3 and window.count(Player.EMPTY) == 1:
                            threats += 1

        return threats
