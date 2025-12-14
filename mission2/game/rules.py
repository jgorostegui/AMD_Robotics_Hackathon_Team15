"""Connect4 rules for a 5x5 board."""


from ..core.types import BoardState, Player, Position


class Connect4Rules:
    """Connect4 rules for a 5x5 board.
    
    Win condition: 4 in a row (horizontal, vertical, or diagonal)
    """

    def __init__(self, board_size: int = 5, win_length: int = 4):
        """Initialize rules.
        
        Args:
            board_size: Size of the board (5x5 default)
            win_length: Number in a row to win (4 default)
        """
        self.board_size = board_size
        self.win_length = win_length

    def get_legal_moves(self, board: BoardState) -> list[int]:
        """Get columns that aren't full.
        
        Args:
            board: Current board state
            
        Returns:
            List of column indices (0-4) that can accept a piece
        """
        legal = []
        for col in range(self.board_size):
            # Column is legal if top row is empty
            if board.grid[0][col] == Player.EMPTY:
                legal.append(col)
        return legal

    def get_landing_row(self, board: BoardState, column: int) -> int:
        """Get the row where a piece would land in given column.
        
        Args:
            board: Current board state
            column: Column to drop piece in
            
        Returns:
            Row index where piece lands, or -1 if column is full
        """
        for row in range(self.board_size - 1, -1, -1):
            if board.grid[row][column] == Player.EMPTY:
                return row
        return -1  # Column full

    def check_winner(self, board: BoardState) -> tuple[Player | None, list[Position]]:
        """Check if there's a winner.
        
        Args:
            board: Current board state
            
        Returns:
            Tuple of (winner, winning_positions). Winner is None if no winner.
        """
        # Check all 4 directions from each cell
        directions = [
            (0, 1),   # Horizontal (right)
            (1, 0),   # Vertical (down)
            (1, 1),   # Diagonal down-right
            (1, -1),  # Diagonal down-left
        ]

        for row in range(self.board_size):
            for col in range(self.board_size):
                player = board.grid[row][col]
                if player == Player.EMPTY:
                    continue

                for dr, dc in directions:
                    positions = self._check_direction(board, row, col, dr, dc, player)
                    if positions:
                        return player, positions

        return None, []

    def _check_direction(
        self,
        board: BoardState,
        start_row: int,
        start_col: int,
        dr: int,
        dc: int,
        player: Player
    ) -> list[Position]:
        """Check for win_length in a row in given direction.
        
        Returns:
            List of winning positions, or empty list if no win
        """
        positions = []

        for i in range(self.win_length):
            row = start_row + i * dr
            col = start_col + i * dc

            # Check bounds
            if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                return []

            # Check if same player
            if board.grid[row][col] != player:
                return []

            positions.append(Position(row=row, col=col))

        return positions

    def is_draw(self, board: BoardState) -> bool:
        """Check if game is a draw (board full, no winner).
        
        Args:
            board: Current board state
            
        Returns:
            True if draw (board full and no winner)
        """
        winner, _ = self.check_winner(board)
        if winner:
            return False

        # Check if any empty cells remain
        for row in board.grid:
            if Player.EMPTY in row:
                return False

        return True

    def is_valid_move(self, board: BoardState, column: int) -> bool:
        """Check if a move is valid.
        
        Args:
            board: Current board state
            column: Column to check
            
        Returns:
            True if move is valid
        """
        if column < 0 or column >= self.board_size:
            return False
        return column in self.get_legal_moves(board)

    def apply_move(self, board: BoardState, column: int, player: Player) -> tuple[BoardState, Position]:
        """Apply a move to the board (creates new BoardState).
        
        Args:
            board: Current board state
            column: Column to drop piece
            player: Player making the move
            
        Returns:
            Tuple of (new_board_state, position_where_piece_landed)
            
        Raises:
            ValueError: If move is invalid
        """
        if not self.is_valid_move(board, column):
            raise ValueError(f"Invalid move: column {column}")

        row = self.get_landing_row(board, column)
        if row < 0:
            raise ValueError(f"Column {column} is full")

        # Create new board with move applied
        new_grid = [[cell for cell in r] for r in board.grid]
        new_grid[row][column] = player

        new_board = BoardState(
            grid=new_grid,
            board_detected=board.board_detected,
            confidence=board.confidence,
        )

        return new_board, Position(row=row, col=column)
