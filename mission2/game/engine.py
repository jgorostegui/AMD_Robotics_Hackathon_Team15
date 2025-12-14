"""Game engine for Connect4 state management."""


from ..core.bus import EventBus, get_event_bus
from ..core.events import Event, EventType
from ..core.types import BoardState, GamePhase, GameState, Move, Player, Position
from .rules import Connect4Rules


class GameEngine:
    """Manages game state and enforces rules.
    
    Stateful engine that:
    - Tracks current game state
    - Validates moves
    - Detects wins/draws
    - Emits events for state changes
    """

    def __init__(self, rules: Connect4Rules | None = None, bus: EventBus | None = None):
        """Initialize game engine.
        
        Args:
            rules: Game rules (uses defaults if None)
            bus: Event bus (uses global if None)
        """
        self.rules = rules or Connect4Rules()
        self.bus = bus or get_event_bus()
        self._state: GameState | None = None
        self._vision_last_signature: tuple[tuple[Player, ...], ...] | None = None
        self._vision_stable_count: int = 0
        self._vision_last_board: BoardState | None = None

    def new_game(self, first_player: Player = Player.ORANGE) -> GameState:
        """Initialize a new game.
        
        Args:
            first_player: Who goes first (default: ORANGE/robot)
            
        Returns:
            Initial game state
        """
        empty_board = BoardState(
            grid=[[Player.EMPTY] * 5 for _ in range(5)],
            board_detected=True
        )

        phase = GamePhase.ROBOT_TURN if first_player == Player.ORANGE else GamePhase.HUMAN_TURN

        self._state = GameState(
            board=empty_board,
            phase=phase,
            current_player=first_player,
            move_history=[],
            winner=None,
            winning_positions=[],
            legal_moves=self.rules.get_legal_moves(empty_board),
            turn_number=1,
        )

        self.bus.publish(Event(
            type=EventType.GAME_STARTED,
            data={"first_player": first_player.value},
            source="game_engine"
        ))

        return self._state

    def make_move(self, column: int, player: Player) -> GameState:
        """Make a move.
        
        Args:
            column: Column to drop piece (0-4)
            player: Player making the move
            
        Returns:
            Updated game state
            
        Raises:
            ValueError: If game not started
        """
        if self._state is None:
            raise ValueError("Game not started. Call new_game() first.")

        # Apply move
        try:
            new_board, position = self.rules.apply_move(self._state.board, column, player)
        except ValueError as e:
            return self._state

        move = Move(column=column, player=player, position=position)

        # Check for win
        winner, winning_positions = self.rules.check_winner(new_board)

        if winner:
            self._state = self._create_game_over_state(
                new_board, move, winner, winning_positions
            )
            self.bus.publish(Event(
                type=EventType.GAME_WON,
                data={"winner": winner.value, "positions": winning_positions},
                source="game_engine"
            ))
        elif self.rules.is_draw(new_board):
            self._state = self._create_game_over_state(new_board, move, None, [])
            self.bus.publish(Event(
                type=EventType.GAME_DRAW,
                source="game_engine"
            ))
        else:
            # Continue game - switch turns
            next_player = Player.YELLOW if player == Player.ORANGE else Player.ORANGE
            next_phase = GamePhase.HUMAN_TURN if next_player == Player.YELLOW else GamePhase.ROBOT_TURN

            self._state = GameState(
                board=new_board,
                phase=next_phase,
                current_player=next_player,
                move_history=self._state.move_history + [move],
                winner=None,
                winning_positions=[],
                legal_moves=self.rules.get_legal_moves(new_board),
                turn_number=self._state.turn_number + 1,
            )

            self.bus.publish(Event(
                type=EventType.TURN_CHANGED,
                data={"player": next_player.value, "turn": self._state.turn_number},
                source="game_engine"
            ))

        self.bus.publish(Event(
            type=EventType.MOVE_MADE,
            data={"move": move, "column": column, "player": player.value},
            source="game_engine"
        ))

        return self._state

    def _create_game_over_state(
        self,
        board: BoardState,
        last_move: Move,
        winner: Player | None,
        winning_positions: list[Position]
    ) -> GameState:
        """Create a game over state."""
        return GameState(
            board=board,
            phase=GamePhase.GAME_OVER,
            current_player=last_move.player,  # Last player who moved
            move_history=self._state.move_history + [last_move],
            winner=winner,
            winning_positions=winning_positions,
            legal_moves=[],  # No more moves
            turn_number=self._state.turn_number + 1,
        )

    def update_from_vision(
        self,
        board_state: BoardState,
        *,
        stable_frames_required: int = 3,
        ignore_during_robot_moving: bool = True,
        max_missing_moves: int = 2,
        max_rewrite_moves: int = 1,
    ) -> GameState:
        """Update game state from vision detection.
        
        Detects if a move was made by comparing boards.
        
        Args:
            board_state: New board state from vision
            
        Returns:
            Updated game state
        """
        if self._state is None:
            return self.new_game()

        if ignore_during_robot_moving and self._state.phase == GamePhase.ROBOT_MOVING:
            # Vision is unreliable while the robot arm occludes the board.
            self._vision_last_signature = None
            self._vision_stable_count = 0
            self._vision_last_board = None
            return self._state

        if not board_state.board_detected:
            self.bus.publish(Event(
                type=EventType.DETECTION_FAILED,
                data={"reason": "board_not_detected"},
                source="game_engine"
            ))
            self._vision_last_signature = None
            self._vision_stable_count = 0
            self._vision_last_board = None
            return self._state

        # Temporal filtering: require N consecutive identical grids before committing.
        if stable_frames_required > 1:
            signature = tuple(tuple(cell for cell in row) for row in board_state.grid)
            if signature == self._vision_last_signature:
                self._vision_stable_count += 1
            else:
                self._vision_last_signature = signature
                self._vision_stable_count = 1
            self._vision_last_board = board_state

            if self._vision_stable_count < stable_frames_required:
                return self._state

            # Use the stable board snapshot for processing.
            board_state = self._vision_last_board
            self._vision_last_signature = None
            self._vision_stable_count = 0
            self._vision_last_board = None

            if board_state is None:
                return self._state

        # Fast path: identical grid, just refresh the board reference/meta.
        if self._boards_equal(self._state.board, board_state):
            self._state = GameState(
                board=board_state,
                phase=self._state.phase,
                current_player=self._state.current_player,
                move_history=self._state.move_history,
                winner=self._state.winner,
                winning_positions=self._state.winning_positions,
                legal_moves=self._state.legal_moves,
                turn_number=self._state.turn_number,
                error_message=self._state.error_message,
            )
            self.bus.publish(Event(
                type=EventType.VISION_STATE_UPDATED,
                data={"changes": 0},
                source="game_engine"
            ))
            return self._state

        # Try strict single-move detection first.
        move_detected = self._detect_move_from_boards(self._state.board, board_state)
        if move_detected:
            column, player = move_detected
            updated = self.make_move(column, player)
            # Preserve vision metadata (confidence/images) by swapping in the observed board.
            self._state = GameState(
                board=board_state,
                phase=updated.phase,
                current_player=updated.current_player,
                move_history=updated.move_history,
                winner=updated.winner,
                winning_positions=updated.winning_positions,
                legal_moves=updated.legal_moves,
                turn_number=updated.turn_number,
                error_message=updated.error_message,
            )
            self.bus.publish(Event(
                type=EventType.VISION_STATE_UPDATED,
                data={"changes": 1, "applied_move": {"column": column, "player": player.value}},
                source="game_engine"
            ))
            return self._state

        # Otherwise, attempt best-effort reconciliation for common mismatch cases.
        return self._reconcile_with_vision(
            board_state,
            max_missing_moves=max_missing_moves,
            max_rewrite_moves=max_rewrite_moves,
        )

    def _boards_equal(self, a: BoardState, b: BoardState) -> bool:
        """Compare board grids (ignores vision metadata)."""
        return a.grid == b.grid

    def _count_pieces(self, board: BoardState) -> dict[Player, int]:
        counts = {Player.ORANGE: 0, Player.YELLOW: 0, Player.EMPTY: 0}
        for row in board.grid:
            for cell in row:
                if cell in counts:
                    counts[cell] += 1
        return counts

    def _replay_board_from_history(self, moves: list[Move]) -> BoardState:
        board = BoardState(grid=[[Player.EMPTY] * self.rules.board_size for _ in range(self.rules.board_size)])
        for move in moves:
            board, _ = self.rules.apply_move(board, move.column, move.player)
        return board

    def _find_move_sequences(
        self,
        start_board: BoardState,
        target_board: BoardState,
        players: list[Player],
        *,
        max_solutions: int = 2,
    ) -> list[list[int]]:
        """Find sequences of columns that transform start_board into target_board."""
        solutions: list[list[int]] = []

        def backtrack(board: BoardState, idx: int, cols: list[int]) -> None:
            if len(solutions) >= max_solutions:
                return
            if idx >= len(players):
                if self._boards_equal(board, target_board):
                    solutions.append(cols.copy())
                return

            player = players[idx]
            for col in self.rules.get_legal_moves(board):
                try:
                    next_board, _ = self.rules.apply_move(board, col, player)
                except ValueError:
                    continue
                backtrack(next_board, idx + 1, cols + [col])

        backtrack(start_board, 0, [])
        return solutions

    def _reconcile_with_vision(
        self,
        observed: BoardState,
        *,
        max_missing_moves: int = 2,
        max_rewrite_moves: int = 1,
    ) -> GameState:
        """Best-effort reconcile engine state with an observed vision board.

        Handles two common discrepancy cases:
        - Missed detections: camera shows more pieces than engine (`max_missing_moves`).
        - Mis-executed last move (e.g., robot slip): same piece count but different grid
          (`max_rewrite_moves`).
        """
        assert self._state is not None

        current = self._state.board
        current_counts = self._count_pieces(current)
        observed_counts = self._count_pieces(observed)
        current_total = current_counts[Player.ORANGE] + current_counts[Player.YELLOW]
        observed_total = observed_counts[Player.ORANGE] + observed_counts[Player.YELLOW]
        delta_total = observed_total - current_total

        # Case 1: Vision shows extra pieces -> infer a small number of missing moves.
        if 0 < delta_total <= max_missing_moves:
            players: list[Player] = []
            next_player = self._state.current_player
            for _ in range(delta_total):
                players.append(next_player)
                next_player = Player.YELLOW if next_player == Player.ORANGE else Player.ORANGE

            sequences = self._find_move_sequences(current, observed, players)
            if len(sequences) == 1:
                cols = sequences[0]
                for col, player in zip(cols, players, strict=True):
                    self.make_move(col, player)

                assert self._state is not None
                # Swap in the observed board to retain vision metadata.
                self._state = GameState(
                    board=observed,
                    phase=self._state.phase,
                    current_player=self._state.current_player,
                    move_history=self._state.move_history,
                    winner=self._state.winner,
                    winning_positions=self._state.winning_positions,
                    legal_moves=self.rules.get_legal_moves(observed) if self._state.phase != GamePhase.GAME_OVER else [],
                    turn_number=self._state.turn_number,
                )
                self.bus.publish(Event(
                    type=EventType.VISION_STATE_UPDATED,
                    data={"changes": delta_total, "inferred_columns": cols},
                    source="game_engine"
                ))
                return self._state

        # Case 2: Same piece count, different grid -> try rewriting last move(s).
        if delta_total == 0 and max_rewrite_moves > 0 and self._state.move_history:
            for k in range(1, min(max_rewrite_moves, len(self._state.move_history)) + 1):
                prefix_moves = self._state.move_history[:-k]
                tail_moves = self._state.move_history[-k:]
                prefix_board = self._replay_board_from_history(prefix_moves)
                players = [m.player for m in tail_moves]

                sequences = self._find_move_sequences(prefix_board, observed, players)
                if len(sequences) == 1:
                    cols = sequences[0]

                    # Rebuild tail moves with corrected columns/positions.
                    rebuilt_tail: list[Move] = []
                    board = prefix_board
                    for col, player in zip(cols, players, strict=True):
                        board, pos = self.rules.apply_move(board, col, player)
                        rebuilt_tail.append(Move(column=col, player=player, position=pos))

                    full_history = prefix_moves + rebuilt_tail
                    winner, winning_positions = self.rules.check_winner(observed)
                    if winner:
                        phase = GamePhase.GAME_OVER
                        current_player = full_history[-1].player
                        legal_moves: list[int] = []
                    elif self.rules.is_draw(observed):
                        phase = GamePhase.GAME_OVER
                        current_player = full_history[-1].player if full_history else self._state.current_player
                        legal_moves = []
                    else:
                        last_player = full_history[-1].player
                        current_player = Player.YELLOW if last_player == Player.ORANGE else Player.ORANGE
                        phase = GamePhase.HUMAN_TURN if current_player == Player.YELLOW else GamePhase.ROBOT_TURN
                        legal_moves = self.rules.get_legal_moves(observed)

                    self._state = GameState(
                        board=observed,
                        phase=phase,
                        current_player=current_player,
                        move_history=full_history,
                        winner=winner,
                        winning_positions=winning_positions,
                        legal_moves=legal_moves,
                        turn_number=len(full_history) + 1,
                    )
                    self.bus.publish(Event(
                        type=EventType.VISION_STATE_UPDATED,
                        data={"changes": 0, "rewritten_moves": k, "corrected_columns": cols},
                        source="game_engine"
                    ))
                    return self._state

        # If we get here, we couldn't reconcile automatically.
        error = {
            "delta_total": delta_total,
            "engine_total": current_total,
            "vision_total": observed_total,
        }
        self._state = GameState(
            board=current,
            phase=GamePhase.ERROR,
            current_player=self._state.current_player,
            move_history=self._state.move_history,
            winner=self._state.winner,
            winning_positions=self._state.winning_positions,
            legal_moves=self._state.legal_moves,
            turn_number=self._state.turn_number,
            error_message=f"Vision/engine desync: {error}",
        )
        self.bus.publish(Event(
            type=EventType.VISION_STATE_DESYNC,
            data=error,
            source="game_engine"
        ))
        return self._state

    def _detect_move_from_boards(
        self,
        old_board: BoardState,
        new_board: BoardState
    ) -> tuple[int, Player] | None:
        """Detect what move was made by comparing boards.
        
        Returns:
            Tuple of (column, player) if a single move detected, None otherwise
        """
        changes = []

        size = self.rules.board_size
        for row in range(size):
            for col in range(size):
                old_cell = old_board.grid[row][col]
                new_cell = new_board.grid[row][col]

                if old_cell == Player.EMPTY and new_cell != Player.EMPTY:
                    changes.append((row, col, new_cell))

        # Should be exactly one new piece
        if len(changes) == 1:
            row, col, player = changes[0]
            return (col, player)

        return None

    def reset(self) -> None:
        """Reset game state."""
        self._state = None
        self.bus.publish(Event(
            type=EventType.GAME_RESET,
            source="game_engine"
        ))

    @property
    def state(self) -> GameState | None:
        """Get current game state."""
        return self._state

    @property
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return self._state is not None and self._state.phase == GamePhase.GAME_OVER

    @property
    def is_robot_turn(self) -> bool:
        """Check if it's the robot's turn."""
        return self._state is not None and self._state.phase == GamePhase.ROBOT_TURN

    @property
    def is_human_turn(self) -> bool:
        """Check if it's the human's turn."""
        return self._state is not None and self._state.phase == GamePhase.HUMAN_TURN
