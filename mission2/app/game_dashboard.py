"""Streamlit dashboard for Connect4 game testing.

Test game engine, AI, robot control, and vision in a visual interface.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from mission2.ai.minimax import MinimaxAI
from mission2.ai.random_ai import RandomAI
from mission2.core.bus import reset_event_bus
from mission2.core.config import get_settings
from mission2.core.types import GamePhase, Move, Player
from mission2.game.engine import GameEngine
from mission2.robot.lerobot_record import load_dotenv
from mission2.robot.mock import MockRobot
from mission2.streaming.http_receiver import HTTPFrameReceiver
from mission2.vision.board_detector import grid_to_ascii, run_detection
from mission2.vision.calibration import load_calibration_file
from mission2.vision.mock_detector import MockVisionDetector


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _vision_params():
    """Return (config, manual_corners, min_cell_frac, cell_radius_frac, grid_method)."""
    settings = get_settings()
    cal = load_calibration_file(settings.vision.calibration_path)
    config = cal.to_hsv_config() if cal is not None else None
    manual_corners = cal.corners if (cal is not None and cal.corners and len(cal.corners) == 4) else None
    min_cell_frac = float(cal.min_cell_frac) if cal is not None else 0.11
    cell_radius_frac = float(cal.cell_radius_frac) if cal is not None else 0.33
    grid_method = (os.environ.get("MISSION2_GRID_METHOD") or settings.ui.grid_method or "sample_hsv")
    return config, manual_corners, min_cell_frac, cell_radius_frac, grid_method


def init_session_state():
    """Initialize session state."""
    # Load env so dashboard defaults (policy/backend/etc.) are available on first render.
    load_dotenv(_project_root() / ".env", override=False)

    if "robot_mode" not in st.session_state:
        st.session_state.robot_mode = "SmolVLA"
    if "smolvla_policy" not in st.session_state:
        st.session_state.smolvla_policy = os.environ.get("SMOLVLA_POLICY", "")
    if "smolvla_subprocess_backend" not in st.session_state:
        st.session_state.smolvla_subprocess_backend = os.environ.get(
            "MISSION2_SUBPROCESS_BACKEND", "lerobot-record"
        )
    if "smolvla_display_data" not in st.session_state:
        st.session_state.smolvla_display_data = False

    if "engine" not in st.session_state:
        reset_event_bus()
        st.session_state.engine = GameEngine()
        st.session_state.game = None

    if "ai" not in st.session_state:
        st.session_state.ai = MinimaxAI(depth=5, player=Player.ORANGE)

    if "robot" not in st.session_state:
        if st.session_state.robot_mode == "SmolVLA" and st.session_state.smolvla_policy.strip():
            try:
                from mission2.robot.smolvla import SmolVLARobot

                os.environ["MISSION2_SUBPROCESS_BACKEND"] = st.session_state.smolvla_subprocess_backend
                st.session_state.robot = SmolVLARobot(
                    policy_path=st.session_state.smolvla_policy.strip(),
                    display_data=bool(st.session_state.smolvla_display_data),
                )
                st.session_state.robot.connect()
            except Exception:
                st.session_state.robot = MockRobot(move_delay=0.1)
                st.session_state.robot.connect()
        else:
            st.session_state.robot = MockRobot(move_delay=0.1)
            st.session_state.robot.connect()

    # Vision state
    settings = get_settings()
    if "vision" not in st.session_state:
        st.session_state.vision = MockVisionDetector()
        st.session_state.vision.connect()
    if "vision_mode" not in st.session_state:
        st.session_state.vision_mode = "Mock"
    if "vision_host" not in st.session_state:
        st.session_state.vision_host = settings.stream.host
    if "vision_port" not in st.session_state:
        st.session_state.vision_port = 8080  # HTTP port
    if "vision_live_mode" not in st.session_state:
        st.session_state.vision_live_mode = False
    if "http_receiver" not in st.session_state:
        st.session_state.http_receiver = HTTPFrameReceiver(
            host=settings.stream.host,
            port=8080,
            output_dir=settings.stream.output_dir,
        )
    if "vision_last_result" not in st.session_state:
        st.session_state.vision_last_result = None
    if "vision_timeout_s" not in st.session_state:
        st.session_state.vision_timeout_s = 5.0
    if "vision_retries" not in st.session_state:
        st.session_state.vision_retries = 2

    if "auto_play" not in st.session_state:
        st.session_state.auto_play = False

    if "ai_vs_ai_running" not in st.session_state:
        st.session_state.ai_vs_ai_running = False
    if "ai_vs_ai_delay_s" not in st.session_state:
        st.session_state.ai_vs_ai_delay_s = 0.5

    if "game_log" not in st.session_state:
        st.session_state.game_log = []
    if "pending_robot_move" not in st.session_state:
        st.session_state.pending_robot_move = None
    if "pending_robot_explanation" not in st.session_state:
        st.session_state.pending_robot_explanation = ""


def render_board(game_state) -> None:
    """Render the game board with clickable columns."""
    if game_state is None:
        st.info("Click 'New Game' to start")
        return

    symbols = {
        Player.ORANGE: "🟠",
        Player.YELLOW: "🟡",
        Player.EMPTY: "⚪",
    }

    # Highlight winning positions
    winning_set = set()
    if game_state.winning_positions:
        winning_set = {(p.row, p.col) for p in game_state.winning_positions}

    pending_col = st.session_state.get("pending_robot_move", None)

    # Render board
    for row_idx, row in enumerate(game_state.board.grid):
        cols = st.columns(5)
        for col_idx, cell in enumerate(row):
            with cols[col_idx]:
                symbol = symbols[cell]
                pending_style = ""
                if pending_col is not None and int(pending_col) == int(col_idx):
                    pending_style = "border:2px solid #ffcc00;border-radius:8px;"
                # Highlight winning cells
                if (row_idx, col_idx) in winning_set:
                    st.markdown(
                        f"<div style='text-align:center;font-size:2.5rem;background:#90EE90;{pending_style}'>{symbol}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div style='text-align:center;font-size:2.5rem;{pending_style}'>{symbol}</div>",
                        unsafe_allow_html=True
                    )

    # Column buttons for human moves
    if game_state.phase == GamePhase.HUMAN_TURN:
        st.markdown("---")
        st.markdown("**Your turn - click a column:**")
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                disabled = i not in game_state.legal_moves
                if st.button(f"Col {i}", key=f"col_{i}", disabled=disabled, use_container_width=True):
                    make_human_move(i)


def make_human_move(column: int):
    """Handle human move."""
    engine = st.session_state.engine
    game = st.session_state.game

    if game and game.phase == GamePhase.HUMAN_TURN:
        st.session_state.game = engine.make_move(column, Player.YELLOW)
        st.session_state.game_log.append(f"🟡 Human played column {column}")
        st.rerun()


def make_ai_move():
    """Handle AI move."""
    engine = st.session_state.engine
    game = st.session_state.game
    ai = st.session_state.ai
    robot = st.session_state.robot

    if game and game.phase == GamePhase.ROBOT_TURN:
        # Get AI move
        move, explanation = ai.get_move_with_explanation(game)
        st.session_state.game_log.append(f"🤖 AI chose column {move}: {explanation}")

        # Execute robot move
        game.phase = GamePhase.ROBOT_MOVING
        robot_move = Move(column=move, player=Player.ORANGE)
        st.session_state.game_log.append("🦾 Starting robot subprocess...")
        with st.spinner("Running robot (this can take a while on first load)..."):
            action = robot.execute_move(robot_move)
        st.session_state.game_log.append(f"🦾 Robot: {robot.get_last_instruction()}")

        if getattr(action, "status", None) is None or action.status.name != "COMPLETED":
            game.phase = GamePhase.ERROR
            st.session_state.game_log.append(f"❌ Robot move failed: {getattr(action, 'error', 'unknown error')}")
            st.rerun()

        # Update game state only on success
        st.session_state.game = engine.make_move(move, Player.ORANGE)
        st.rerun()


def plan_ai_move() -> None:
    """Compute and display the next robot move without executing it yet."""
    engine = st.session_state.engine
    game = st.session_state.game
    ai = st.session_state.ai

    if not game or game.phase != GamePhase.ROBOT_TURN:
        return

    move, explanation = ai.get_move_with_explanation(game)
    st.session_state.pending_robot_move = int(move)
    st.session_state.pending_robot_explanation = str(explanation)
    game.phase = GamePhase.ROBOT_MOVING
    st.session_state.game = game
    st.session_state.game_log.append(f"🤖 Planned move: column {move} ({explanation})")


def execute_planned_move() -> None:
    """Execute the previously planned robot move."""
    engine = st.session_state.engine
    game = st.session_state.game
    robot = st.session_state.robot
    move = st.session_state.get("pending_robot_move", None)

    if game is None or move is None:
        return

    game.phase = GamePhase.ROBOT_MOVING
    st.session_state.game = game

    robot_move = Move(column=int(move), player=Player.ORANGE)
    st.session_state.game_log.append("🦾 Starting robot subprocess...")
    with st.spinner("Running robot (this can take a while on first load)..."):
        action = robot.execute_move(robot_move)
    st.session_state.game_log.append(f"🦾 Robot: {robot.get_last_instruction()}")

    if getattr(action, "status", None) is None or action.status.name != "COMPLETED":
        game.phase = GamePhase.ERROR
        st.session_state.game_log.append(f"❌ Robot move failed: {getattr(action, 'error', 'unknown error')}")
        st.session_state.pending_robot_move = None
        st.session_state.pending_robot_explanation = ""
        st.session_state.game = game
        return

    st.session_state.game = engine.make_move(int(move), Player.ORANGE)
    st.session_state.pending_robot_move = None
    st.session_state.pending_robot_explanation = ""


def cancel_planned_move() -> None:
    game = st.session_state.game
    if game is not None and game.phase == GamePhase.ROBOT_MOVING:
        game.phase = GamePhase.ROBOT_TURN
        st.session_state.game = game
    st.session_state.pending_robot_move = None
    st.session_state.pending_robot_explanation = ""


def render_sidebar():
    """Render sidebar controls."""
    st.sidebar.header("🎮 Game Controls")

    # New game button
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🆕 New Game", use_container_width=True):
            first = Player.YELLOW  # Human first by default
            st.session_state.game = st.session_state.engine.new_game(first_player=first)
            st.session_state.game_log = ["Game started! Human (🟡) goes first."]

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.engine.reset()
            st.session_state.game = None
            st.session_state.game_log = []
            st.rerun()

    # Who goes first
    first_player = st.sidebar.radio(
        "First player",
        ["Human (🟡)", "Robot (🟠)"],
        horizontal=True
    )

    st.sidebar.markdown("---")

    # Robot Settings
    st.sidebar.subheader("🤖 Robot Settings")
    robot = st.session_state.robot

    mode = st.sidebar.selectbox("Mode", ["Mock", "SmolVLA"], key="robot_mode")
    if mode == "SmolVLA":
        st.sidebar.text_input(
            "Policy path / HF repo",
            key="smolvla_policy",
            help="Example: jlamperez/smolvla_connect4",
        )
        st.sidebar.selectbox(
            "Subprocess backend",
            options=["direct-inference", "lerobot-record"],
            key="smolvla_subprocess_backend",
            help="direct-inference runs simple_run.py (no dataset saving). lerobot-record records datasets.",
        )
        st.sidebar.toggle(
            "Display data",
            key="smolvla_display_data",
            help="Show camera windows/visualization (subprocess-dependent).",
        )
        if st.sidebar.button("Apply Robot", use_container_width=True):
            policy_path = st.session_state.get("smolvla_policy", "").strip()
            if not policy_path:
                st.sidebar.error("Enter a policy path/repo to use SmolVLA.")
            else:
                try:
                    from mission2.robot.smolvla import SmolVLARobot

                    try:
                        st.session_state.robot.disconnect()
                    except Exception:
                        pass
                    os.environ["MISSION2_SUBPROCESS_BACKEND"] = st.session_state.get(
                        "smolvla_subprocess_backend", "direct-inference"
                    )
                    st.session_state.robot = SmolVLARobot(
                        policy_path=policy_path.strip(),
                        display_data=bool(st.session_state.get("smolvla_display_data", False)),
                    )
                    st.session_state.robot.connect()
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to initialize SmolVLA robot: {e}")
        st.sidebar.caption(f"Active backend: {os.environ.get('MISSION2_SUBPROCESS_BACKEND', '')}")
    else:
        if st.sidebar.button("Apply Robot", use_container_width=True):
            try:
                st.session_state.robot.disconnect()
            except Exception:
                pass
            st.session_state.robot = MockRobot(move_delay=0.1)
            st.session_state.robot.connect()
            st.rerun()

    status = "🟢 Connected" if robot.is_connected() else "🔴 Disconnected"
    st.sidebar.markdown(f"**Status:** {status}")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Connect", use_container_width=True):
            robot.connect()
            st.rerun()
    with col2:
        if st.button("Home", use_container_width=True):
            robot.go_home()
            st.session_state.game_log.append("🦾 Robot moved to home position")
            st.rerun()

    st.sidebar.markdown("---")

    # Vision Settings
    st.sidebar.subheader("📷 Vision Settings")

    st.sidebar.selectbox(
        "Vision Mode",
        ["Mock", "Real (HTTP)"],
        key="vision_mode",
    )

    vision = st.session_state.vision
    is_real_mode = st.session_state.vision_mode == "Real (HTTP)"
    if is_real_mode and not st.session_state.get("_vision_live_auto_enabled", False):
        st.session_state.vision_live_mode = True
        st.session_state._vision_live_auto_enabled = True
    if not is_real_mode:
        st.session_state._vision_live_auto_enabled = False
    
    if is_real_mode:
        settings = get_settings()
        # Ensure defaults are set
        if not st.session_state.get("vision_host"):
            st.session_state.vision_host = settings.stream.host
        if not st.session_state.get("vision_port"):
            st.session_state.vision_port = 8080
        st.sidebar.text_input("Host", value=st.session_state.vision_host, key="vision_host")
        st.sidebar.number_input("Port", min_value=1, max_value=65535, value=st.session_state.vision_port, key="vision_port")
        
        # Update HTTP receiver settings (use defaults if empty)
        host = st.session_state.vision_host.strip() or settings.stream.host
        port = int(st.session_state.vision_port or 8080)
        st.session_state.http_receiver.update_settings(host, port)
        
        receiver = st.session_state.http_receiver
        st.sidebar.caption(f"URL: {receiver.frame_url}")
        st.sidebar.slider(
            "Capture timeout (s)",
            min_value=1.0,
            max_value=10.0,
            value=float(st.session_state.get("vision_timeout_s", 5.0)),
            step=0.5,
            key="vision_timeout_s",
        )
        st.sidebar.slider(
            "Retries",
            min_value=0,
            max_value=5,
            value=int(st.session_state.get("vision_retries", 2)),
            step=1,
            key="vision_retries",
        )
        cal = load_calibration_file(settings.vision.calibration_path)
        if cal is None:
            st.sidebar.warning("No calibration file found; vision will use defaults (less reliable).")
        else:
            st.sidebar.caption(f"Calibration: `{settings.vision.calibration_path}`")
            st.sidebar.caption(f"Corners: {'yes' if cal.corners else 'no'}")
            st.sidebar.caption(f"Grid method: `{os.environ.get('MISSION2_GRID_METHOD', settings.ui.grid_method)}`")

    # Live mode toggle (Real mode only)
    if is_real_mode:
        st.sidebar.toggle(
            "🔴 Live Mode",
            key="vision_live_mode",
            help="Continuously sync vision to game board",
        )
        if st.session_state.vision_live_mode:
            st.sidebar.slider("Interval (s)", 1.0, 5.0, 2.0, 0.5, key="live_interval")

    st.sidebar.markdown("---")

    # Game mode
    st.sidebar.subheader("🎯 Game Mode")

    mode = st.sidebar.radio(
        "Mode",
        ["Human vs AI", "AI vs AI"],
        horizontal=True
    )

    if mode == "AI vs AI":
        st.sidebar.slider("Step delay (seconds)", 0.0, 2.0, key="ai_vs_ai_delay_s")
        if not st.session_state.ai_vs_ai_running:
            if st.sidebar.button("▶️ Start AI vs AI", use_container_width=True):
                engine = st.session_state.engine
                st.session_state.game = engine.new_game(first_player=Player.ORANGE)
                st.session_state.game_log = ["AI vs AI game started!"]
                st.session_state.ai_vs_ai_running = True
                st.rerun()
        else:
            if st.sidebar.button("⏹ Stop AI vs AI", use_container_width=True):
                st.session_state.ai_vs_ai_running = False
                st.rerun()
    else:
        # Ensure background stepping doesn't continue in the wrong mode.
        st.session_state.ai_vs_ai_running = False

    st.sidebar.markdown("---")

    # AI Settings (moved to bottom)
    st.sidebar.subheader("🧠 AI Settings")

    ai_type = st.sidebar.selectbox("AI Type", ["Minimax", "Random"])

    if ai_type == "Minimax":
        depth = st.sidebar.slider("Search Depth", 1, 7, 5)
        if st.sidebar.button("Apply AI Settings"):
            st.session_state.ai = MinimaxAI(depth=depth, player=Player.ORANGE)
            st.sidebar.success(f"AI updated: Minimax depth={depth}")
    else:
        if st.sidebar.button("Apply AI Settings"):
            st.session_state.ai = RandomAI()
            st.sidebar.success("AI updated: Random")


def _step_ai_vs_ai_game() -> None:
    """Advance AI vs AI by one move (non-blocking Streamlit pattern)."""
    if not st.session_state.ai_vs_ai_running:
        return

    engine = st.session_state.engine
    game = st.session_state.game
    if game is None:
        st.session_state.ai_vs_ai_running = False
        return

    if game.phase == GamePhase.GAME_OVER:
        st.session_state.ai_vs_ai_running = False
        if game.winner:
            winner = "Orange (🟠)" if game.winner == Player.ORANGE else "Yellow (🟡)"
            st.session_state.game_log.append(f"🏆 {winner} wins!")
        else:
            st.session_state.game_log.append("🤝 Draw!")
        return

    ai_orange = st.session_state.ai
    ai_yellow = MinimaxAI(depth=3, player=Player.YELLOW)
    robot = st.session_state.robot

    if game.current_player == Player.ORANGE:
        move = ai_orange.get_move(game)
        st.session_state.game_log.append(f"🟠 Orange AI: column {move}")
        game.phase = GamePhase.ROBOT_MOVING
        st.session_state.game_log.append("🦾 Starting robot subprocess...")
        with st.spinner("Running robot (this can take a while on first load)..."):
            action = robot.execute_move(Move(column=move, player=Player.ORANGE))
        st.session_state.game_log.append(f"🦾 Robot: {robot.get_last_instruction()}")
        if getattr(action, "status", None) is None or action.status.name != "COMPLETED":
            game.phase = GamePhase.ERROR
            st.session_state.ai_vs_ai_running = False
            st.session_state.game_log.append(f"❌ Robot move failed: {getattr(action, 'error', 'unknown error')}")
            st.rerun()
    else:
        move = ai_yellow.get_move(game)
        st.session_state.game_log.append(f"🟡 Yellow AI: column {move}")

    st.session_state.game = engine.make_move(move, game.current_player)

    time.sleep(float(st.session_state.get("ai_vs_ai_delay_s", 0.5)))
    st.rerun()


def render_status_panel(game_state):
    """Render game status panel."""
    st.subheader("📊 Game Status")

    if game_state is None:
        st.info("No active game")
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Turn", game_state.turn_number)

    with col2:
        phase_display = {
            GamePhase.ROBOT_TURN: "🟠 Robot's Turn",
            GamePhase.HUMAN_TURN: "🟡 Human's Turn",
            GamePhase.GAME_OVER: "🏁 Game Over",
            GamePhase.ROBOT_MOVING: "🦾 Robot Moving",
            GamePhase.WAITING_FOR_START: "⏳ Waiting",
            GamePhase.ERROR: "❌ Error",
        }
        st.metric("Phase", phase_display.get(game_state.phase, "Unknown"))

    with col3:
        if game_state.winner:
            winner = "🟠 Robot" if game_state.winner == Player.ORANGE else "🟡 Human"
            st.metric("Winner", winner)
        elif game_state.phase == GamePhase.GAME_OVER:
            st.metric("Result", "🤝 Draw")
        else:
            current = "🟠" if game_state.current_player == Player.ORANGE else "🟡"
            st.metric("Current", current)

    # Legal moves
    if game_state.legal_moves:
        st.markdown(f"**Legal moves:** {game_state.legal_moves}")


def render_robot_panel():
    """Render robot instruction panel."""
    st.subheader("🦾 SmolVLA Prompt")

    robot = st.session_state.robot

    # Last instruction (SmolVLA prompt)
    last_instr = robot.get_last_instruction()
    if last_instr:
        st.code(last_instr, language=None)
    else:
        st.info("No instructions yet")


def render_game_log():
    """Render game event log."""
    st.subheader("📜 Game Log")

    log = st.session_state.game_log
    if log:
        # Show last 10 entries
        for entry in log[-10:]:
            st.text(entry)
    else:
        st.info("No events yet")

    if st.button("Clear Log"):
        st.session_state.game_log = []
        st.rerun()


def render_vision_panel(result):
    """Render vision detection panel with detection result."""
    st.subheader("📷 Vision")
    
    is_mock = st.session_state.vision_mode == "Mock"
    settings = get_settings()
    warped_path = os.path.join(settings.stream.output_dir, "game_warped.jpg")
    
    if is_mock:
        # Mock mode: editable grid
        vision = st.session_state.vision
        st.markdown("**Mock Board:**")
        _render_mock_editor(vision)
        
        if st.button("📥 Sync to Game", use_container_width=True):
            _sync_grid_to_game(vision.get_grid())
            st.rerun()
    else:
        # Real mode: show detection result
        if result and not result.error:
            st.code(grid_to_ascii(result.grid), language=None)
            if os.path.exists(warped_path) and os.path.getsize(warped_path) > 100:
                st.image(warped_path, caption="Warped", width="stretch")
        elif result and result.error:
            st.warning(f"Detection: {result.error}")
        else:
            st.info("Waiting for detection...")





def _render_mock_editor(vision: MockVisionDetector) -> None:
    """Render clickable mock board editor."""
    for row_idx in range(5):
        cols = st.columns(5)
        for col_idx in range(5):
            with cols[col_idx]:
                cell = vision.get_grid()[row_idx][col_idx]
                symbols = {"orange": "🟠", "yellow": "🟡", None: "⚪"}
                if st.button(
                    symbols.get(cell, "⚪"),
                    key=f"mock_{row_idx}_{col_idx}",
                    use_container_width=True,
                ):
                    vision.toggle_cell(row_idx, col_idx)
                    st.rerun()

    if st.button("🗑️ Clear Mock Board", use_container_width=True):
        vision.clear_board()
        st.rerun()


def _sync_grid_to_game(grid: list[list[str | None]]):
    """Sync a grid to game state."""
    engine = st.session_state.engine

    # Count pieces
    orange_count = sum(1 for row in grid for cell in row if cell == "orange")
    yellow_count = sum(1 for row in grid for cell in row if cell == "yellow")
    total_pieces = orange_count + yellow_count

    # Determine whose turn it is based on piece count
    if orange_count > yellow_count:
        current_player = Player.YELLOW
        phase = GamePhase.HUMAN_TURN
    else:
        current_player = Player.ORANGE
        phase = GamePhase.ROBOT_TURN

    # Create fresh game state
    game = engine.new_game(first_player=Player.ORANGE)
    
    # Update board from vision
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            if cell == "orange":
                game.board.grid[row_idx][col_idx] = Player.ORANGE
            elif cell == "yellow":
                game.board.grid[row_idx][col_idx] = Player.YELLOW
            else:
                game.board.grid[row_idx][col_idx] = Player.EMPTY

    # Update game state
    game.current_player = current_player
    game.turn_number = total_pieces + 1
    
    # Calculate legal moves (columns that aren't full)
    legal_moves = [col for col in range(5) if game.board.grid[0][col] == Player.EMPTY]
    game.legal_moves = legal_moves

    # Check for winner using rules
    from mission2.game.rules import Connect4Rules
    rules = Connect4Rules()
    winner, winning_positions = rules.check_winner(game.board)
    
    if winner:
        game.winner = winner
        game.winning_positions = winning_positions
        game.phase = GamePhase.GAME_OVER
        st.session_state.game_log.append(f"📥 Synced: {'🟠 Robot' if winner == Player.ORANGE else '🟡 Human'} wins!")
    elif not legal_moves:
        game.phase = GamePhase.GAME_OVER
        st.session_state.game_log.append("📥 Synced: Draw!")
    else:
        game.phase = phase
        turn_str = "🟠 Robot's turn" if current_player == Player.ORANGE else "🟡 Human's turn"
        st.session_state.game_log.append(f"📥 Synced ({orange_count}🟠, {yellow_count}🟡) - {turn_str}")

    st.session_state.game = game


def main():
    """Main dashboard entry point."""
    st.set_page_config(page_title="Connect4 Game Dashboard", page_icon="🎮", layout="wide")
    st.title("🎮 Connect4 Game Dashboard")

    init_session_state()
    render_sidebar()

    settings = get_settings()
    frame_path = settings.stream.frame_path
    warped_path = os.path.join(settings.stream.output_dir, "game_warped.jpg")
    
    live_mode = st.session_state.get("vision_live_mode", False)
    is_real_mode = st.session_state.vision_mode == "Real (HTTP)"
    result = None

    # Vision detection (like calibrate app)
    if is_real_mode:
        receiver = st.session_state.http_receiver
        host = st.session_state.get("vision_host") or settings.stream.host
        port = int(st.session_state.get("vision_port") or 8080)
        receiver.update_settings(host, port)

        cfg, manual_corners, min_cell_frac, cell_radius_frac, grid_method = _vision_params()
        timeout_s = float(st.session_state.get("vision_timeout_s", 5.0))
        retries = int(st.session_state.get("vision_retries", 2))
        
        if live_mode:
            # Live mode: capture + detect
            interval = float(st.session_state.get("live_interval", 2.0))
            frame = None
            for _ in range(max(1, retries + 1)):
                frame = receiver.capture_frame(timeout=timeout_s, save=True)
                if frame is not None:
                    break
            if frame is None:
                # Fallback: try MJPEG/OpenCV method (sometimes more tolerant than urllib)
                frame = receiver.capture_frame_cv(save=True)
            
            if frame is None:
                st.error(f"🔴 Capture failed: {receiver.last_error}")
                st.info(f"Start HTTP server: `python scripts/stream_http.py --port {port}`")
            else:
                time.sleep(0.05)
                result = run_detection(
                    frame_path,
                    config=cfg,
                    manual_corners=manual_corners,
                    min_cell_frac=min_cell_frac,
                    cell_radius_frac=cell_radius_frac,
                    grid_method=str(grid_method),
                )
                if result.warped_image is not None:
                    tmp = os.path.join(settings.stream.output_dir, ".game_warped_tmp.jpg")
                    cv2.imwrite(tmp, result.warped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.replace(tmp, warped_path)
                if not result.error:
                    _sync_grid_to_game(result.grid)
                st.session_state.vision_last_result = result
        else:
            # Manual mode: buttons
            def do_capture():
                f = None
                for _ in range(max(1, retries + 1)):
                    f = receiver.capture_frame(timeout=timeout_s, save=True)
                    if f is not None:
                        break
                if f is None:
                    f = receiver.capture_frame_cv(save=True)
                st.session_state._capture_ok = f is not None
                st.session_state._capture_err = receiver.last_error if f is None else None

            def do_detect():
                st.session_state._do_detect = True

            col1, col2 = st.columns(2)
            with col1:
                st.button("📷 Capture", on_click=do_capture, use_container_width=True)
            with col2:
                st.button("🔍 Detect & Sync", on_click=do_detect, type="primary", 
                         use_container_width=True, disabled=not os.path.exists(frame_path))

            if st.session_state.get("_capture_ok"):
                st.success("Frame captured!")
                st.session_state._capture_ok = False
            if st.session_state.get("_capture_err"):
                st.error(f"Capture failed: {st.session_state._capture_err}")
                st.session_state._capture_err = None

            if st.session_state.get("_do_detect") and os.path.exists(frame_path):
                st.session_state._do_detect = False
                time.sleep(0.05)
                result = run_detection(
                    frame_path,
                    config=cfg,
                    manual_corners=manual_corners,
                    min_cell_frac=min_cell_frac,
                    cell_radius_frac=cell_radius_frac,
                    grid_method=str(grid_method),
                )
                if result.warped_image is not None:
                    tmp = os.path.join(settings.stream.output_dir, ".game_warped_tmp.jpg")
                    cv2.imwrite(tmp, result.warped_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    os.replace(tmp, warped_path)
                if not result.error:
                    _sync_grid_to_game(result.grid)
                st.session_state._last_vision_result = result
                st.session_state.vision_last_result = result

            result = st.session_state.get("_last_vision_result")

    # Main layout
    col_board, col_vision, col_status = st.columns([2, 2, 1])

    with col_board:
        st.subheader("🎯 Game Board")
        render_board(st.session_state.game)
        _step_ai_vs_ai_game()
        
        game = st.session_state.game
        pending_move = st.session_state.get("pending_robot_move", None)
        if game and game.phase == GamePhase.ROBOT_TURN and pending_move is None:
            if st.button("🤖 Plan AI Move", use_container_width=True):
                plan_ai_move()
                st.rerun()
        if pending_move is not None:
            st.info(f"Planned robot move: column {pending_move}. Click Execute to run it.")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("▶️ Execute Robot Move", use_container_width=True, type="primary"):
                    execute_planned_move()
                    st.rerun()
            with col_b:
                if st.button("✖ Cancel", use_container_width=True):
                    cancel_planned_move()
                    st.rerun()

    with col_vision:
        render_vision_panel(result)

    with col_status:
        render_status_panel(st.session_state.game)
        st.markdown("---")
        render_robot_panel()
        st.markdown("---")
        render_game_log()

    # Live mode rerun
    if live_mode and is_real_mode:
        interval = float(st.session_state.get("live_interval", 2.0))
        time.sleep(interval)
        st.rerun()


if __name__ == "__main__":
    main()
