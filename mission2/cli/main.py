"""
CLI for Connect4 game and robot inference.

Usage:
    uv run python -m mission2.cli.main --help
    uv run python -m mission2.cli.main play --mock
    uv run python -m mission2.cli.main play --policy jlamperez/smolvla_connect4
    uv run python -m mission2.cli.main inference jlamperez/smolvla_connect4 task1
    uv run python -m mission2.cli.main inference jlamperez/smolvla_connect4 --interactive
"""

import json
import os
import re
import shlex
import time
from pathlib import Path
from typing import Annotated

import typer

from ..ai.minimax import MinimaxAI
from ..core.types import GamePhase, Move, Player
from ..game.engine import GameEngine
from ..robot.interface import RobotInterface
from ..robot.lerobot_record import (
    DEFAULT_EVAL_DATASET_NAME,
    DEFAULT_EVAL_EPISODE_TIME_S,
    DEFAULT_EVAL_RESET_TIME_S,
    DEFAULT_FOLLOWER_ID,
    DEFAULT_FOLLOWER_PORT,
    build_lerobot_record_command,
    load_dotenv,
    run_lerobot_record,
)
from ..robot.mock import MockRobot


app = typer.Typer(
    name="connect4",
    help="Connect4 game CLI with SmolVLA robot support.",
    add_completion=False,
)


def board_to_ascii(game_state) -> str:
    """Convert board to ASCII display."""
    lines = []
    lines.append("\n  0   1   2   3   4")
    lines.append("+" + "---+" * 5)

    symbols = {
        Player.ORANGE: "🟠",
        Player.YELLOW: "🟡",
        Player.EMPTY: "  ",
    }

    for i, row in enumerate(game_state.board.grid):
        cells = [symbols[cell] for cell in row]
        lines.append(f"|{cells[0]} |{cells[1]} |{cells[2]} |{cells[3]} |{cells[4]} |")
        lines.append("+" + "---+" * 5)

    return "\n".join(lines)


def print_status(game_state, last_move: int | None = None):
    """Print game status."""
    print(board_to_ascii(game_state))
    print(f"\nTurn: {game_state.turn_number}")

    if last_move is not None:
        print(f"Last move: Column {last_move}")

    if game_state.phase == GamePhase.GAME_OVER:
        if game_state.winner:
            winner_name = "Robot (🟠)" if game_state.winner == Player.ORANGE else "Human (🟡)"
            print(f"\n🎉 {winner_name} WINS! 🎉")
        else:
            print("\n🤝 It's a DRAW!")
    else:
        current = "Robot (🟠)" if game_state.current_player == Player.ORANGE else "Human (🟡)"
        print(f"Current player: {current}")
        print(f"Legal moves: {game_state.legal_moves}")


def get_robot(mock: bool, policy: str | None) -> RobotInterface:
    """Get robot instance based on mode."""
    if mock or policy is None:
        return MockRobot(move_delay=0.1)

    try:
        from ..robot.smolvla import SmolVLARobot

        return SmolVLARobot(policy_path=policy)
    except Exception as e:
        typer.echo(f"Failed to initialize SmolVLARobot ({e}). Falling back to mock.", err=True)
        return MockRobot(move_delay=0.1)


@app.command()
def play(
    mock: Annotated[bool, typer.Option("--mock", help="Use mock robot (no real execution)")] = False,
    policy: Annotated[str | None, typer.Option("--policy", "-p", help="SmolVLA policy path or HF repo")] = None,
    depth: Annotated[int, typer.Option("--depth", "-d", help="AI search depth")] = 5,
    human_first: Annotated[bool, typer.Option("--human-first", help="Human plays first")] = False,
    ai_vs_ai: Annotated[bool, typer.Option("--ai-vs-ai", help="Watch AI vs AI")] = False,
):
    """
    Play Connect4 game (Human vs AI or AI vs AI).

    Examples:
        play --mock                    # Human vs AI with mock robot
        play --policy jlamperez/model  # Human vs AI with real robot
        play --ai-vs-ai                # Watch two AIs play
    """
    typer.echo("\n" + "=" * 50)
    typer.echo("  CONNECT 4")
    typer.echo("=" * 50)

    engine = GameEngine()
    ai = MinimaxAI(depth=depth, player=Player.ORANGE)
    robot = get_robot(mock, policy)
    robot.connect()

    typer.echo(f"\nAI: {ai.get_name()}")
    typer.echo(f"Robot: {'Mock' if mock else policy or 'Mock'}")

    if ai_vs_ai:
        _play_ai_vs_ai(engine, ai, robot)
    else:
        _play_human_vs_ai(engine, ai, robot, human_first)

    robot.disconnect()


def _play_human_vs_ai(engine: GameEngine, ai: MinimaxAI, robot: RobotInterface, human_first: bool):
    """Human vs AI game loop."""
    first = Player.YELLOW if human_first else Player.ORANGE
    state = engine.new_game(first_player=first)

    typer.echo(f"First player: {'Human' if human_first else 'Robot'}")
    typer.echo("\nEnter column number (0-4) to play, 'q' to quit\n")

    last_move = None

    while not engine.is_game_over:
        print_status(state, last_move)

        if engine.is_robot_turn:
            typer.echo("\n🤖 Robot is thinking...")
            move, explanation = ai.get_move_with_explanation(state)
            typer.echo(f"AI chose: Column {move} - {explanation}")

            robot_move = Move(column=move, player=Player.ORANGE)
            robot.execute_move(robot_move)
            typer.echo(f"SmolVLA: {robot.get_last_instruction()}")

            state = engine.make_move(move, Player.ORANGE)
            last_move = move
        else:
            while True:
                try:
                    user_input = typer.prompt("\n👤 Your move (0-4)")
                    if user_input.lower() == "q":
                        typer.echo("Game quit.")
                        return

                    col = int(user_input)
                    if col not in state.legal_moves:
                        typer.echo(f"Invalid! Legal moves: {state.legal_moves}")
                        continue

                    state = engine.make_move(col, Player.YELLOW)
                    last_move = col
                    break
                except ValueError:
                    typer.echo("Enter a number 0-4")
                except KeyboardInterrupt:
                    typer.echo("\nGame quit.")
                    return

    print_status(state, last_move)


def _play_ai_vs_ai(engine: GameEngine, ai_orange: MinimaxAI, robot: RobotInterface, delay: float = 0.5):
    """AI vs AI game loop."""
    ai_yellow = MinimaxAI(depth=3, player=Player.YELLOW)

    typer.echo(f"🟠 Orange: {ai_orange.get_name()}")
    typer.echo(f"🟡 Yellow: {ai_yellow.get_name()}")
    typer.echo("\nPress Ctrl+C to stop\n")

    state = engine.new_game(first_player=Player.ORANGE)
    last_move = None

    try:
        while not engine.is_game_over:
            print_status(state, last_move)

            if state.current_player == Player.ORANGE:
                ai = ai_orange
                typer.echo("\n🟠 Orange thinking...")
            else:
                ai = ai_yellow
                typer.echo("\n🟡 Yellow thinking...")

            move, explanation = ai.get_move_with_explanation(state)
            typer.echo(f"Move: Column {move} - {explanation}")

            if state.current_player == Player.ORANGE:
                robot_move = Move(column=move, player=Player.ORANGE)
                robot.execute_move(robot_move)
                typer.echo(f"SmolVLA: {robot.get_last_instruction()}")

            state = engine.make_move(move, state.current_player)
            last_move = move
            time.sleep(delay)

    except KeyboardInterrupt:
        typer.echo("\nStopped.")

    print_status(state, last_move)


def load_tasks_json() -> dict[str, str]:
    """Backward-compatible wrapper around `load_tasks()`."""
    load_dotenv(_project_root() / ".env", override=False)
    return load_tasks(_resolve_tasks_json(None))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_tasks_json(tasks_json: Path | None) -> Path:
    if tasks_json is not None:
        return tasks_json

    env_value = os.environ.get("TASKS_JSON", "").strip()
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = _project_root() / candidate
        return candidate

    return Path(__file__).resolve().parents[1] / "tasks_smolvla.json"


def load_tasks(tasks_json: Path) -> dict[str, str]:
    """Load tasks from JSON. Returns {task_id: prompt} preserving file order."""
    if not tasks_json.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_json}")

    with open(tasks_json, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tasks" in data:
        data = data["tasks"]

    tasks: list[tuple[str, str]] = []

    def add(task_id: object, prompt: object) -> None:
        if task_id is None or prompt is None:
            return
        task_id_str = str(task_id).strip()
        prompt_str = str(prompt).strip()
        if not task_id_str or not prompt_str:
            return
        tasks.append((task_id_str, prompt_str))

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                add(key, value.get("prompt") or value.get("instruction") or value.get("task"))
            else:
                add(key, value)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, dict):
                task_id = item.get("id") or item.get("name") or item.get("task_id") or f"task{i+1}"
                add(task_id, item.get("prompt") or item.get("instruction") or item.get("task"))
            elif isinstance(item, str):
                add(f"task{i+1}", item)
            else:
                raise ValueError(
                    f"Unsupported task entry at index {i}: {type(item).__name__}"
                )
    else:
        raise ValueError(f"Unsupported tasks schema: {type(data).__name__}")

    if not tasks:
        raise ValueError(f"No tasks found in {tasks_json}")

    return dict(tasks)


def _parse_task_spec(spec: str, *, default_episodes: int) -> tuple[str, int]:
    raw = (spec or "").strip()
    if not raw:
        raise ValueError("Empty task spec")

    if ":" in raw:
        task_id, episodes_raw = raw.split(":", 1)
        task_id = task_id.strip()
        try:
            episodes = int(episodes_raw.strip())
        except ValueError as e:
            raise ValueError(f"Invalid episodes in '{spec}' (expected task:episodes)") from e
    else:
        task_id = raw
        episodes = int(default_episodes)

    if not task_id:
        raise ValueError(f"Invalid task id in '{spec}'")
    if episodes <= 0:
        raise ValueError(f"Episodes must be positive (got {episodes})")
    return task_id, episodes


def _format_tasks(tasks: dict[str, str]) -> str:
    lines = []
    for task_id, prompt in tasks.items():
        lines.append(f"- {task_id}: {prompt}")
    return "\n".join(lines)


def _env_number(keys: list[str], default: float, *, cast: type[int] | type[float]) -> float:
    for key in keys:
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        try:
            return cast(raw)  # type: ignore[call-arg]
        except ValueError:
            continue
    return default


def _normalize_inference_backend(value: str) -> str:
    raw = (value or "").strip().lower()
    if raw in {"controller", "inproc", "in-process", "in_process"}:
        return "controller"
    if raw in {"subprocess", "sub-process", "sub_process", "lerobot-record", "lerobot_record", "lerobot"}:
        return "lerobot-record"
    raise ValueError(f"Unknown backend '{value}'. Use 'controller' or 'lerobot-record'.")


def _resolve_eval_dataset_root(*, dataset_root: Path | None, resume: bool) -> Path:
    """Match `scripts/run_inference_vla.sh` dataset-root behavior."""
    if dataset_root is not None:
        root = dataset_root.expanduser()
        if resume:
            if not root.exists():
                raise FileNotFoundError(f"--resume specified but dataset root not found: {root}")
            return root

        if not root.exists():
            return root

        i = 1
        while Path(f"{root}_v{i}").exists():
            i += 1
        return Path(f"{root}_v{i}")

    base = Path(os.environ.get("EVAL_DATASET_ROOT_BASE", str(Path.home() / "so101_datasets"))).expanduser()
    name = os.environ.get("EVAL_DATASET_NAME", DEFAULT_EVAL_DATASET_NAME)
    root = base / name

    if resume:
        if root.exists():
            return root

        best: Path | None = None
        best_v = -1
        pattern = re.compile(rf"^{re.escape(name)}_v(\d+)$")
        for candidate in base.glob(f"{name}_v*"):
            match = pattern.match(candidate.name)
            if match is None:
                continue
            v = int(match.group(1))
            if v > best_v:
                best_v = v
                best = candidate

        if best is None:
            raise FileNotFoundError(
                f"--resume specified but no existing eval dataset found under {base} for name '{name}'"
            )
        return best

    if not root.exists():
        return root

    i = 1
    while (base / f"{name}_v{i}").exists():
        i += 1
    return base / f"{name}_v{i}"


def _run_task_plan_lerobot_record(
    *,
    tasks: dict[str, str],
    plan: list[tuple[str, int]],
    policy: str,
    device: str,
    follower_port: str | None,
    follower_id: str | None,
    dataset_root: Path,
    dataset_repo_id: str | None,
    dataset_fps: int | None,
    episode_time_s: float,
    reset_time_s: float,
    display_data: bool,
    play_sounds: bool,
    resume: bool,
    mock: bool,
) -> bool:
    for task_id, episodes in plan:
        prompt = tasks[task_id]
        typer.echo(f"\n=== {task_id} ===")
        typer.echo(f"Prompt: {prompt}")
        typer.echo(f"Episodes: {episodes}")

        cmd, _ = build_lerobot_record_command(
            policy_path=policy,
            prompt=prompt,
            episodes=episodes,
            resume=resume,
            device=device,
            follower_port=follower_port,
            follower_id=follower_id,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
            dataset_fps=dataset_fps,
            episode_time_s=int(episode_time_s),
            reset_time_s=int(reset_time_s),
            display_data=display_data,
            play_sounds=play_sounds,
        )

        if mock:
            typer.echo(f"[Mock] {shlex.join(cmd)}")
        else:
            ok, _, error = run_lerobot_record(
                policy_path=policy,
                prompt=prompt,
                episodes=episodes,
                resume=resume,
                device=device,
                follower_port=follower_port,
                follower_id=follower_id,
                dataset_root=dataset_root,
                dataset_repo_id=dataset_repo_id,
                dataset_fps=dataset_fps,
                episode_time_s=int(episode_time_s),
                reset_time_s=int(reset_time_s),
                display_data=display_data,
                play_sounds=play_sounds,
            )
            if not ok:
                typer.echo(f"Error: {error}", err=True)
                raise typer.Exit(1)

        resume = True

    return resume


def _run_task_plan(
    *,
    tasks: dict[str, str],
    plan: list[tuple[str, int]],
    controller,
    episode_time_s: float,
    reset_time_s: float,
    mock: bool,
) -> None:
    for task_id, episodes in plan:
        prompt = tasks[task_id]
        typer.echo(f"\n=== {task_id} ===")
        typer.echo(f"Prompt: {prompt}")
        typer.echo(f"Episodes: {episodes}")

        for ep in range(1, episodes + 1):
            typer.echo(f"\n[{task_id}] Episode {ep}/{episodes}")
            if mock:
                typer.echo(f"[Mock] Would run for ~{episode_time_s:.1f}s: {prompt}")
                time.sleep(min(0.25, float(episode_time_s)))
            else:
                controller.run_prompt(prompt, duration_s=float(episode_time_s))

            if reset_time_s > 0 and ep != episodes:
                typer.echo(f"[{task_id}] Waiting {reset_time_s:.1f}s before next episode...")
                time.sleep(float(reset_time_s))


@app.command()
def inference(
    policy: Annotated[str, typer.Argument(help="SmolVLA policy path or HuggingFace repo")],
    tasks: Annotated[
        list[str],
        typer.Argument(
            help="Task IDs to run (e.g., task1 task2:3). Use 'all' to run every task.",
        ),
    ] = [],
    tasks_json: Annotated[
        Path | None,
        typer.Option(
            "--tasks-json",
            help="Path to tasks JSON (defaults to TASKS_JSON or mission2/tasks_smolvla.json)",
        ),
    ] = None,
    backend: Annotated[
        str,
        typer.Option(
            "--backend",
            help="Inference backend: 'controller' (in-process) or 'lerobot-record' (subprocess).",
        ),
    ] = "controller",
    episodes: Annotated[
        int | None,
        typer.Option(
            "--episodes",
            "-e",
            help="Default episodes per task (lerobot-record defaults to 1; otherwise EPISODES_PER_TASK/EVAL_EPISODES_PER_TASK/EVAL_NUM_EPISODES).",
        ),
    ] = None,
    episode_time_s: Annotated[
        float | None,
        typer.Option(
            "--episode-time-s",
            help="Seconds to run each task prompt (defaults to EVAL_EPISODE_TIME, like scripts/run_inference_vla.sh).",
        ),
    ] = None,
    reset_time_s: Annotated[
        float | None,
        typer.Option(
            "--reset-time-s",
            help="Seconds to wait between episodes (lerobot-record defaults to 0; otherwise EVAL_RESET_TIME).",
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            help="Keep listening for task IDs after the plan completes",
        ),
    ] = False,
    mock: Annotated[bool, typer.Option("--mock", help="Mock mode (show command, don't execute)")] = False,
    device: Annotated[
        str, typer.Option("--device", help="Compute device for inference (e.g., cuda, cpu)")
    ] = "cuda",
    dataset_root: Annotated[
        Path | None,
        typer.Option(
            "--dataset-root",
            help="(lerobot-record backend) Dataset root directory (defaults to EVAL_DATASET_ROOT_BASE/EVAL_DATASET_NAME with versioning).",
        ),
    ] = None,
    resume_dataset: Annotated[
        bool,
        typer.Option(
            "--resume",
            help="(lerobot-record backend) Resume existing eval dataset root (like scripts/run_inference_vla.sh --resume).",
        ),
    ] = False,
    dataset_repo_id: Annotated[
        str | None,
        typer.Option(
            "--dataset-repo-id",
            help="(lerobot-record backend) Override dataset.repo_id (defaults to EVAL_DATASET_REPO_ID).",
        ),
    ] = None,
    display_data: Annotated[
        bool,
        typer.Option(
            "--display-data/--no-display-data",
            help="(lerobot-record backend) Display camera feed windows.",
        ),
    ] = False,
    dataset_fps: Annotated[
        int | None,
        typer.Option(
            "--fps",
            help="(lerobot-record backend) Dataset/control FPS (camera capture stays at CAMERA_FPS).",
        ),
    ] = 15,
    play_sounds: Annotated[
        bool,
        typer.Option(
            "--play-sounds/--no-play-sounds",
            help="(lerobot-record backend) Enable sounds.",
        ),
    ] = False,
    follower_port: Annotated[
        str | None,
        typer.Option("--port", help="Robot serial port (defaults to FOLLOWER_PORT)"),
    ] = None,
    follower_id: Annotated[
        str | None,
        typer.Option("--robot-id", help="Robot id (defaults to FOLLOWER_ID)"),
    ] = None,
):
    """
    Run SmolVLA inference for one or more tasks.

    By default this uses the in-process controller (keeps the model loaded).
    For app-style one-shot runs, use `--backend lerobot-record` (defaults to 1 episode per task).

    Examples:
        inference jlamperez/smolvla_connect4 task1
        inference jlamperez/smolvla_connect4 task1 task2:3
        inference jlamperez/smolvla_connect4 all --episodes 2
        inference jlamperez/smolvla_connect4 --interactive
    """
    typer.echo("\n" + "=" * 50)
    typer.echo("  SmolVLA Inference")
    typer.echo("=" * 50)

    load_dotenv(_project_root() / ".env", override=False)
    tasks_path = _resolve_tasks_json(tasks_json)

    try:
        task_map = load_tasks(tasks_path)
    except Exception as e:
        typer.echo(f"Error loading tasks JSON: {e}", err=True)
        raise typer.Exit(1) from e

    try:
        backend_name = _normalize_inference_backend(backend)
    except ValueError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(2) from e

    default_episodes = int(
        episodes
        if episodes is not None
        else (
            1
            if backend_name == "lerobot-record"
            else _env_number(
                ["EPISODES_PER_TASK", "EVAL_EPISODES_PER_TASK", "EVAL_NUM_EPISODES"],
                1,
                cast=int,
            )
        )
    )
    default_episode_time_s = float(
        episode_time_s
        if episode_time_s is not None
        else _env_number(["EVAL_EPISODE_TIME"], float(DEFAULT_EVAL_EPISODE_TIME_S), cast=float)
    )
    default_reset_time_s = float(
        reset_time_s
        if reset_time_s is not None
        else _env_number(["EVAL_RESET_TIME"], float(DEFAULT_EVAL_RESET_TIME_S), cast=float)
    )

    if default_episodes <= 0:
        typer.echo(f"Error: --episodes must be positive (got {default_episodes})", err=True)
        raise typer.Exit(2)
    if default_episode_time_s <= 0:
        typer.echo(f"Error: --episode-time-s must be positive (got {default_episode_time_s})", err=True)
        raise typer.Exit(2)
    if default_reset_time_s < 0:
        typer.echo(f"Error: --reset-time-s cannot be negative (got {default_reset_time_s})", err=True)
        raise typer.Exit(2)

    typer.echo(f"\nPolicy: {policy}")
    typer.echo(f"Tasks JSON: {tasks_path}")
    typer.echo(f"Mode: {'Mock' if mock else 'Real'}")
    typer.echo(f"Backend: {backend_name}")
    typer.echo(f"Device: {device}")
    typer.echo(f"Default episodes per task: {default_episodes}")
    typer.echo(f"Episode time: {default_episode_time_s}")
    typer.echo(f"Reset time: {default_reset_time_s}")

    port = follower_port or os.environ.get("FOLLOWER_PORT", DEFAULT_FOLLOWER_PORT)
    robot_id = follower_id or os.environ.get("FOLLOWER_ID", DEFAULT_FOLLOWER_ID)
    typer.echo(f"Robot port: {port}")
    typer.echo(f"Robot id: {robot_id}")
    typer.echo("")

    # Build initial plan.
    plan: list[tuple[str, int]] = []
    if tasks:
        for spec in tasks:
            if spec.strip().lower() == "all":
                plan.extend((task_id, int(default_episodes)) for task_id in task_map.keys())
                continue
            try:
                task_id, task_episodes = _parse_task_spec(spec, default_episodes=default_episodes)
            except ValueError as e:
                typer.echo(f"Error: {e}", err=True)
                raise typer.Exit(2) from e
            if task_id not in task_map:
                typer.echo(
                    f"Error: Unknown task '{task_id}'. Available:\n{_format_tasks(task_map)}",
                    err=True,
                )
                raise typer.Exit(2)
            plan.append((task_id, task_episodes))
    else:
        if not interactive:
            plan.extend((task_id, int(default_episodes)) for task_id in task_map.keys())

    if not mock:
        if port and not os.path.exists(port):
            typer.echo(f"Error: Robot port not found: {port}", err=True)
            raise typer.Exit(2)

    controller = None
    eval_dataset_root: Path | None = None
    eval_resume = False

    if backend_name == "controller":
        if not mock:
            try:
                from ..robot.smolvla_controller import (
                    SmolVLAController,
                    SmolVLAControllerConfig,
                    SmolVLAControllerError,
                )

                controller = SmolVLAController(
                    SmolVLAControllerConfig(
                        policy_path=policy,
                        device=device,
                        follower_port=port,
                        follower_id=robot_id,
                        episode_time_s=float(default_episode_time_s),
                        fps=5.0,
                    )
                )
                controller.connect()
            except SmolVLAControllerError as e:
                typer.echo(f"Error: failed to connect/load policy: {e}", err=True)
                raise typer.Exit(1) from e
            except Exception as e:
                typer.echo(f"Error: failed to connect/load policy: {e}", err=True)
                raise typer.Exit(1) from e
    else:
        try:
            eval_dataset_root = _resolve_eval_dataset_root(dataset_root=dataset_root, resume=bool(resume_dataset))
        except FileNotFoundError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(2) from e
        eval_resume = (eval_dataset_root / "meta/info.json").is_file()
        typer.echo(f"Eval dataset root: {eval_dataset_root}")

    try:
        if plan:
            if backend_name == "controller":
                _run_task_plan(
                    tasks=task_map,
                    plan=plan,
                    controller=controller,
                    episode_time_s=float(default_episode_time_s),
                    reset_time_s=float(default_reset_time_s),
                    mock=mock,
                )
            else:
                if eval_dataset_root is None:
                    raise typer.Exit(1)
                eval_resume = _run_task_plan_lerobot_record(
                    tasks=task_map,
                    plan=plan,
                    policy=policy,
                    device=device,
                    follower_port=port,
                    follower_id=robot_id,
                    dataset_root=eval_dataset_root,
                    dataset_repo_id=dataset_repo_id,
                    dataset_fps=dataset_fps,
                    episode_time_s=float(default_episode_time_s),
                    reset_time_s=float(default_reset_time_s),
                    display_data=bool(display_data),
                    play_sounds=bool(play_sounds),
                    resume=bool(eval_resume),
                    mock=mock,
                )

        if interactive:
            typer.echo("\nInteractive mode. Enter:")
            typer.echo("- a task id (e.g., task1) or task:episodes (e.g., task2:3)")
            typer.echo("- 'all' to run all tasks")
            typer.echo("- 'list' to show tasks")
            typer.echo("- 'quit' to exit\n")

            while True:
                try:
                    line = input("task> ").strip()
                except (EOFError, KeyboardInterrupt):
                    typer.echo("\nExiting.")
                    break

                if not line:
                    continue

                cmd = line.lower()
                if cmd in {"q", "quit", "exit"}:
                    break
                if cmd == "help":
                    typer.echo("Enter task ids like: task1, task2:3, all, list, quit")
                    continue
                if cmd == "list":
                    typer.echo(_format_tasks(task_map))
                    continue

                specs = line.split()
                next_plan: list[tuple[str, int]] = []
                for spec in specs:
                    if spec.strip().lower() == "all":
                        next_plan.extend((task_id, int(default_episodes)) for task_id in task_map.keys())
                        continue

                    try:
                        task_id, task_episodes = _parse_task_spec(spec, default_episodes=default_episodes)
                    except ValueError as e:
                        typer.echo(f"Error: {e}", err=True)
                        next_plan = []
                        break

                    if task_id not in task_map:
                        typer.echo(
                            f"Error: Unknown task '{task_id}'. Type 'list' to see options.",
                            err=True,
                        )
                        next_plan = []
                        break

                    next_plan.append((task_id, task_episodes))

                if not next_plan:
                    continue

                if backend_name == "controller":
                    _run_task_plan(
                        tasks=task_map,
                        plan=next_plan,
                        controller=controller,
                        episode_time_s=float(default_episode_time_s),
                        reset_time_s=float(default_reset_time_s),
                        mock=mock,
                    )
                else:
                    if eval_dataset_root is None:
                        raise typer.Exit(1)
                    eval_resume = _run_task_plan_lerobot_record(
                        tasks=task_map,
                        plan=next_plan,
                        policy=policy,
                        device=device,
                        follower_port=port,
                        follower_id=robot_id,
                        dataset_root=eval_dataset_root,
                        dataset_repo_id=dataset_repo_id,
                        dataset_fps=dataset_fps,
                        episode_time_s=float(default_episode_time_s),
                        reset_time_s=float(default_reset_time_s),
                        display_data=bool(display_data),
                        play_sounds=bool(play_sounds),
                        resume=bool(eval_resume),
                        mock=mock,
                    )
    finally:
        if controller is not None:
            try:
                controller.disconnect()
            except Exception:
                pass

    typer.echo("\n" + "=" * 50)
    typer.echo("Inference complete!")
    typer.echo("=" * 50)


@app.command()
def test():
    """Run quick test of game logic and robot mock."""
    from ..game.rules import Connect4Rules

    typer.echo("\n" + "=" * 50)
    typer.echo("  TESTING GAME LOGIC")
    typer.echo("=" * 50)

    # Test rules
    rules = Connect4Rules()
    typer.echo("\n✓ Rules initialized")

    # Test engine
    engine = GameEngine(rules=rules)
    state = engine.new_game()
    typer.echo(f"✓ New game started, turn {state.turn_number}")
    typer.echo(f"  Legal moves: {state.legal_moves}")

    # Test moves
    state = engine.make_move(2, Player.ORANGE)
    typer.echo("✓ Orange played column 2")

    state = engine.make_move(2, Player.YELLOW)
    typer.echo("✓ Yellow played column 2")

    # Test AI
    ai = MinimaxAI(depth=3)
    move = ai.get_move(state)
    typer.echo(f"✓ AI suggests column {move}")

    # Test robot mock
    robot = MockRobot(move_delay=0.01)
    robot.connect()
    typer.echo(f"✓ Robot connected: {robot.is_connected()}")

    action = robot.execute_move(Move(column=move, player=Player.ORANGE))
    typer.echo(f"✓ Robot executed move: {action.status.name}")
    typer.echo(f"  SmolVLA prompt: {robot.get_last_instruction()}")

    robot.disconnect()
    typer.echo("✓ Robot disconnected")

    typer.echo("\n✅ All tests passed!")


@app.command()
def list_tasks(
    tasks_json: Annotated[Path | None, typer.Option("--tasks-json", help="Path to tasks JSON")] = None,
):
    """List available tasks from tasks_smolvla.json."""
    load_dotenv(_project_root() / ".env", override=False)
    tasks_path = _resolve_tasks_json(tasks_json)
    try:
        task_map = load_tasks(tasks_path)
    except Exception as e:
        typer.echo(f"Error loading tasks JSON: {e}", err=True)
        raise typer.Exit(1) from e

    typer.echo(f"\nTasks from: {tasks_path}\n")
    typer.echo(f"{'ID':<10} {'Prompt'}")
    typer.echo("-" * 60)

    for task_id, prompt in task_map.items():
        typer.echo(f"{task_id:<10} {prompt}")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
