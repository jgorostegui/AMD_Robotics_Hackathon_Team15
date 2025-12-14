# Mission 2: Connect 4 (Robotic Opponent)

Mission 2 implements a Connect 4-style game on a physical 5×5 board, combining:
- **Game + AI**: Rules + minimax move selection
- **Vision**: Board state detection with a calibration workflow
- **Robot**: Prompt-based pick-and-place via SmolVLA (optional)
- **UI**: Streamlit dashboard for operating and debugging the system

This page is written as a **runbook** first, with a submission-ready section at the end.

---

## Quickstart (No Hardware)

```bash
cd AMD_Robotics_Hackathon_Team15
uv sync
```

### Run the dashboard (mock robot + mock vision)

```bash
uv run streamlit run mission2/app/game_dashboard.py
```

In the sidebar:
- **Robot Mode**: `Mock`
- **Vision Mode**: `Mock`

### Run the CLI (optional)

```bash
uv run python -m mission2.cli.main play --mock
uv run python -m mission2.cli.main test
```

---

## Real Vision (HTTP)

The Streamlit apps use a simple HTTP camera server for reliability in UI workflows.

### 1) Start the camera server (on the camera machine)

```bash
uv run python scripts/stream_http.py --camera 0 --port 8080
```

### 2) Calibrate board detection (operator machine)

```bash
uv run streamlit run mission2/app/calibrate_connect4.py
```

Save calibration to `mission2/vision/calibration.json` (default).

### 3) Use vision in the game dashboard

```bash
uv run streamlit run mission2/app/game_dashboard.py
```

In the sidebar:
- **Vision Mode**: `Real (HTTP)`
- Use **Detect & Sync** (manual) or enable **Live Mode** (continuous sync)

---

## Real Robot (SmolVLA)

The dashboard and CLI can run the robot if you provide a policy and the robot is connected.

### Dashboard

```bash
uv run streamlit run mission2/app/game_dashboard.py
```

In the sidebar:
- **Robot Mode**: `SmolVLA`
- **Policy path / HF repo**: set your policy
- **Subprocess backend**:
  - `direct-inference`: one-shot runner
  - `lerobot-record`: records evaluation episodes

### CLI

```bash
uv run python -m mission2.cli.main play --policy <POLICY>
uv run python -m mission2.cli.main inference <POLICY> task1 task2:3
uv run python -m mission2.cli.main inference <POLICY> --interactive
```

---

## Configuration (practical)

The dashboard/CLI load `.env` from the repo root. Start from `.env.example`.

Common knobs:
- `VISION_CALIBRATION_PATH` (default: `mission2/vision/calibration.json`)
- `STREAM_HOST` (camera server host; default matches the UI defaults)
- `MISSION2_GRID_METHOD` (grid inference method: `sample_hsv|hybrid|hough`)
- `SMOLVLA_POLICY` (default policy shown in the dashboard)
- `FOLLOWER_PORT`, `FOLLOWER_ID` (robot serial + id)
- `TASKS_JSON` (task prompt file; default: `mission2/tasks_smolvla.json`)

---

## Key Files

- UI dashboard: `mission2/app/game_dashboard.py`
- Vision calibration UI: `mission2/app/calibrate_connect4.py`
- Game engine (incl. vision reconciliation): `mission2/game/engine.py`
- AI: `mission2/ai/minimax.py`
- Robot:
  - Subprocess-based: `mission2/robot/smolvla.py`
  - Persistent in-process controller: `mission2/robot/smolvla_controller.py`
- Vision detection core: `mission2/vision/board_detector.py`
- HTTP camera server: `scripts/stream_http.py`

---

## Submission Details (copy/paste)

### 1) Mission 2 Description (10 points)

We built an interactive robotic Connect 4 opponent on a physical 5×5 board. The system observes the board state via vision, selects the next move with a minimax-based AI, and (optionally) executes the move on a SO-101 arm using a Vision-Language-Action policy driven by natural-language task prompts.

### 2) Creativity (30 points)

- **Prompt-based actuation**: moves are expressed as natural-language tasks (“place in column X”), keeping game logic independent from robot kinematics.
- **Operator-friendly tools**: a Streamlit control center and a dedicated calibration app make the demo repeatable in different setups.
- **Robust state handling**: the engine includes mechanisms to reconcile perceived board state with internal state during live play.

### 3) Technical Implementations (20 points)

#### Teleoperation / dataset capture
- Multi-task prompts are defined in `mission2/tasks_smolvla.json`.
- Inference can be run via `lerobot-record` to capture evaluation episodes.

#### Training
- Training workflow and notes live in `mission2/code/training-models-on-rocm-smolvla.ipynb`.

#### Inference
- Dashboard supports a SmolVLA robot mode.
- CLI supports an in-process controller backend (keeps the model loaded) and a `lerobot-record` backend.

### 4) Ease of Use (10 points)

- **Mock-first workflow**: dashboard and CLI run without hardware for rapid iteration.
- **One place to operate**: dashboard consolidates game state, AI decisions, robot control, and vision sync.
- **Calibration UI**: vision calibration is interactive and saved to a single JSON file.

---

## Troubleshooting

- **No camera frames**: start `scripts/stream_http.py` on the camera machine and verify `http://<host>:8080/health`.
- **Vision is unstable**: run `mission2/app/calibrate_connect4.py`, save calibration, and retry with `MISSION2_GRID_METHOD=sample_hsv` first.
- **Robot connect fails**: confirm `FOLLOWER_PORT` exists and the robot is powered/connected.
