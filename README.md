# AMD Robotics Hackathon 2025 - Team 15

SO-101 robot arm imitation learning using LeRobot and AMD ROCm.

## Mission 2: Connect 4

- Game dashboard (mock robot): `uv run streamlit run mission2/app/game_dashboard.py`
- Vision calibrator: `uv run streamlit run mission2/app/calibrate_connect4.py`
- CLI game (Human vs AI): `uv run python -m mission2.cli.main play --mock` (or `--policy <HF_REPO_OR_PATH>`)
- Run SmolVLA task prompts (keeps model loaded, no GUI): `uv run python -m mission2.cli.main inference <POLICY> task1 task2:3`
- Interactive task runner: `uv run python -m mission2.cli.main inference <POLICY> --interactive` (type `list`, `task1`, `all`, `quit`)

**Dataset**: [jlamperez/mission2_smolvla_multitask_v2_120ep](https://huggingface.co/datasets/jlamperez/mission2_smolvla_multitask_v2_120ep)

**Model**: [jlamperez/mission2_smolvla_multitask_policy_30ksteps_120ep](https://huggingface.co/jlamperez/mission2_smolvla_multitask_policy_30ksteps_120ep)

**Training Notebook**: `mission2/code/training-models-on-rocm-smolvla.ipynb`

## Mission 1: Pick & Place

Train an ACT (Action Chunking Transformer) policy to pick objects and place them at target locations.

**Dataset**: [jlamperez/mission1_pick_place](https://huggingface.co/datasets/jlamperez/mission1_pick_place)

**Model**: [jlamperez/act_mission1_pick_place](https://huggingface.co/jlamperez/act_mission1_pick_place)

**Training Notebook**: `mission1/code/training-models-on-rocm.ipynb`

## Tech Stack

- [LeRobot](https://github.com/huggingface/lerobot) v0.4.2+
- SO-101 robot arm (6-DOF + gripper)
- AMD Instinct MI300X (ROCm 6.3)
- [uv](https://github.com/astral-sh/uv) package manager

## License

MIT
