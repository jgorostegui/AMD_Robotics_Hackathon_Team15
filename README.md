# AMD Robotics Hackathon 2025 - Team 15

SO-101 robot arm imitation learning using LeRobot and AMD ROCm.

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
