# Training Guide (AMD MI300X)

## Environment Setup

```bash
# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install FFmpeg 7.x (required for video datasets)
sudo add-apt-repository ppa:ubuntuhandbook1/ffmpeg7
sudo apt update && sudo apt install ffmpeg

# Authenticate
huggingface-cli login
wandb login  # optional
```

## Training Commands

### Smoke Test (1k steps)

```bash
lerobot-train \
  --dataset.repo_id=jlamperez/mission1_pick_place \
  --policy.type=act \
  --steps=1000 \
  --batch_size=32 \
  --policy.device=cuda \
  --wandb.enable=true
```

### Full Training (50k steps)

```bash
lerobot-train \
  --dataset.repo_id=jlamperez/mission1_pick_place \
  --policy.type=act \
  --steps=50000 \
  --batch_size=32 \
  --output_dir=outputs/train/act_mission1_50k \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=jlamperez/act_mission1_pick_place \
  --wandb.enable=true \
  --wandb.project=amd-hackathon-2025
```

### Resume Training

```bash
lerobot-train \
  --resume=true \
  --config_path=outputs/train/act_mission1_50k/checkpoints/last/pretrained_model/train_config.json \
  --steps=100000
```

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--batch_size` | 32 | Reduce to 16/8 if OOM |
| `--steps` | 50000 | Start with 20k, increase if needed |
| `--policy.type` | act | ACT, diffusion, or smolvla |

## Upload Checkpoint

```bash
huggingface-cli upload jlamperez/act_mission1_pick_place \
  outputs/train/act_mission1_50k/checkpoints/last/pretrained_model
```
