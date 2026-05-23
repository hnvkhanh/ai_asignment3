# Atari DQN Agent with Gymnasium

This project trains a Deep Q-Network agent on Atari environments through
Gymnasium and ALE.

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Atari ROMs are required for ALE environments. The dependency file includes
Gymnasium's ROM-license extra and AutoROM support; only install/download ROMs
when you have the legal right to use them.

If ROMs are still missing after installation, run:

```bash
AutoROM --accept-license
```

## Train

Start with Pong because it is a common sanity-check Atari task:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --total-timesteps 100000
```

Use CUDA explicitly when you have an NVIDIA GPU and a CUDA-enabled PyTorch
install:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --total-timesteps 100000 --device cuda
```

For newer NVIDIA GPUs, `--amp` can make the neural-network updates faster:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --total-timesteps 100000 --device cuda --amp
```

Pong training uses custom reward shaping by default:

- `+1` when the agent hits the ball
- `-2` when the agent misses the ball
- `+2` when the agent scores a point

You can adjust those values:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --hit-reward 1 --miss-penalty -2 --score-reward 2
```

To use the original clipped Atari rewards instead:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --no-pong-reward-shaping
```

Useful options:

```bash
python -m atari_agent.train_dqn \
  --env-id ALE/Breakout-v5 \
  --total-timesteps 1000000 \
  --device auto \
  --capture-video
```

Checkpoints and a CSV training log are written under `runs/`.

## Evaluate

Run a saved checkpoint:

```bash
python -m atari_agent.train_dqn \
  --eval-only \
  --checkpoint runs/Pong-v5_*/dqn_final.pt \
  --env-id ALE/Pong-v5 \
  --render-mode human
```

## Notes

The trainer uses the standard Atari preprocessing stack: no-op reset, frame
skip, max-pooling, grayscale 84x84 observations, and four-frame stacking. The
environment is created with `frameskip=1` because `AtariPreprocessing` performs
the frame skipping itself.
