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

## Plot Reward-Shaping Comparison

Export aligned learning curves for the baseline and heuristic shaped-reward
Pong runs documented in `note.txt`:

```bash
python -m atari_agent.plot_pong_reward_shaping
```

The plot is saved to
`visualizations/pong_baseline_vs_shaped_reward.png`. The baseline CSV records
the original Pong episode score, while the shaped-reward CSV records the
heuristic training return, so the plot uses separate panels for the two reward
scales.

Evaluate the baseline and shaped-reward policies at 100K, 300K, 600K, 1.2M,
and 1.8M checkpoints on the original Pong score:

```bash
python -m atari_agent.evaluate_pong_checkpoints
```

This selects the resumed training path in `note.txt` ending with the
`buffer 20_000` continuation, evaluates both agents with identical episode
seeds, and writes CSV results under `evaluations/pong_mixed_replay_buffer/`.
Add `--plot` to also export a comparison plot when Matplotlib is installed,
or use `--dry-run` to inspect the resolved checkpoint files without starting
Atari evaluation.

Create the evaluation plot later from the exported summary CSV, without
rerunning the environments:

```bash
python -m atari_agent.plot_pong_evaluation
```

The output is saved to
`evaluations/pong_mixed_replay_buffer/evaluation_comparison.png`.

## Plot Breakout Reward Shaping

Compare the frame-skip-4 Breakout baseline against the heuristic shaped-reward
training run:

```bash
python -m atari_agent.plot_breakout_reward_shaping
```

The plot is saved to
`visualizations/breakout_baseline_vs_shaped_reward.png`. The comparable panel
uses raw Breakout episode scores for both agents; a second panel shows the
heuristic agent's shaped training return on its separate reward scale.

Evaluate both Breakout agents on raw game score at every saved 100K checkpoint:

```bash
python -m atari_agent.evaluate_breakout_checkpoints
python -m atari_agent.plot_breakout_evaluation
```

The evaluator selects the latest frame-skip-4 baseline and shaped-reward run,
uses identical episode seeds, and writes CSV results under
`evaluations/breakout_checkpoint_comparison/`. The plotting command exports
`evaluation_comparison.png` in the same directory. Evaluation results are
saved after each checkpoint because full Breakout episodes can take time.

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
