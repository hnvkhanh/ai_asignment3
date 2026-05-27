# DQN Experiments for Pong, Breakout, and CartPole

This project implements Deep Q-Network (DQN) experiments in Gymnasium for
Atari Pong, Atari Breakout, and CartPole. It includes training scripts,
checkpoint evaluation scripts, and plots for reward shaping, replay-buffer
size, frame skip, and epsilon-greedy exploration schedules.

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Pong and Breakout require Atari ROMs for ALE environments. Install or download
ROMs only when you have the legal right to use them. If they are not available
after dependency installation, run:

```bash
AutoROM --accept-license
```

## Main Scripts

| Script | Purpose |
| --- | --- |
| `atari_agent.train_dqn` | Main Pong-capable Atari DQN trainer; uses heuristic Pong reward shaping by default. |
| `atari_agent.train_dqn_original` | Baseline Atari DQN trainer with clipped environment rewards and no game-specific shaping. |
| `atari_agent.train_breakout_experiments` | Trains Breakout baselines for frame skips `2`, `4`, and `8` by default. |
| `atari_agent.train_breakout_shaped` | Trains Breakout with heuristic reward shaping. |
| `atari_agent.train_cartpole_epsilon_experiments` | Trains and evaluates CartPole DQN agents under four exploration schedules. |
| `atari_agent.evaluate_pong_checkpoints` | Evaluates the documented Pong baseline and shaped checkpoint paths on original score. |
| `atari_agent.evaluate_breakout_checkpoints` | Evaluates Breakout baseline and shaped checkpoints on raw game score. |

## Pong

Train the heuristic reward-shaped Pong agent:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --total-timesteps 100000
```

The command-line defaults for Pong shaping are:

- `+1` for a detected paddle hit
- `-2` for losing a point
- up to an additional `-1` miss-distance penalty
- `+4` for scoring a point

Train a baseline using clipped original rewards instead of Pong shaping:

```bash
python -m atari_agent.train_dqn --env-id ALE/Pong-v5 --no-pong-reward-shaping
```

Training writes `training_log.csv`, periodic `dqn_step_<step>.pt`
checkpoints, and `dqn_final.pt` under `runs/Pong-v5_<timestamp>/`.
`train_dqn.py` uses a `20_000` replay buffer by default; the retained
`train_dqn_frame.py` variant differs by using a `10_000` default buffer.

Generate Pong training plots from the run chains documented in `note.txt`:

```bash
python -m atari_agent.plot_pong_baseline --buffer-size 20000
python -m atari_agent.plot_pong_baseline --compare-buffers 10000 20000
python -m atari_agent.plot_pong_reward_shaping --buffer-size 20000
```

Evaluate the baseline and heuristic shaped-reward checkpoints at `100K`,
`300K`, `600K`, `1.2M`, and `1.8M` training steps on original Pong score:

```bash
python -m atari_agent.evaluate_pong_checkpoints --plot
```

By default, evaluation follows the `buffer 20_000` continuation recorded in
`note.txt`, uses 10 greedy episodes per checkpoint, and writes:

- `evaluations/pong_mixed_replay_buffer/evaluation_summary.csv`
- `evaluations/pong_mixed_replay_buffer/evaluation_episodes.csv`
- `evaluations/pong_mixed_replay_buffer/evaluation_comparison.png` with `--plot`

Use `--dry-run` to inspect which Pong checkpoint files will be selected.

## Breakout

Train the baseline frame-skip experiment:

```bash
python -m atari_agent.train_breakout_experiments
python -m atari_agent.plot_breakout_frame_skip
```

The experiment trains frame skips `2`, `4`, and `8` for `600_000` agent steps
each by default, with frame skip `4` treated as the baseline. It writes
checkpoints, logs, `frame_skip_summary.csv`, and an experiment notes file under
`runs_breakout/Breakout_frame_skip_<timestamp>/`.

Train the shaped-reward Breakout agent and compare it with the frame-skip-4
baseline:

```bash
python -m atari_agent.train_breakout_shaped
python -m atari_agent.plot_breakout_reward_shaping
```

Breakout reward shaping gives a configurable reward for brick hits and paddle
hits and applies a configurable miss-distance penalty after life loss. Its log
records both shaped return and raw game return under
`runs_breakout_shaped/Breakout-v5_shaped_<timestamp>/`.

Evaluate the latest frame-skip-4 baseline and shaped-reward run at each saved
`100K` checkpoint through `600K`, then plot the exported results:

```bash
python -m atari_agent.evaluate_breakout_checkpoints
python -m atari_agent.plot_breakout_evaluation
```

Evaluation uses 10 greedy episodes per checkpoint and raw Breakout score by
default. Results are written to:

- `evaluations/breakout_checkpoint_comparison/evaluation_summary.csv`
- `evaluations/breakout_checkpoint_comparison/evaluation_episodes.csv`
- `evaluations/breakout_checkpoint_comparison/evaluation_comparison.png`

## CartPole

Train and evaluate the epsilon-greedy schedule experiment, then plot it:

```bash
python -m atari_agent.train_cartpole_epsilon_experiments
python -m atari_agent.plot_cartpole_epsilon_schedules
```

The experiment runs three seeds for four schedules by default: fast decay from
`1.0` to `0.05`, slow decay from `1.0` to `0.05`, fixed epsilon `0.1`, and
fixed epsilon `0.2`. Each model is trained for 300 episodes and then evaluated
for 10 greedy episodes by `evaluate_model()` in the same training script.
There is no separate CartPole evaluator module.

CartPole outputs are written under
`runs_cartpole/CartPole_epsilon_<timestamp>/`, including:

- per-seed training logs and `cartpole_dqn_final.pt` checkpoints
- `epsilon_schedule_summary.csv`
- `evaluation_episodes.csv`
- `experiment_notes.txt`

The plot command exports
`visualizations/cartpole_epsilon_schedule_comparison.png`.

## Single Checkpoint Evaluation

Evaluate one Pong or other `train_dqn.py` Atari checkpoint:

```bash
python -m atari_agent.train_dqn \
  --eval-only \
  --checkpoint runs/Pong-v5_<timestamp>/dqn_final.pt \
  --env-id ALE/Pong-v5
```

Evaluate one shaped Breakout checkpoint:

```bash
python -m atari_agent.train_breakout_shaped \
  --eval-only \
  --checkpoint runs_breakout_shaped/Breakout-v5_shaped_<timestamp>/breakout_shaped_final.pt
```

## Current Exported Results

The repository currently contains exported evaluation CSVs and figures for:

- Pong baseline versus reward-shaped checkpoint evaluation
- Pong baseline replay-buffer and reward-shaping learning curves
- Breakout baseline versus reward-shaped checkpoint evaluation and learning curves
- CartPole epsilon-schedule comparison

Training checkpoints and raw run directories (`runs/`, `runs_breakout/`,
`runs_breakout_shaped/`, and `runs_cartpole/`) are ignored by Git because they
can be large, although they may exist in a working copy used to create the
exported results.

## Implementation Notes

The Atari trainers use no-op reset, configurable frame skip, max-pooling,
grayscale `84 x 84` observations, and four-frame stacking by default. ALE
environments are created with `frameskip=1` because `AtariPreprocessing`
performs frame skipping.
