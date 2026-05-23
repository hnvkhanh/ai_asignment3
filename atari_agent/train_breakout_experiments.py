from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from train_dqn_original import (
        DQN,
        ReplayBuffer,
        as_observation_array,
        linear_schedule,
        make_env,
        save_checkpoint,
        scalar,
    )
except ImportError:
    from .train_dqn_original import (
        DQN,
        ReplayBuffer,
        as_observation_array,
        linear_schedule,
        make_env,
        save_checkpoint,
        scalar,
    )


@dataclass
class ExperimentResult:
    frame_skip: int
    is_baseline: bool
    run_dir: Path
    final_average_score: float
    best_score: float
    training_stability_std: float
    episode_count: int
    final_step: int
    wall_clock_seconds: float
    agent_steps_per_second: float
    atari_frames_per_second: float
    final_checkpoint: Path


def pick_device(device_name: str, gpu_id: int | None = None) -> torch.device:
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError("--gpu-id was set, but CUDA is not available.")
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(f"--gpu-id must be between 0 and {gpu_count - 1}; got {gpu_id}.")
        return torch.device(f"cuda:{gpu_id}")

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_episode_metrics(
    episode_returns: list[float],
    final_average_episodes: int,
) -> tuple[float, float, float]:
    if not episode_returns:
        return np.nan, np.nan, np.nan

    returns = np.asarray(episode_returns, dtype=np.float32)
    window = max(1, final_average_episodes)
    final_returns = returns[-window:]
    return (
        float(np.mean(final_returns)),
        float(np.max(returns)),
        float(np.std(final_returns)),
    )


def train_frame_skip_experiment(
    args: argparse.Namespace,
    base_run_dir: Path,
    device: torch.device,
    frame_skip: int,
    experiment_index: int,
) -> ExperimentResult:
    experiment_args = argparse.Namespace(**vars(args))
    experiment_args.frame_skip = frame_skip
    if args.different_seeds:
        experiment_args.seed = args.seed + experiment_index

    run_dir = base_run_dir / f"frame_skip_{frame_skip}"
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(experiment_args.seed)
    np.random.seed(experiment_args.seed)
    torch.manual_seed(experiment_args.seed)

    env = make_env(experiment_args, run_dir)
    observation_shape = tuple(env.observation_space.shape)
    action_count = env.action_space.n

    policy_net = DQN(observation_shape, action_count).to(device)
    target_net = DQN(observation_shape, action_count).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, observation_shape, device)

    log_path = run_dir / "training_log.csv"
    log_file = log_path.open("w", newline="")
    logger = csv.DictWriter(
        log_file,
        fieldnames=[
            "step",
            "atari_frames",
            "episode",
            "return",
            "moving_average_return",
            "length",
            "epsilon",
            "loss",
            "agent_steps_per_second",
            "atari_frames_per_second",
        ],
    )
    logger.writeheader()

    observation, _ = env.reset(seed=experiment_args.seed)
    observation = as_observation_array(observation)
    episode_count = 0
    episode_returns: list[float] = []
    last_loss = np.nan
    global_step = 0
    start_time = time.time()
    epsilon_duration = int(args.exploration_fraction * args.total_timesteps)

    label = "baseline" if frame_skip == args.baseline_frame_skip else "variant"
    print(
        f"\nTraining Breakout {label}: frame_skip={frame_skip}, "
        f"steps={args.total_timesteps}, logs={log_path}"
    )

    try:
        for global_step in range(1, args.total_timesteps + 1):
            epsilon = linear_schedule(
                args.epsilon_start,
                args.epsilon_end,
                epsilon_duration,
                global_step,
            )

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.from_numpy(observation).to(
                    device,
                    dtype=torch.float32,
                )
                obs_tensor = obs_tensor.unsqueeze(0).div(255.0)
                with torch.no_grad():
                    action = int(policy_net(obs_tensor).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_observation = as_observation_array(next_observation)

            replay_buffer.add(observation, action, reward, next_observation, done)
            observation = next_observation

            if (
                global_step > args.learning_starts
                and len(replay_buffer) >= args.batch_size
                and global_step % args.train_frequency == 0
            ):
                batch = replay_buffer.sample(args.batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

                with torch.no_grad():
                    next_q = target_net(next_obs_batch).max(dim=1).values
                    target_q = reward_batch + args.gamma * (1.0 - done_batch) * next_q

                current_q = policy_net(obs_batch).gather(
                    1,
                    action_batch.unsqueeze(1),
                ).squeeze(1)
                loss = F.smooth_l1_loss(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                optimizer.step()
                last_loss = float(loss.item())

            if global_step % args.target_update_frequency == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if "episode" in info:
                episode_count += 1
                episode_return = scalar(info["episode"]["r"])
                episode_length = scalar(info["episode"]["l"])
                episode_returns.append(episode_return)
                elapsed = max(1.0, time.time() - start_time)
                agent_steps_per_second = global_step / elapsed
                atari_frames_per_second = agent_steps_per_second * frame_skip
                moving_average_return = float(
                    np.mean(episode_returns[-args.moving_average_window:])
                )

                logger.writerow(
                    {
                        "step": global_step,
                        "atari_frames": global_step * frame_skip,
                        "episode": episode_count,
                        "return": episode_return,
                        "moving_average_return": moving_average_return,
                        "length": episode_length,
                        "epsilon": epsilon,
                        "loss": last_loss,
                        "agent_steps_per_second": agent_steps_per_second,
                        "atari_frames_per_second": atari_frames_per_second,
                    }
                )
                log_file.flush()
                print(
                    "frame_skip={frame_skip} step={step} episode={episode} "
                    "return={ret:.2f} moving_avg={avg:.2f} "
                    "epsilon={eps:.3f} fps={fps:.0f}".format(
                        frame_skip=frame_skip,
                        step=global_step,
                        episode=episode_count,
                        ret=episode_return,
                        avg=moving_average_return,
                        eps=epsilon,
                        fps=agent_steps_per_second,
                    )
                )

            if done:
                observation, _ = env.reset()
                observation = as_observation_array(observation)

            if global_step % args.save_frequency == 0:
                save_checkpoint(
                    run_dir / f"breakout_frame_skip_{frame_skip}_step_{global_step}.pt",
                    policy_net,
                    optimizer,
                    experiment_args,
                    global_step,
                )

    finally:
        final_checkpoint = run_dir / f"breakout_frame_skip_{frame_skip}_final.pt"
        save_checkpoint(final_checkpoint, policy_net, optimizer, experiment_args, global_step)
        log_file.close()
        env.close()

    wall_clock_seconds = max(1e-9, time.time() - start_time)
    final_average_score, best_score, training_stability_std = compute_episode_metrics(
        episode_returns,
        args.final_average_episodes,
    )
    agent_steps_per_second = global_step / wall_clock_seconds
    atari_frames_per_second = agent_steps_per_second * frame_skip

    return ExperimentResult(
        frame_skip=frame_skip,
        is_baseline=frame_skip == args.baseline_frame_skip,
        run_dir=run_dir,
        final_average_score=final_average_score,
        best_score=best_score,
        training_stability_std=training_stability_std,
        episode_count=episode_count,
        final_step=global_step,
        wall_clock_seconds=wall_clock_seconds,
        agent_steps_per_second=agent_steps_per_second,
        atari_frames_per_second=atari_frames_per_second,
        final_checkpoint=final_checkpoint,
    )


def write_summary(summary_path: Path, results: list[ExperimentResult]) -> None:
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "frame_skip",
                "is_baseline",
                "final_average_score",
                "best_score",
                "training_stability_std",
                "episode_count",
                "final_step",
                "estimated_atari_frames",
                "wall_clock_seconds",
                "agent_steps_per_second",
                "atari_frames_per_second",
                "run_dir",
                "final_checkpoint",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "frame_skip": result.frame_skip,
                    "is_baseline": result.is_baseline,
                    "final_average_score": result.final_average_score,
                    "best_score": result.best_score,
                    "training_stability_std": result.training_stability_std,
                    "episode_count": result.episode_count,
                    "final_step": result.final_step,
                    "estimated_atari_frames": result.final_step * result.frame_skip,
                    "wall_clock_seconds": result.wall_clock_seconds,
                    "agent_steps_per_second": result.agent_steps_per_second,
                    "atari_frames_per_second": result.atari_frames_per_second,
                    "run_dir": result.run_dir,
                    "final_checkpoint": result.final_checkpoint,
                }
            )


def write_notes(notes_path: Path, args: argparse.Namespace) -> None:
    notes_path.write_text(
        "\n".join(
            [
                "Breakout frame-skip experiment notes",
                "",
                f"Environment: {args.env_id}",
                f"Algorithm: DQN from train_dqn_original.py",
                f"Training budget per run: {args.total_timesteps} agent steps",
                f"Baseline frame skip: {args.baseline_frame_skip}",
                f"Compared frame skips: {', '.join(str(value) for value in args.frame_skips)}",
                "",
                "Metrics:",
                "final_average_score = mean raw episode return over the final N episodes",
                "best_score = best raw episode return observed during training",
                "training_stability_std = standard deviation of final N episode returns",
                "agent_steps_per_second = wrapper steps per wall-clock second",
                "atari_frames_per_second = agent_steps_per_second * frame_skip",
                "",
                "Interpretation:",
                "Lower frame skip gives more frequent decisions but usually slower Atari-frame throughput.",
                "Higher frame skip may train faster in raw frames but can miss short timing windows.",
                "Frame skip 4 is the common Atari DQN baseline setting.",
            ]
        ),
        encoding="utf-8",
    )


def run_experiments(args: argparse.Namespace, device: torch.device) -> None:
    if "breakout" not in args.env_id.lower():
        print(f"Warning: env_id={args.env_id!r} does not look like a Breakout env.")

    run_name = args.experiment_name or f"Breakout_frame_skip_{time.strftime('%Y%m%d_%H%M%S')}"
    base_run_dir = Path(args.run_dir) / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Experiment root: {base_run_dir}")
    print(f"Frame skips: {args.frame_skips}")

    results: list[ExperimentResult] = []
    for experiment_index, frame_skip in enumerate(args.frame_skips):
        result = train_frame_skip_experiment(
            args,
            base_run_dir,
            device,
            frame_skip,
            experiment_index,
        )
        results.append(result)
        write_summary(base_run_dir / "frame_skip_summary.csv", results)

    summary_path = base_run_dir / "frame_skip_summary.csv"
    notes_path = base_run_dir / "experiment_notes.txt"
    write_summary(summary_path, results)
    write_notes(notes_path, args)

    print("\nFrame-skip comparison summary:")
    for result in results:
        baseline = " baseline" if result.is_baseline else ""
        print(
            "frame_skip={frame_skip}{baseline}: final_avg={final_avg:.2f}, "
            "best={best:.2f}, stability_std={std:.2f}, "
            "agent_fps={agent_fps:.0f}, atari_fps={atari_fps:.0f}".format(
                frame_skip=result.frame_skip,
                baseline=baseline,
                final_avg=result.final_average_score,
                best=result.best_score,
                std=result.training_stability_std,
                agent_fps=result.agent_steps_per_second,
                atari_fps=result.atari_frames_per_second,
            )
        )

    print(f"\nSummary CSV: {summary_path}")
    print(f"Notes: {notes_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Breakout DQN frame-skip experiments."
    )
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--total-timesteps", type=int, default=600_000)
    parser.add_argument("--frame-skips", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--baseline-frame-skip", type=int, default=4)

    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)

    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--exploration-fraction", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--different-seeds",
        action="store_true",
        help="Use seed + experiment index instead of the same seed for every run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to train on: auto, cuda, cuda:0, cuda:1, cpu, or mps.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="CUDA GPU index to use, e.g. 0 or 1. Overrides --device.",
    )
    parser.add_argument("--run-dir", default="runs_breakout")
    parser.add_argument("--experiment-name")
    parser.add_argument("--save-frequency", type=int, default=100_000)

    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-rewards", dest="clip_rewards", action="store_false")
    parser.set_defaults(clip_rewards=True)

    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--render-mode", default=None)

    parser.add_argument("--final-average-episodes", type=int, default=10)
    parser.add_argument("--moving-average-window", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device, args.gpu_id)
    run_experiments(args, device)


if __name__ == "__main__":
    main()
