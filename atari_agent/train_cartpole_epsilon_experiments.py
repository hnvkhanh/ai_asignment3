from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class EpsilonSchedule:
    name: str
    label: str
    start: float
    end: float
    decay_fraction: float | None

    def value(self, episode: int, total_episodes: int) -> float:
        if self.decay_fraction is None:
            return self.start
        decay_episodes = max(1, int(total_episodes * self.decay_fraction))
        progress = min(1.0, (episode - 1) / decay_episodes)
        return self.start + progress * (self.end - self.start)

    @property
    def description(self) -> str:
        if self.decay_fraction is None:
            return f"fixed epsilon = {self.start:.2f}"
        return (
            f"linear {self.start:.2f} -> {self.end:.2f} over "
            f"{self.decay_fraction:.0%} of training episodes"
        )


SCHEDULES = {
    "fast_decay": EpsilonSchedule("fast_decay", "Fast decay", 1.0, 0.05, 0.20),
    "slow_decay": EpsilonSchedule("slow_decay", "Slow decay", 1.0, 0.05, 0.80),
    "fixed_0.1": EpsilonSchedule("fixed_0.1", "Fixed epsilon 0.1", 0.10, 0.10, None),
    "fixed_0.2": EpsilonSchedule("fixed_0.2", "Fixed epsilon 0.2", 0.20, 0.20, None),
}


@dataclass(frozen=True)
class ExperimentResult:
    schedule: EpsilonSchedule
    seed: int
    log_path: Path
    checkpoint_path: Path
    episode_returns: tuple[float, ...]
    evaluation_returns: tuple[float, ...]
    total_steps: int

    @property
    def final_training_average(self) -> float:
        return float(np.mean(self.episode_returns[-20:]))

    @property
    def best_training_return(self) -> float:
        return float(np.max(self.episode_returns))

    @property
    def evaluation_mean(self) -> float:
        return float(np.mean(self.evaluation_returns))

    @property
    def evaluation_std(self) -> float:
        return float(np.std(self.evaluation_returns))


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int):
        self.capacity = capacity
        self.observations = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_observations = np.empty(
            (capacity, observation_size), dtype=np.float32
        )
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminal: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.terminals[self.position] = float(terminal)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(self.size, size=batch_size)
        return (
            torch.from_numpy(self.observations[indices]).to(device),
            torch.from_numpy(self.actions[indices]).to(device),
            torch.from_numpy(self.rewards[indices]).to(device),
            torch.from_numpy(self.next_observations[indices]).to(device),
            torch.from_numpy(self.terminals[indices]).to(device),
        )

    def __len__(self) -> int:
        return self.size


class DQN(nn.Module):
    def __init__(self, observation_size: int, action_count: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def pick_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate_model(
    model: DQN,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[float, ...]:
    env = gym.make(args.env_id)
    returns: list[float] = []
    try:
        for episode in range(args.eval_episodes):
            observation, _ = env.reset(seed=args.eval_seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                observation_tensor = torch.from_numpy(
                    np.asarray(observation, dtype=np.float32)
                ).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = int(model(observation_tensor).argmax(dim=1).item())
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
                done = terminated or truncated
            returns.append(episode_return)
    finally:
        env.close()
    return tuple(returns)


def train_run(
    schedule: EpsilonSchedule,
    seed: int,
    run_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> ExperimentResult:
    schedule_dir = run_dir / schedule.name / f"seed_{seed}"
    schedule_dir.mkdir(parents=True, exist_ok=True)
    log_path = schedule_dir / "training_log.csv"
    checkpoint_path = schedule_dir / "cartpole_dqn_final.pt"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed)

    env = gym.make(args.env_id)
    env.action_space.seed(seed)
    observation_size = int(env.observation_space.shape[0])
    action_count = int(env.action_space.n)
    policy_net = DQN(observation_size, action_count, args.hidden_size).to(device)
    target_net = DQN(observation_size, action_count, args.hidden_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, observation_size)

    returns: list[float] = []
    total_steps = 0
    last_loss = np.nan

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "episode",
                "total_steps",
                "epsilon",
                "return",
                "moving_average_return",
                "loss",
            ],
        )
        writer.writeheader()

        try:
            for episode in range(1, args.episodes + 1):
                observation, _ = env.reset(seed=seed + episode)
                observation = np.asarray(observation, dtype=np.float32)
                epsilon = schedule.value(episode, args.episodes)
                episode_return = 0.0
                done = False

                while not done:
                    if rng.random() < epsilon:
                        action = rng.randrange(action_count)
                    else:
                        observation_tensor = torch.from_numpy(observation).unsqueeze(0).to(
                            device
                        )
                        with torch.no_grad():
                            action = int(
                                policy_net(observation_tensor).argmax(dim=1).item()
                            )

                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_observation = np.asarray(next_observation, dtype=np.float32)
                    replay_buffer.add(
                        observation,
                        action,
                        float(reward),
                        next_observation,
                        terminated,
                    )
                    observation = next_observation
                    episode_return += float(reward)
                    total_steps += 1
                    done = terminated or truncated

                    if (
                        total_steps >= args.learning_starts
                        and len(replay_buffer) >= args.batch_size
                        and total_steps % args.train_frequency == 0
                    ):
                        (
                            observation_batch,
                            action_batch,
                            reward_batch,
                            next_observation_batch,
                            terminal_batch,
                        ) = replay_buffer.sample(args.batch_size, device)
                        with torch.no_grad():
                            next_q = target_net(next_observation_batch).max(dim=1).values
                            target_q = reward_batch + args.gamma * (
                                1.0 - terminal_batch
                            ) * next_q
                        current_q = policy_net(observation_batch).gather(
                            1, action_batch.unsqueeze(1)
                        ).squeeze(1)
                        loss = F.smooth_l1_loss(current_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                        optimizer.step()
                        last_loss = float(loss.item())

                    if total_steps % args.target_update_frequency == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                returns.append(episode_return)
                moving_average = float(
                    np.mean(returns[-args.moving_average_window :])
                )
                writer.writerow(
                    {
                        "episode": episode,
                        "total_steps": total_steps,
                        "epsilon": epsilon,
                        "return": episode_return,
                        "moving_average_return": moving_average,
                        "loss": last_loss,
                    }
                )
                log_file.flush()
        finally:
            env.close()

    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "schedule": schedule.name,
            "seed": seed,
            "episodes": args.episodes,
            "observation_size": observation_size,
            "action_count": action_count,
            "hidden_size": args.hidden_size,
        },
        checkpoint_path,
    )
    evaluation_returns = evaluate_model(policy_net, args, device)
    return ExperimentResult(
        schedule=schedule,
        seed=seed,
        log_path=log_path,
        checkpoint_path=checkpoint_path,
        episode_returns=tuple(returns),
        evaluation_returns=evaluation_returns,
        total_steps=total_steps,
    )


def write_results(run_dir: Path, results: list[ExperimentResult]) -> tuple[Path, Path]:
    summary_path = run_dir / "epsilon_schedule_summary.csv"
    evaluation_path = run_dir / "evaluation_episodes.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=[
                "schedule",
                "label",
                "description",
                "seed",
                "episodes",
                "total_steps",
                "final_training_average",
                "best_training_return",
                "evaluation_mean",
                "evaluation_std",
                "training_log",
                "checkpoint",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "schedule": result.schedule.name,
                    "label": result.schedule.label,
                    "description": result.schedule.description,
                    "seed": result.seed,
                    "episodes": len(result.episode_returns),
                    "total_steps": result.total_steps,
                    "final_training_average": result.final_training_average,
                    "best_training_return": result.best_training_return,
                    "evaluation_mean": result.evaluation_mean,
                    "evaluation_std": result.evaluation_std,
                    "training_log": result.log_path,
                    "checkpoint": result.checkpoint_path,
                }
            )

    with evaluation_path.open("w", newline="", encoding="utf-8") as evaluation_file:
        writer = csv.DictWriter(
            evaluation_file,
            fieldnames=["schedule", "label", "seed", "episode", "return"],
        )
        writer.writeheader()
        for result in results:
            for episode, episode_return in enumerate(result.evaluation_returns, start=1):
                writer.writerow(
                    {
                        "schedule": result.schedule.name,
                        "label": result.schedule.label,
                        "seed": result.seed,
                        "episode": episode,
                        "return": episode_return,
                    }
                )
    return summary_path, evaluation_path


def write_notes(run_dir: Path, args: argparse.Namespace, schedules: list[EpsilonSchedule]) -> None:
    notes_path = run_dir / "experiment_notes.txt"
    notes_path.write_text(
        "\n".join(
            [
                "CartPole epsilon-greedy schedule experiment",
                "",
                f"Environment: {args.env_id}",
                f"Training episodes per run: {args.episodes}",
                f"Training seeds: {', '.join(str(seed) for seed in args.seeds)}",
                f"Greedy evaluation episodes per run: {args.eval_episodes}",
                "",
                "Schedules:",
                *[
                    f"- {schedule.label}: {schedule.description}"
                    for schedule in schedules
                ],
                "",
                "Metric:",
                "evaluation_mean = mean CartPole return during greedy final evaluation",
                "training plots use episode return with a moving average.",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare epsilon-greedy schedules for a CartPole DQN agent."
    )
    parser.add_argument("--env-id", default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--schedules",
        choices=tuple(SCHEDULES),
        nargs="+",
        default=list(SCHEDULES),
    )
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-starts", type=int, default=500)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--target-update-frequency", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--moving-average-window", type=int, default=20)
    parser.add_argument("--run-dir", type=Path, default=PROJECT_ROOT / "runs_cartpole")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()
    if args.episodes <= 0 or args.eval_episodes <= 0:
        parser.error("--episodes and --eval-episodes must be greater than zero.")
    if args.hidden_size <= 0 or args.batch_size <= 0 or args.buffer_size <= 0:
        parser.error("Network and replay-buffer sizes must be greater than zero.")
    if args.learning_starts < 0 or args.train_frequency <= 0:
        parser.error("Learning thresholds and frequencies must be valid.")
    args.schedules = list(dict.fromkeys(args.schedules))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def main() -> None:
    args = parse_args()
    schedules = [SCHEDULES[name] for name in args.schedules]
    device = pick_device(args.device)
    run_name = args.experiment_name or (
        f"CartPole_epsilon_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = (args.run_dir / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    write_notes(run_dir, args, schedules)

    print(f"Training CartPole DQN on {device}. Experiment: {run_dir}")
    results: list[ExperimentResult] = []
    for schedule in schedules:
        print(f"\n{schedule.label}: {schedule.description}")
        for seed in args.seeds:
            result = train_run(schedule, seed, run_dir, args, device)
            results.append(result)
            write_results(run_dir, results)
            print(
                f"seed={seed}: final_train_avg={result.final_training_average:.2f} "
                f"eval_mean={result.evaluation_mean:.2f} "
                f"eval_std={result.evaluation_std:.2f}"
            )

    summary_path, evaluation_path = write_results(run_dir, results)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved evaluation episodes: {evaluation_path}")


if __name__ == "__main__":
    main()
