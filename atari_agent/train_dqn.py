from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch import nn


ALE_REGISTERED = False


def register_ale() -> None:
    global ALE_REGISTERED
    if not ALE_REGISTERED:
        gym.register_envs(ale_py)
        ALE_REGISTERED = True


class ClipReward(gym.RewardWrapper):
    """Clips Atari rewards to -1, 0, or 1 for more stable DQN updates."""

    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple[int, ...], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.observations = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_observations = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.position = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self.observations[self.position] = observation
        self.next_observations[self.position] = next_observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, len(self), size=batch_size)

        observations = torch.from_numpy(self.observations[indices]).to(
            self.device, dtype=torch.float32
        )
        next_observations = torch.from_numpy(self.next_observations[indices]).to(
            self.device, dtype=torch.float32
        )
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        observations.div_(255.0)
        next_observations.div_(255.0)

        return observations, actions, rewards, next_observations, dones


class DQN(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], action_count: int):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError(
                "Expected stacked Atari observations shaped like "
                "(frames, height, width)."
            )

        self.observation_shape = observation_shape
        self.action_count = action_count
        channels = observation_shape[0]
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape)
            feature_count = self.features(dummy).shape[1]

        self.q_values = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float().div(255.0)
        return self.q_values(self.features(observations))


def make_env(args: argparse.Namespace, run_dir: Path, eval_mode: bool = False) -> gym.Env:
    register_ale()

    render_mode = args.render_mode
    if args.capture_video and not eval_mode:
        render_mode = "rgb_array"

    env = gym.make(
        args.env_id,
        frameskip=1,
        full_action_space=args.full_action_space,
        render_mode=render_mode,
    )

    if args.capture_video and not eval_mode:
        video_dir = run_dir / "videos"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id % args.video_every == 0,
        )

    env = AtariPreprocessing(
        env,
        noop_max=args.noop_max,
        frame_skip=args.frame_skip,
        screen_size=args.screen_size,
        terminal_on_life_loss=args.terminal_on_life_loss,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=args.frame_stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if args.clip_rewards and not eval_mode:
        env = ClipReward(env)

    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


def pick_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def linear_schedule(start: float, end: float, duration: int, step: int) -> float:
    if duration <= 0 or step >= duration:
        return end
    mix = step / duration
    return start + mix * (end - start)


def as_observation_array(observation: Any) -> np.ndarray:
    return np.asarray(observation, dtype=np.uint8)


def scalar(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def save_checkpoint(
    path: Path,
    model: DQN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    global_step: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "env_id": args.env_id,
            "global_step": global_step,
            "observation_shape": model.observation_shape,
            "action_count": model.action_count,
            "args": vars(args),
        },
        path,
    )


def load_model(checkpoint_path: Path, env: gym.Env, device: torch.device) -> DQN:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    observation_shape = tuple(env.observation_space.shape)
    action_count = env.action_space.n
    model = DQN(observation_shape, action_count).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --eval-only")

    run_dir = Path(args.run_dir) / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args, run_dir, eval_mode=True)
    model = load_model(Path(args.checkpoint), env, device)

    returns: list[float] = []
    for episode in range(1, args.eval_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            if random.random() < args.eval_epsilon:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.from_numpy(as_observation_array(observation)).to(
                    device, dtype=torch.float32
                )
                obs_tensor = obs_tensor.unsqueeze(0).div(255.0)
                with torch.no_grad():
                    action = int(model(obs_tensor).argmax(dim=1).item())

            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated

        returns.append(total_reward)
        print(f"eval_episode={episode} return={total_reward:.2f}")

    env.close()
    mean_return = float(np.mean(returns))
    print(f"mean_eval_return={mean_return:.2f}")


def train(args: argparse.Namespace, device: torch.device) -> None:
    run_name = f"{args.env_id.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args, run_dir)
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
        fieldnames=["step", "episode", "return", "length", "epsilon", "loss"],
    )
    logger.writeheader()

    observation, _ = env.reset(seed=args.seed)
    observation = as_observation_array(observation)
    episode_count = 0
    last_loss = np.nan
    epsilon_duration = int(args.exploration_fraction * args.total_timesteps)

    print(f"Training {args.env_id} on {device}. Logs: {log_path}")

    global_step = 0
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
                    device, dtype=torch.float32
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
                    1, action_batch.unsqueeze(1)
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
                logger.writerow(
                    {
                        "step": global_step,
                        "episode": episode_count,
                        "return": episode_return,
                        "length": episode_length,
                        "epsilon": epsilon,
                        "loss": last_loss,
                    }
                )
                log_file.flush()
                print(
                    "step={step} episode={episode} return={ret:.2f} "
                    "length={length:.0f} epsilon={eps:.3f} loss={loss:.4f}".format(
                        step=global_step,
                        episode=episode_count,
                        ret=episode_return,
                        length=episode_length,
                        eps=epsilon,
                        loss=last_loss,
                    )
                )

            if done:
                observation, _ = env.reset()
                observation = as_observation_array(observation)

            if global_step % args.save_frequency == 0:
                save_checkpoint(
                    run_dir / f"dqn_step_{global_step}.pt",
                    policy_net,
                    optimizer,
                    args,
                    global_step,
                )

    finally:
        save_checkpoint(run_dir / "dqn_final.pt", policy_net, optimizer, args, global_step)
        log_file.close()
        env.close()

    print(f"Finished. Final checkpoint: {run_dir / 'dqn_final.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari.")
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--save-frequency", type=int, default=50_000)

    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-rewards", dest="clip_rewards", action="store_false")
    parser.set_defaults(clip_rewards=True)

    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--render-mode", default=None)

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if args.eval_only:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
