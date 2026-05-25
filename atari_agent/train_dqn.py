from __future__ import annotations

import argparse
import csv
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import ale_py
import cv2
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


class PongRewardShaping(gym.Wrapper):
    """Adds denser Pong rewards for paddle hits, misses, and scored points."""

    def __init__(
        self,
        env: gym.Env,
        hit_reward: float = 1.0,
        miss_penalty: float = -1.0,
        miss_distance_penalty: float = -1.0,
        score_reward: float = 2.0,
        hit_x_fraction: float = 0.70,
        min_hit_interval: int = 8,
    ):
        super().__init__(env)
        self.hit_reward = hit_reward
        self.miss_penalty = miss_penalty
        self.miss_distance_penalty = miss_distance_penalty
        self.score_reward = score_reward
        self.hit_x_fraction = hit_x_fraction
        self.min_hit_interval = min_hit_interval
        self.raw_step = 0
        self.last_hit_step = -min_hit_interval
        self.previous_ball: tuple[float, float] | None = None
        self.previous_paddle: tuple[float, float] | None = None
        self.previous_dx: float | None = None

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.raw_step = 0
        self.last_hit_step = -self.min_hit_interval
        self.previous_ball = self._find_ball(observation)
        self.previous_paddle = self._find_agent_paddle(observation)
        self.previous_dx = None
        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.raw_step += 1

        raw_reward = float(reward)
        hit_ball = False
        miss_distance_penalty = 0.0
        if raw_reward > 0:
            shaped_reward = self.score_reward
            self._reset_ball_motion(observation)
        elif raw_reward < 0:
            miss_distance_penalty = self._miss_distance_penalty(observation)
            shaped_reward = self.miss_penalty + miss_distance_penalty
            self._reset_ball_motion(observation)
        else:
            hit_ball = self._agent_hit_ball(observation)
            shaped_reward = self.hit_reward if hit_ball else 0.0
            self.previous_paddle = self._find_agent_paddle(observation)

        info = dict(info)
        info["raw_reward"] = raw_reward
        info["hit_ball"] = hit_ball
        info["miss_distance_penalty"] = miss_distance_penalty
        return observation, shaped_reward, terminated, truncated, info

    def _reset_ball_motion(self, observation: Any) -> None:
        self.previous_ball = self._find_ball(observation)
        self.previous_paddle = self._find_agent_paddle(observation)
        self.previous_dx = None

    def _miss_distance_penalty(self, observation: Any) -> float:
        ball = self.previous_ball or self._find_ball(observation)
        paddle = self._find_agent_paddle(observation) or self.previous_paddle
        if ball is None or paddle is None:
            return 0.0

        frame = np.asarray(observation)
        y_min = 34
        y_max = min(frame.shape[0], 194)
        playfield_height = max(1.0, float(y_max - y_min))
        distance_fraction = min(abs(ball[1] - paddle[1]) / playfield_height, 1.0)
        max_penalty = min(0.0, self.miss_distance_penalty)
        return max_penalty * distance_fraction

    def _agent_hit_ball(self, observation: Any) -> bool:
        ball = self._find_ball(observation)
        if ball is None:
            self.previous_ball = None
            self.previous_dx = None
            return False

        hit = False
        if self.previous_ball is not None:
            dx = ball[0] - self.previous_ball[0]
            hit_x_min = np.asarray(observation).shape[1] * self.hit_x_fraction
            if (
                self.previous_dx is not None
                and self.previous_dx > 0
                and dx < 0
                and max(ball[0], self.previous_ball[0]) >= hit_x_min
                and self.raw_step - self.last_hit_step >= self.min_hit_interval
            ):
                hit = True
                self.last_hit_step = self.raw_step

            if abs(dx) > 0.25:
                self.previous_dx = dx

        self.previous_ball = ball
        return hit

    def _find_ball(self, observation: Any) -> tuple[float, float] | None:
        frame = np.asarray(observation)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None

        y_min = 34
        y_max = min(frame.shape[0], 194)
        crop = frame[y_min:y_max, :, :3].astype(np.int16)
        background = np.median(crop.reshape(-1, 3), axis=0)
        color_distance = np.abs(crop - background).sum(axis=2)
        mask = color_distance > 60

        center_x = mask.shape[1] // 2
        mask[:, max(0, center_x - 4): center_x + 5] = False

        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        candidates: list[tuple[float, float, float]] = []
        for label in range(1, component_count):
            x, y, width, height, area = stats[label]
            if 2 <= area <= 80 and 1 <= width <= 12 and 1 <= height <= 12:
                center = centroids[label]
                candidates.append((float(center[0]), float(center[1] + y_min), area))

        if not candidates:
            return None
        if self.previous_ball is None:
            x, y, _ = max(candidates, key=lambda candidate: candidate[2])
            return x, y

        previous_x, previous_y = self.previous_ball
        x, y, _ = min(
            candidates,
            key=lambda candidate: (
                candidate[0] - previous_x
            ) ** 2 + (candidate[1] - previous_y) ** 2,
        )
        return x, y

    def _find_agent_paddle(self, observation: Any) -> tuple[float, float] | None:
        frame = np.asarray(observation)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None

        y_min = 34
        y_max = min(frame.shape[0], 194)
        crop = frame[y_min:y_max, :, :3].astype(np.int16)
        background = np.median(crop.reshape(-1, 3), axis=0)
        color_distance = np.abs(crop - background).sum(axis=2)
        mask = color_distance > 60

        center_x = mask.shape[1] // 2
        mask[:, max(0, center_x - 4): center_x + 5] = False

        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        candidates: list[tuple[float, float, float]] = []
        paddle_x_min = frame.shape[1] * self.hit_x_fraction
        for label in range(1, component_count):
            x, y, width, height, area = stats[label]
            center = centroids[label]
            if (
                center[0] >= paddle_x_min
                and 2 <= width <= 10
                and 8 <= height <= 40
                and 12 <= area <= 160
            ):
                candidates.append((float(center[0]), float(center[1] + y_min), area))

        if not candidates:
            return None
        x, y, _ = max(candidates, key=lambda candidate: candidate[2])
        return x, y


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        device: torch.device,
        pin_memory: bool = False,
    ):
        self.capacity = capacity
        self.device = device
        self.pin_memory = pin_memory
        self.observations = torch.empty(
            (capacity, *obs_shape), dtype=torch.uint8, pin_memory=pin_memory
        )
        self.next_observations = torch.empty(
            (capacity, *obs_shape), dtype=torch.uint8, pin_memory=pin_memory
        )
        self.actions = torch.empty(capacity, dtype=torch.int64, pin_memory=pin_memory)
        self.rewards = torch.empty(capacity, dtype=torch.float32, pin_memory=pin_memory)
        self.dones = torch.empty(capacity, dtype=torch.float32, pin_memory=pin_memory)
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
        self.observations[self.position].copy_(torch.as_tensor(observation))
        self.next_observations[self.position].copy_(torch.as_tensor(next_observation))
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.full = self.full or self.position == 0

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (batch_size,))

        observations = self.observations[indices].to(
            self.device, dtype=torch.float32, non_blocking=self.pin_memory
        )
        next_observations = self.next_observations[indices].to(
            self.device, dtype=torch.float32, non_blocking=self.pin_memory
        )
        actions = self.actions[indices].to(self.device, non_blocking=self.pin_memory)
        rewards = self.rewards[indices].to(self.device, non_blocking=self.pin_memory)
        dones = self.dones[indices].to(self.device, non_blocking=self.pin_memory)

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

    use_pong_reward_shaping = (
        args.pong_reward_shaping and "pong" in args.env_id.lower() and not eval_mode
    )
    if use_pong_reward_shaping:
        env = PongRewardShaping(
            env,
            hit_reward=args.hit_reward,
            miss_penalty=args.miss_penalty,
            miss_distance_penalty=args.miss_distance_penalty,
            score_reward=args.score_reward,
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

    if args.clip_rewards and not eval_mode and not use_pong_reward_shaping:
        env = ClipReward(env)

    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)
    return env


def pick_device(device_name: str, gpu_id: int | None = None) -> torch.device:
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError("--gpu-id was set, but CUDA is not available.")
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(
                f"--gpu-id must be between 0 and {gpu_count - 1}; got {gpu_id}."
            )
        return torch.device(f"cuda:{gpu_id}")

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch_runtime(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        if args.amp:
            print("--amp was requested, but mixed precision is only enabled on CUDA.")
        return

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.set_float32_matmul_precision(args.float32_matmul_precision)
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")


def autocast_context(enabled: bool):
    if enabled:
        return torch.amp.autocast(device_type="cuda")
    return nullcontext()


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


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_model(checkpoint_path: Path, env: gym.Env, device: torch.device) -> DQN:
    checkpoint = load_checkpoint(checkpoint_path, device)
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

    optimizer = torch.optim.Adam(
        policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        observation_shape,
        device,
        pin_memory=device.type == "cuda",
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    resume_checkpoint = args.resume_checkpoint or args.checkpoint
    start_step = 1
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(Path(resume_checkpoint), device)
        checkpoint_observation_shape = tuple(
            checkpoint.get("observation_shape", observation_shape)
        )
        checkpoint_action_count = int(checkpoint.get("action_count", action_count))
        if checkpoint_observation_shape != observation_shape:
            raise ValueError(
                "Checkpoint observation shape "
                f"{checkpoint_observation_shape} does not match current env "
                f"shape {observation_shape}."
            )
        if checkpoint_action_count != action_count:
            raise ValueError(
                "Checkpoint action count "
                f"{checkpoint_action_count} does not match current env "
                f"action count {action_count}."
            )

        policy_net.load_state_dict(checkpoint["model_state_dict"])
        target_net.load_state_dict(policy_net.state_dict())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("global_step", 0)) + 1
        if start_step > args.total_timesteps:
            raise ValueError(
                "--total-timesteps must be greater than the checkpoint step "
                f"({start_step - 1}) to continue training."
            )

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
    if resume_checkpoint is not None:
        print(f"Resuming from {resume_checkpoint} at step {start_step}.")

    global_step = start_step - 1
    try:
        for global_step in range(start_step, args.total_timesteps + 1):
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
                    with autocast_context(amp_enabled):
                        action = int(policy_net(obs_tensor).argmax(dim=1).item())

            next_observation, reward, terminated, truncated, info = env.step(
                action)
            done = terminated or truncated
            next_observation = as_observation_array(next_observation)

            replay_buffer.add(observation, action, reward,
                              next_observation, done)
            observation = next_observation

            if (
                global_step > args.learning_starts
                and len(replay_buffer) >= args.batch_size
                and global_step % args.train_frequency == 0
            ):
                batch = replay_buffer.sample(args.batch_size)
                obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = batch

                with torch.no_grad():
                    with autocast_context(amp_enabled):
                        next_q = target_net(next_obs_batch).max(dim=1).values
                    target_q = reward_batch + args.gamma * \
                        (1.0 - done_batch) * next_q

                with autocast_context(amp_enabled):
                    current_q = policy_net(obs_batch).gather(
                        1, action_batch.unsqueeze(1)
                    ).squeeze(1)
                loss = F.smooth_l1_loss(current_q.float(), target_q.float())

                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        policy_net.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        policy_net.parameters(), args.max_grad_norm)
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
        save_checkpoint(run_dir / "dqn_final.pt", policy_net,
                        optimizer, args, global_step)
        log_file.close()
        env.close()

    print(f"Finished. Final checkpoint: {run_dir / 'dqn_final.pt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DQN agent on Atari.")
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--buffer-size", type=int, default=20_000)
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
    parser.add_argument("--device", default="auto",
                        help="Device to train on: auto, cuda, cuda:0, cuda:1, cpu, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="CUDA GPU index to use, e.g. 0 or 1. Overrides --device.")
    parser.add_argument("--amp", action="store_true",
                        help="Use CUDA automatic mixed precision.")
    parser.add_argument("--float32-matmul-precision",
                        choices=["highest", "high", "medium"], default="high",
                        help="CUDA matmul precision setting for PyTorch 2.x.")
    parser.add_argument("--no-cudnn-benchmark",
                        dest="cudnn_benchmark", action="store_false",
                        help="Disable cuDNN autotuning.")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--save-frequency", type=int, default=50_000)
    parser.set_defaults(cudnn_benchmark=True)

    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-rewards",
                        dest="clip_rewards", action="store_false")
    parser.set_defaults(clip_rewards=True)

    parser.add_argument("--no-pong-reward-shaping",
                        dest="pong_reward_shaping", action="store_false")
    parser.add_argument("--hit-reward", type=float, default=1.0)
    parser.add_argument("--miss-penalty", type=float, default=-2.0)
    parser.add_argument("--miss-distance-penalty", type=float, default=-1.0)
    parser.add_argument("--score-reward", type=float, default=4.0)
    parser.set_defaults(pong_reward_shaping=True)

    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--render-mode", default=None)

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device, args.gpu_id)
    configure_torch_runtime(args, device)
    if args.eval_only:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
