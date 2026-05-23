from __future__ import annotations

import argparse
import csv
import random
import time
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

try:
    from train_dqn_original import (
        ClipReward,
        DQN,
        ReplayBuffer,
        as_observation_array,
        linear_schedule,
        load_checkpoint,
        save_checkpoint,
        scalar,
    )
except ImportError:
    from .train_dqn_original import (
        ClipReward,
        DQN,
        ReplayBuffer,
        as_observation_array,
        linear_schedule,
        load_checkpoint,
        save_checkpoint,
        scalar,
    )


ALE_REGISTERED = False


def register_ale() -> None:
    global ALE_REGISTERED
    if not ALE_REGISTERED:
        gym.register_envs(ale_py)
        ALE_REGISTERED = True


class BreakoutRewardShaping(gym.Wrapper):
    """Adds dense Breakout rewards for brick hits, paddle hits, and misses."""

    def __init__(
        self,
        env: gym.Env,
        brick_reward: float = 2.0,
        paddle_hit_reward: float = 1.0,
        life_loss_penalty: float = -20.0,
        miss_distance_penalty: float = -2.0,
        middle_position_penalty: float = 0.5,
        min_paddle_hit_interval: int = 6,
    ):
        super().__init__(env)
        self.brick_reward = brick_reward
        self.paddle_hit_reward = paddle_hit_reward
        self.life_loss_penalty = life_loss_penalty
        self.miss_distance_penalty = miss_distance_penalty
        self.middle_position_penalty = max(0.0, middle_position_penalty)
        self.min_paddle_hit_interval = min_paddle_hit_interval
        self.raw_step = 0
        self.last_paddle_hit_step = -min_paddle_hit_interval
        self.previous_lives: int | None = None
        self.previous_ball: tuple[float, float] | None = None
        self.previous_paddle: tuple[float, float] | None = None
        self.previous_dy: float | None = None
        self.raw_episode_return = 0.0
        self.episode_brick_hits = 0
        self.episode_paddle_hits = 0
        self.episode_life_losses = 0

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self.raw_step = 0
        self.last_paddle_hit_step = -self.min_paddle_hit_interval
        self.previous_lives = self._lives()
        self.previous_ball = self._find_ball(observation)
        self.previous_paddle = self._find_paddle(observation)
        self.previous_dy = None
        self.raw_episode_return = 0.0
        self.episode_brick_hits = 0
        self.episode_paddle_hits = 0
        self.episode_life_losses = 0
        return observation, info

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.raw_step += 1

        raw_reward = float(reward)
        self.raw_episode_return += raw_reward
        shaped_reward = self.brick_reward if raw_reward > 0.0 else 0.0
        middle_position_penalty = 0.0
        if raw_reward > 0.0:
            self.episode_brick_hits += 1
            middle_position_penalty = self._middle_position_penalty(observation)
            shaped_reward += middle_position_penalty
        hit_paddle = self._hit_paddle(observation)
        if hit_paddle:
            self.episode_paddle_hits += 1
            shaped_reward += self.paddle_hit_reward

        current_lives = self._lives()
        lost_life = (
            self.previous_lives is not None
            and current_lives is not None
            and current_lives < self.previous_lives
        )
        miss_distance_penalty = 0.0
        if lost_life:
            self.episode_life_losses += 1
            miss_distance_penalty = self._miss_distance_penalty(observation)
            shaped_reward += self.life_loss_penalty + miss_distance_penalty
            self.previous_dy = None
            self.previous_ball = self._find_ball(observation)

        if current_lives is not None:
            self.previous_lives = current_lives
        self.previous_paddle = self._find_paddle(observation) or self.previous_paddle

        info = dict(info)
        info["raw_reward"] = raw_reward
        info["hit_brick"] = raw_reward > 0.0
        info["hit_paddle"] = hit_paddle
        info["lost_life"] = lost_life
        info["miss_distance_penalty"] = miss_distance_penalty
        info["middle_position_penalty"] = middle_position_penalty
        if terminated or truncated:
            info["raw_episode_return"] = self.raw_episode_return
            info["brick_hits"] = self.episode_brick_hits
            info["paddle_hits"] = self.episode_paddle_hits
            info["life_losses"] = self.episode_life_losses
        return observation, shaped_reward, terminated, truncated, info

    def _lives(self) -> int | None:
        ale = getattr(self.unwrapped, "ale", None)
        if ale is None:
            return None
        return int(ale.lives())

    def _hit_paddle(self, observation: Any) -> bool:
        ball = self._find_ball(observation)
        paddle = self._find_paddle(observation)
        if ball is None:
            self.previous_ball = None
            self.previous_dy = None
            return False

        hit = False
        if self.previous_ball is not None and paddle is not None:
            dy = ball[1] - self.previous_ball[1]
            dx_to_paddle = abs(ball[0] - paddle[0])
            close_to_paddle_y = abs(ball[1] - paddle[1]) <= 14.0
            close_to_paddle_x = dx_to_paddle <= 28.0
            if (
                self.previous_dy is not None
                and self.previous_dy > 0.0
                and dy < 0.0
                and close_to_paddle_y
                and close_to_paddle_x
                and self.raw_step - self.last_paddle_hit_step >= self.min_paddle_hit_interval
            ):
                hit = True
                self.last_paddle_hit_step = self.raw_step

            if abs(dy) > 0.25:
                self.previous_dy = dy

        self.previous_ball = ball
        self.previous_paddle = paddle or self.previous_paddle
        return hit

    def _miss_distance_penalty(self, observation: Any) -> float:
        ball = self.previous_ball or self._find_ball(observation)
        paddle = self._find_paddle(observation) or self.previous_paddle
        if ball is None or paddle is None:
            return 0.0

        frame = np.asarray(observation)
        max_distance = max(1.0, frame.shape[1] / 2.0)
        distance_fraction = min(abs(ball[0] - paddle[0]) / max_distance, 1.0)
        max_penalty = min(0.0, self.miss_distance_penalty)
        return max_penalty * distance_fraction

    def _middle_position_penalty(self, observation: Any) -> float:
        paddle = self._find_paddle(observation) or self.previous_paddle
        if paddle is None:
            return 0.0

        frame = np.asarray(observation)
        center_x = frame.shape[1] / 2.0
        max_distance = max(1.0, center_x)
        distance_fraction = min(abs(paddle[0] - center_x) / max_distance, 1.0)
        return -self.middle_position_penalty * distance_fraction

    def _foreground_mask(self, observation: Any) -> np.ndarray | None:
        frame = np.asarray(observation)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        crop = frame[:, :, :3].astype(np.int16)
        color_strength = crop.max(axis=2)
        return color_strength > 35

    def _find_ball(self, observation: Any) -> tuple[float, float] | None:
        frame = np.asarray(observation)
        mask = self._foreground_mask(frame)
        if mask is None:
            return None

        # Ignore score, bricks, and paddle-heavy bottom pixels when searching for the ball.
        y_min = 90
        y_max = min(frame.shape[0], 190)
        search = mask[y_min:y_max].copy()
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            search.astype(np.uint8),
            connectivity=8,
        )

        candidates: list[tuple[float, float, float]] = []
        for label in range(1, component_count):
            _, _, width, height, area = stats[label]
            if 1 <= width <= 8 and 1 <= height <= 8 and 1 <= area <= 35:
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

    def _find_paddle(self, observation: Any) -> tuple[float, float] | None:
        frame = np.asarray(observation)
        mask = self._foreground_mask(frame)
        if mask is None:
            return None

        y_min = min(frame.shape[0] - 1, 180)
        y_max = min(frame.shape[0], 205)
        search = mask[y_min:y_max].copy()
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(
            search.astype(np.uint8),
            connectivity=8,
        )

        candidates: list[tuple[float, float, float]] = []
        for label in range(1, component_count):
            _, _, width, height, area = stats[label]
            if 8 <= width <= 48 and 1 <= height <= 10 and 8 <= area <= 220:
                center = centroids[label]
                candidates.append((float(center[0]), float(center[1] + y_min), area))

        if not candidates:
            return None
        x, y, _ = max(candidates, key=lambda candidate: candidate[2])
        return x, y


def make_env(args: argparse.Namespace, run_dir: Path, eval_mode: bool = False) -> gym.Env:
    register_ale()

    render_mode = args.render_mode
    if args.capture_video and not eval_mode:
        render_mode = "rgb_array"

    env_kwargs: dict[str, Any] = {
        "frameskip": 1,
        "full_action_space": args.full_action_space,
        "render_mode": render_mode,
    }
    if args.mode is not None:
        env_kwargs["mode"] = args.mode
    if args.difficulty is not None:
        env_kwargs["difficulty"] = args.difficulty

    env = gym.make(args.env_id, **env_kwargs)

    if args.capture_video and not eval_mode:
        video_dir = run_dir / "videos"
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode_id: episode_id % args.video_every == 0,
        )

    use_reward_shaping = args.reward_shaping and not eval_mode
    if use_reward_shaping:
        env = BreakoutRewardShaping(
            env,
            brick_reward=args.brick_reward,
            paddle_hit_reward=args.paddle_hit_reward,
            life_loss_penalty=args.life_loss_penalty,
            miss_distance_penalty=args.miss_distance_penalty,
            middle_position_penalty=args.middle_position_penalty,
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

    if args.clip_rewards and not eval_mode and not use_reward_shaping:
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
            raise ValueError(f"--gpu-id must be between 0 and {gpu_count - 1}; got {gpu_id}.")
        return torch.device(f"cuda:{gpu_id}")

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
                    device,
                    dtype=torch.float32,
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
    print(f"mean_eval_return={float(np.mean(returns)):.2f}")


def train(args: argparse.Namespace, device: torch.device) -> None:
    run_name = f"{args.env_id.split('/')[-1]}_shaped_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    env = make_env(args, run_dir)
    observation_shape = tuple(env.observation_space.shape)
    action_count = env.action_space.n

    policy_net = DQN(observation_shape, action_count).to(device)
    target_net = DQN(observation_shape, action_count).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayBuffer(args.buffer_size, observation_shape, device)

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
        fieldnames=[
            "step",
            "episode",
            "shaped_return",
            "raw_return",
            "brick_hits",
            "paddle_hits",
            "life_losses",
            "last_middle_position_penalty",
            "length",
            "epsilon",
            "loss",
        ],
    )
    logger.writeheader()

    observation, _ = env.reset(seed=args.seed)
    observation = as_observation_array(observation)
    episode_count = 0
    last_loss = np.nan
    epsilon_duration = int(args.exploration_fraction * args.total_timesteps)

    mode_text = "default" if args.mode is None else str(args.mode)
    difficulty_text = "default" if args.difficulty is None else str(args.difficulty)
    print(
        f"Training {args.env_id} shaped DQN on {device}. "
        f"mode={mode_text} difficulty={difficulty_text}. Logs: {log_path}"
    )
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
                raw_episode_return = info.get("raw_episode_return", np.nan)
                logger.writerow(
                    {
                        "step": global_step,
                        "episode": episode_count,
                        "shaped_return": episode_return,
                        "raw_return": raw_episode_return,
                        "brick_hits": info.get("brick_hits", ""),
                        "paddle_hits": info.get("paddle_hits", ""),
                        "life_losses": info.get("life_losses", ""),
                        "last_middle_position_penalty": info.get(
                            "middle_position_penalty",
                            "",
                        ),
                        "length": episode_length,
                        "epsilon": epsilon,
                        "loss": last_loss,
                    }
                )
                log_file.flush()
                print(
                    "step={step} episode={episode} shaped_return={ret:.2f} "
                    "raw_return={raw} length={length:.0f} "
                    "epsilon={eps:.3f} loss={loss:.4f}".format(
                        step=global_step,
                        episode=episode_count,
                        ret=episode_return,
                        raw=raw_episode_return,
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
                    run_dir / f"breakout_shaped_step_{global_step}.pt",
                    policy_net,
                    optimizer,
                    args,
                    global_step,
                )

    finally:
        final_path = run_dir / "breakout_shaped_final.pt"
        save_checkpoint(final_path, policy_net, optimizer, args, global_step)
        log_file.close()
        env.close()

    print(f"Finished. Final checkpoint: {final_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train shaped-reward DQN on Atari Breakout.")
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--total-timesteps", type=int, default=600_000)
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
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cuda:1, cpu, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None, help="CUDA GPU index. Overrides --device.")
    parser.add_argument("--run-dir", default="runs_breakout_shaped")
    parser.add_argument("--save-frequency", type=int, default=100_000)

    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument("--no-clip-rewards", dest="clip_rewards", action="store_false")
    parser.set_defaults(clip_rewards=True)

    parser.add_argument("--no-reward-shaping", dest="reward_shaping", action="store_false")
    parser.add_argument("--brick-reward", type=float, default=2.0)
    parser.add_argument("--paddle-hit-reward", type=float, default=1.0)
    parser.add_argument("--life-loss-penalty", type=float, default=-20.0)
    parser.add_argument("--miss-distance-penalty", type=float, default=-2.0)
    parser.add_argument("--middle-position-penalty", type=float, default=0.5)
    parser.set_defaults(reward_shaping=True)

    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--render-mode", default=None)

    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-epsilon", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device, args.gpu_id)
    if args.eval_only:
        evaluate(args, device)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
