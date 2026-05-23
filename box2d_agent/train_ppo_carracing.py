from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Iterator

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal


class ChannelFirstObservation(gym.ObservationWrapper):
    """Converts CarRacing RGB observations from HWC uint8 to CHW uint8."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if len(env.observation_space.shape) != 3:
            raise ValueError(
                "Expected image observations shaped like (height, width, channels).")

        height, width, channels = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(channels, height, width),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.transpose(np.asarray(observation, dtype=np.uint8), (2, 0, 1))


class ActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, ...], action_dim: int):
        super().__init__()
        if len(observation_shape) != 3:
            raise ValueError(
                "Expected channel-first image observations shaped like (C, H, W).")

        self.observation_shape = observation_shape
        self.action_dim = action_dim
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

        self.actor_mean = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_count, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        with torch.no_grad():
            self.actor_mean[-1].bias.zero_()
            if action_dim >= 3:
                self.actor_mean[-1].bias[1] = -1.0
                self.actor_mean[-1].bias[2] = -2.0
            self.actor_log_std.fill_(-0.5)

    def _features(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dtype == torch.uint8:
            observations = observations.float().div(255.0)
        return self.features(observations)

    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        return self.critic(self._features(observations)).squeeze(-1)

    def get_action_and_value(
        self,
        observations: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(observations)
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.expand_as(mean)
        std = torch.exp(log_std)
        distribution = Normal(mean, std)

        if action is None:
            pre_tanh_action = mean if deterministic else distribution.sample()
            action = torch.tanh(pre_tanh_action)
        else:
            action = action.clamp(-0.999999, 0.999999)
            pre_tanh_action = atanh(action)

        log_prob = distribution.log_prob(pre_tanh_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        value = self.critic(features).squeeze(-1)
        return action, log_prob, entropy, value


def atanh(value: torch.Tensor) -> torch.Tensor:
    value = value.clamp(-0.999999, 0.999999)
    return 0.5 * (torch.log1p(value) - torch.log1p(-value))


def normalized_to_env_action(
    normalized_action: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0
    action = action_bias + normalized_action * action_scale
    return torch.max(torch.min(action, action_high), action_low)


def make_env(
    args: argparse.Namespace,
    run_dir: Path,
    env_index: int = 0,
    eval_mode: bool = False,
):
    def thunk() -> gym.Env:
        render_mode = args.render_mode
        if args.capture_video and not eval_mode and env_index == 0:
            render_mode = "rgb_array"

        env = gym.make(
            args.env_id,
            continuous=True,
            domain_randomize=args.domain_randomize,
            render_mode=render_mode,
        )

        if args.capture_video and not eval_mode and env_index == 0:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(run_dir / "videos"),
                episode_trigger=lambda episode_id: episode_id % args.video_every == 0,
            )

        env = ChannelFirstObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed + env_index)
        env.observation_space.seed(args.seed + env_index)
        return env

    return thunk


def pick_device(device_name: str, gpu_id: int | None = None) -> torch.device:
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise ValueError("--gpu-id was set, but CUDA is not available.")
        gpu_count = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= gpu_count:
            raise ValueError(
                f"--gpu-id must be between 0 and {gpu_count - 1}; got {gpu_id}.")
        return torch.device(f"cuda:{gpu_id}")

    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scalar(value: Any) -> float:
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def episode_stats_from_infos(infos: dict[str, Any]) -> Iterator[dict[str, Any]]:
    final_infos = infos.get("final_info")
    yielded_final_episode = False
    if final_infos is not None:
        for final_info in final_infos:
            if final_info is not None and "episode" in final_info:
                yielded_final_episode = True
                yield final_info["episode"]
    if yielded_final_episode:
        return

    episode_info = infos.get("episode")
    if not isinstance(episode_info, dict):
        return

    returns = np.asarray(episode_info.get("r", []))
    if returns.ndim == 0:
        yield episode_info
        return

    mask = np.asarray(infos.get("_episode", np.ones(len(returns), dtype=bool)))
    for index, has_episode in enumerate(mask):
        if not has_episode:
            continue
        yield {
            key: np.asarray(value)[index]
            for key, value in episode_info.items()
            if not key.startswith("_")
        }


def save_checkpoint(
    path: Path,
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    global_step: int,
    episode_count: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "env_id": args.env_id,
            "global_step": global_step,
            "episode_count": episode_count,
            "observation_shape": model.observation_shape,
            "action_dim": model.action_dim,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def load_model(
    checkpoint_path: Path,
    observation_shape: tuple[int, ...],
    action_dim: int,
    device: torch.device,
) -> ActorCritic:
    checkpoint = load_checkpoint(checkpoint_path, device)
    model = ActorCritic(observation_shape, action_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    variance = np.var(y_true)
    if variance == 0.0:
        return np.nan
    return float(1.0 - np.var(y_true - y_pred) / variance)


def train(args: argparse.Namespace, device: torch.device) -> None:
    run_name = f"{args.env_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_name = run_name.replace("/", "_")
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = args.cudnn_benchmark
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args, run_dir, env_index)
         for env_index in range(args.num_envs)]
    )
    observation_shape = tuple(envs.single_observation_space.shape)
    action_dim = int(np.prod(envs.single_action_space.shape))
    action_low = torch.as_tensor(
        envs.single_action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(
        envs.single_action_space.high, device=device, dtype=torch.float32)

    model = ActorCritic(observation_shape, action_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    episode_count = 0
    if args.resume_checkpoint is not None:
        checkpoint = load_checkpoint(Path(args.resume_checkpoint), device)
        checkpoint_observation_shape = tuple(
            checkpoint.get("observation_shape", observation_shape))
        checkpoint_action_dim = int(checkpoint.get("action_dim", action_dim))
        if checkpoint_observation_shape != observation_shape:
            raise ValueError(
                "Checkpoint observation shape "
                f"{checkpoint_observation_shape} does not match current shape {observation_shape}."
            )
        if checkpoint_action_dim != action_dim:
            raise ValueError(
                f"Checkpoint action dim {checkpoint_action_dim} does not match {action_dim}."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        episode_count = int(checkpoint.get("episode_count", 0))

    batch_size = args.num_envs * args.num_steps
    if args.total_timesteps <= global_step:
        raise ValueError(
            "--total-timesteps must be greater than the checkpoint step.")
    if batch_size <= 0:
        raise ValueError("--num-envs and --num-steps must be positive.")
    if args.minibatch_size > batch_size:
        raise ValueError("--minibatch-size must be <= num_envs * num_steps.")

    obs = torch.zeros((args.num_steps, args.num_envs, *
                      observation_shape), device=device, dtype=torch.uint8)
    actions = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), device=device)
    log_probs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    log_path = run_dir / "training_log.csv"
    log_file = log_path.open("w", newline="")
    logger = csv.DictWriter(
        log_file,
        fieldnames=[
            "step",
            "episode",
            "return",
            "length",
            "moving_average_return",
            "policy_loss",
            "value_loss",
            "entropy",
            "approx_kl",
            "explained_variance",
            "learning_rate",
            "fps",
        ],
    )
    logger.writeheader()

    next_obs_np, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device)
    episode_returns: list[float] = []
    last_policy_loss = np.nan
    last_value_loss = np.nan
    last_entropy = np.nan
    last_approx_kl = np.nan
    last_explained_variance = np.nan
    start_time = time.time()
    next_save_step = ((global_step // args.save_frequency) +
                      1) * args.save_frequency

    print(f"Training {args.env_id} with PPO on {device}. Logs: {log_path}")
    if args.resume_checkpoint is not None:
        print(f"Resuming from {args.resume_checkpoint} at step {global_step}.")

    try:
        while global_step < args.total_timesteps:
            if args.anneal_lr:
                progress = 1.0 - (global_step / args.total_timesteps)
                optimizer.param_groups[0]["lr"] = progress * args.learning_rate

            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, log_prob, _, value = model.get_action_and_value(
                        next_obs)
                    values[step] = value
                actions[step] = action
                log_probs[step] = log_prob

                env_action = normalized_to_env_action(
                    action, action_low, action_high)
                next_obs_np, reward_np, terminated_np, truncated_np, infos = envs.step(
                    env_action.cpu().numpy()
                )
                done_np = np.logical_or(terminated_np, truncated_np)
                rewards[step] = torch.as_tensor(
                    reward_np, device=device, dtype=torch.float32)
                next_obs = torch.as_tensor(
                    next_obs_np, device=device, dtype=torch.uint8)
                next_done = torch.as_tensor(
                    done_np, device=device, dtype=torch.float32)

                for episode_info in episode_stats_from_infos(infos):
                    episode_count += 1
                    episode_return = scalar(episode_info["r"])
                    episode_length = scalar(episode_info["l"])
                    episode_returns.append(episode_return)
                    moving_average_return = float(
                        np.mean(episode_returns[-args.moving_average_window:])
                    )
                    fps = int(global_step / max(1.0, time.time() - start_time))
                    logger.writerow(
                        {
                            "step": global_step,
                            "episode": episode_count,
                            "return": episode_return,
                            "length": episode_length,
                            "moving_average_return": moving_average_return,
                            "policy_loss": last_policy_loss,
                            "value_loss": last_value_loss,
                            "entropy": last_entropy,
                            "approx_kl": last_approx_kl,
                            "explained_variance": last_explained_variance,
                            "learning_rate": optimizer.param_groups[0]["lr"],
                            "fps": fps,
                        }
                    )
                    log_file.flush()
                    print(
                        "step={step} episode={episode} return={ret:.2f} "
                        "moving_avg={avg:.2f} length={length:.0f} fps={fps}".format(
                            step=global_step,
                            episode=episode_count,
                            ret=episode_return,
                            avg=moving_average_return,
                            length=episode_length,
                            fps=fps,
                        )
                    )

            with torch.no_grad():
                next_value = model.get_value(next_obs)
                advantages = torch.zeros_like(rewards)
                last_gae_lam = 0.0
                for step in reversed(range(args.num_steps)):
                    if step == args.num_steps - 1:
                        next_non_terminal = 1.0 - next_done
                        next_values = next_value
                    else:
                        next_non_terminal = 1.0 - dones[step + 1]
                        next_values = values[step + 1]
                    delta = rewards[step] + args.gamma * \
                        next_values * next_non_terminal - values[step]
                    advantages[step] = last_gae_lam = (
                        delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
                    )
                returns = advantages + values

            b_obs = obs.reshape((-1, *observation_shape))
            b_log_probs = log_probs.reshape(-1)
            b_actions = actions.reshape((-1, action_dim))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            batch_indices = np.arange(batch_size)
            clip_fractions: list[float] = []
            for _ in range(args.update_epochs):
                np.random.shuffle(batch_indices)
                for start in range(0, batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    minibatch_indices = batch_indices[start:end]

                    _, new_log_prob, entropy, new_value = model.get_action_and_value(
                        b_obs[minibatch_indices],
                        b_actions[minibatch_indices],
                    )
                    log_ratio = new_log_prob - b_log_probs[minibatch_indices]
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1.0) - log_ratio).mean()
                        clip_fractions.append(
                            ((ratio - 1.0).abs() >
                             args.clip_coef).float().mean().item()
                        )

                    mb_advantages = b_advantages[minibatch_indices]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    policy_loss_1 = -mb_advantages * ratio
                    policy_loss_2 = -mb_advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_coef,
                        1.0 + args.clip_coef,
                    )
                    policy_loss = torch.max(
                        policy_loss_1, policy_loss_2).mean()

                    new_value = new_value.view(-1)
                    if args.clip_vloss:
                        value_loss_unclipped = (
                            new_value - b_returns[minibatch_indices]).pow(2)
                        value_clipped = b_values[minibatch_indices] + torch.clamp(
                            new_value - b_values[minibatch_indices],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        value_loss_clipped = (
                            value_clipped - b_returns[minibatch_indices]).pow(2)
                        value_loss = 0.5 * torch.max(
                            value_loss_unclipped,
                            value_loss_clipped,
                        ).mean()
                    else:
                        value_loss = 0.5 * \
                            F.mse_loss(new_value, b_returns[minibatch_indices])

                    entropy_loss = entropy.mean()
                    loss = policy_loss - args.ent_coef * entropy_loss + args.vf_coef * value_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred = b_values.detach().cpu().numpy()
            y_true = b_returns.detach().cpu().numpy()
            last_policy_loss = float(policy_loss.item())
            last_value_loss = float(value_loss.item())
            last_entropy = float(entropy_loss.item())
            last_approx_kl = float(approx_kl.item())
            last_explained_variance = explained_variance(y_pred, y_true)

            if global_step >= next_save_step:
                save_checkpoint(
                    run_dir / f"ppo_carracing_step_{global_step}.pt",
                    model,
                    optimizer,
                    args,
                    global_step,
                    episode_count,
                )
                next_save_step += args.save_frequency

    finally:
        final_path = run_dir / "ppo_carracing_final.pt"
        save_checkpoint(final_path, model, optimizer,
                        args, global_step, episode_count)
        log_file.close()
        envs.close()

    print(f"Finished. Final checkpoint: {final_path}")


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required with --eval-only.")

    run_dir = Path(args.run_dir) / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(args, run_dir, eval_mode=True)()
    observation_shape = tuple(env.observation_space.shape)
    action_dim = int(np.prod(env.action_space.shape))
    action_low = torch.as_tensor(
        env.action_space.low, device=device, dtype=torch.float32)
    action_high = torch.as_tensor(
        env.action_space.high, device=device, dtype=torch.float32)
    model = load_model(Path(args.checkpoint),
                       observation_shape, action_dim, device)

    returns: list[float] = []
    for episode in range(1, args.eval_episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            obs_tensor = torch.as_tensor(
                observation, device=device, dtype=torch.uint8).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(
                    obs_tensor, deterministic=True)
            env_action = normalized_to_env_action(
                action, action_low, action_high)
            observation, reward, terminated, truncated, _ = env.step(
                env_action.squeeze(0).cpu().numpy())
            total_reward += float(reward)
            done = terminated or truncated

        returns.append(total_reward)
        print(f"eval_episode={episode} return={total_reward:.2f}")

    env.close()
    print(f"mean_eval_return={float(np.mean(returns)):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on Box2D CarRacing.")
    parser.add_argument("--env-id", default="CarRacing-v3")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--anneal-lr", dest="anneal_lr", action="store_true")
    parser.add_argument("--no-anneal-lr", dest="anneal_lr",
                        action="store_false")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--norm-adv", dest="norm_adv", action="store_true")
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", dest="clip_vloss", action="store_true")
    parser.add_argument("--no-clip-vloss",
                        dest="clip_vloss", action="store_false")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.set_defaults(anneal_lr=True, norm_adv=True, clip_vloss=True)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto",
                        help="auto, cuda, cuda:0, cuda:1, cpu, or mps.")
    parser.add_argument("--gpu-id", type=int, default=None,
                        help="CUDA GPU index. Overrides --device.")
    parser.add_argument("--no-cudnn-benchmark",
                        dest="cudnn_benchmark", action="store_false")
    parser.set_defaults(cudnn_benchmark=True)

    parser.add_argument("--run-dir", default="runs_box2d")
    parser.add_argument("--save-frequency", type=int, default=100_000)
    parser.add_argument("--moving-average-window", type=int, default=10)
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--capture-video", action="store_true")
    parser.add_argument("--video-every", type=int, default=25)
    parser.add_argument("--render-mode", default=None)

    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--eval-episodes", type=int, default=5)
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
