from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = tuple(range(100_000, 600_001, 100_000))


@dataclass(frozen=True)
class CheckpointSpec:
    agent: str
    training_step: int
    checkpoint_path: Path
    source_dir: Path


@dataclass(frozen=True)
class EvaluationResult:
    spec: CheckpointSpec
    returns: tuple[float, ...]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.returns))

    @property
    def std_score(self) -> float:
        return float(np.std(self.returns))


def latest_directory(parent_dir: Path, pattern: str, required_path: Path) -> Path:
    candidates = sorted(
        (
            directory
            for directory in parent_dir.glob(pattern)
            if (directory / required_path).exists()
        ),
        key=lambda directory: directory.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No directory matching {pattern} with {required_path} was found in "
            f"{parent_dir}."
        )
    return candidates[0]


def resolve_checkpoints(args: argparse.Namespace) -> list[CheckpointSpec]:
    baseline_experiment = args.baseline_experiment or latest_directory(
        args.baseline_runs_dir,
        "Breakout_frame_skip_*",
        Path(f"frame_skip_{args.frame_skip}")
        / f"breakout_frame_skip_{args.frame_skip}_step_{args.steps[0]}.pt",
    )
    shaped_run = args.shaped_run or latest_directory(
        args.shaped_runs_dir,
        "Breakout-v5_shaped_*",
        Path(f"breakout_shaped_step_{args.steps[0]}.pt"),
    )
    baseline_dir = baseline_experiment / f"frame_skip_{args.frame_skip}"

    specs: list[CheckpointSpec] = []
    for step in args.steps:
        specs.append(
            CheckpointSpec(
                agent="Baseline",
                training_step=step,
                checkpoint_path=(
                    baseline_dir
                    / f"breakout_frame_skip_{args.frame_skip}_step_{step}.pt"
                ),
                source_dir=baseline_dir,
            )
        )
    for step in args.steps:
        specs.append(
            CheckpointSpec(
                agent="Heuristic shaped-reward",
                training_step=step,
                checkpoint_path=shaped_run / f"breakout_shaped_step_{step}.pt",
                source_dir=shaped_run,
            )
        )

    missing = [spec.checkpoint_path for spec in specs if not spec.checkpoint_path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing Breakout checkpoints:\n{missing_text}")
    return specs


def make_evaluation_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        env_id=args.env_id,
        render_mode=None,
        capture_video=False,
        full_action_space=args.full_action_space,
        mode=args.mode,
        difficulty=args.difficulty,
        frame_stack=args.frame_stack,
        frame_skip=args.frame_skip,
        screen_size=args.screen_size,
        noop_max=args.noop_max,
        terminal_on_life_loss=args.terminal_on_life_loss,
        seed=args.seed,
        reward_shaping=False,
        clip_rewards=False,
        brick_reward=2.0,
        paddle_hit_reward=1.0,
        miss_distance_penalty=-1.0,
    )


def evaluate_checkpoint(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    device: object,
) -> EvaluationResult:
    import torch

    from .train_breakout_shaped import as_observation_array, load_model, make_env

    env = make_env(make_evaluation_env_args(args), args.output_dir, eval_mode=True)
    model = load_model(spec.checkpoint_path, env, device)
    action_rng = random.Random(args.seed)
    returns: list[float] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            done = False
            score = 0.0

            while not done:
                if action_rng.random() < args.eval_epsilon:
                    action = env.action_space.sample()
                else:
                    obs_tensor = torch.from_numpy(as_observation_array(observation)).to(
                        device, dtype=torch.float32
                    )
                    obs_tensor = obs_tensor.unsqueeze(0).div(255.0)
                    with torch.no_grad():
                        action = int(model(obs_tensor).argmax(dim=1).item())

                observation, reward, terminated, truncated, _ = env.step(action)
                score += float(reward)
                done = terminated or truncated
            returns.append(score)
    finally:
        env.close()

    return EvaluationResult(spec=spec, returns=tuple(returns))


def write_results(results: list[EvaluationResult], args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "evaluation_summary.csv"
    episodes_path = args.output_dir / "evaluation_episodes.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "agent",
                "training_step",
                "checkpoint",
                "source_dir",
                "frame_skip",
                "episodes",
                "eval_epsilon",
                "mean_score",
                "std_score",
                "min_score",
                "max_score",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "agent": result.spec.agent,
                    "training_step": result.spec.training_step,
                    "checkpoint": result.spec.checkpoint_path,
                    "source_dir": result.spec.source_dir,
                    "frame_skip": args.frame_skip,
                    "episodes": len(result.returns),
                    "eval_epsilon": args.eval_epsilon,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "min_score": min(result.returns),
                    "max_score": max(result.returns),
                }
            )

    with episodes_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["agent", "training_step", "episode", "score"],
        )
        writer.writeheader()
        for result in results:
            for episode, score in enumerate(result.returns, start=1):
                writer.writerow(
                    {
                        "agent": result.spec.agent,
                        "training_step": result.spec.training_step,
                        "episode": episode,
                        "score": score,
                    }
                )
    return summary_path, episodes_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Breakout baseline and heuristic shaped-reward checkpoints "
            "at each saved 100K training step using original game score."
        )
    )
    parser.add_argument(
        "--baseline-experiment",
        type=Path,
        help="Directory containing the baseline frame_skip_<N> checkpoint folder.",
    )
    parser.add_argument(
        "--baseline-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_breakout",
    )
    parser.add_argument(
        "--shaped-run",
        type=Path,
        help="Directory containing breakout_shaped_step_<N>.pt checkpoints.",
    )
    parser.add_argument(
        "--shaped-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_breakout_shaped",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluations" / "breakout_checkpoint_comparison",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--eval-epsilon", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--env-id", default="ALE/Breakout-v5")
    parser.add_argument("--mode", type=int, default=None)
    parser.add_argument("--difficulty", type=int, default=None)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected checkpoint paths without running Atari evaluation.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero.")
    if not 0.0 <= args.eval_epsilon <= 1.0:
        parser.error("--eval-epsilon must be between zero and one.")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be greater than zero.")
    if any(step <= 0 for step in args.steps):
        parser.error("--steps values must be greater than zero.")
    args.steps = tuple(dict.fromkeys(args.steps))
    return args


def main() -> None:
    args = parse_args()
    specs = resolve_checkpoints(args)
    for spec in specs:
        print(
            f"{spec.agent}: step={spec.training_step:,} "
            f"checkpoint={spec.checkpoint_path}"
        )
    if args.dry_run:
        return

    from .train_breakout_shaped import pick_device

    device = pick_device(args.device, args.gpu_id)
    results: list[EvaluationResult] = []
    for spec in specs:
        result = evaluate_checkpoint(spec, args, device)
        results.append(result)
        write_results(results, args)
        print(
            f"{spec.agent}: step={spec.training_step:,} "
            f"mean_score={result.mean_score:.2f} std={result.std_score:.2f}"
        )

    summary_path, episodes_path = write_results(results, args)
    print(f"Saved summary: {summary_path}")
    print(f"Saved episode returns: {episodes_path}")


if __name__ == "__main__":
    main()
