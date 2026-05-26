from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STEPS = (100_000, 300_000, 600_000, 1_200_000, 1_800_000)
RUN_PATTERN = re.compile(r"(?P<run_id>\d{8}_\d{6})\s*\((?P<steps>[\d_]+)")
BUFFER_PATTERN = re.compile(r"buffer\s+(?P<buffer_size>[\d_]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ListedRun:
    run_id: str
    planned_steps: int
    run_dir: Path
    buffer_size: int | None
    entirely_on_buffer: bool


@dataclass(frozen=True)
class CheckpointSpec:
    agent: str
    training_step: int
    run_id: str
    checkpoint_path: Path


@dataclass(frozen=True)
class EvaluationResult:
    spec: CheckpointSpec
    returns: tuple[float, ...]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns))

    @property
    def std_return(self) -> float:
        return float(np.std(self.returns))


def parse_runs(note_path: Path, runs_dir: Path, section_name: str) -> list[ListedRun]:
    in_section = False
    runs: list[ListedRun] = []

    for line in note_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.casefold() == section_name.casefold():
            in_section = True
            continue
        if not in_section or not text:
            continue

        match = RUN_PATTERN.search(text)
        if match is None:
            if runs:
                break
            continue

        buffer_match = BUFFER_PATTERN.search(text)
        buffer_size = (
            int(buffer_match.group("buffer_size").replace("_", ""))
            if buffer_match is not None
            else None
        )
        run_id = match.group("run_id")
        runs.append(
            ListedRun(
                run_id=run_id,
                planned_steps=int(match.group("steps").replace("_", "")),
                run_dir=runs_dir / f"Pong-v5_{run_id}",
                buffer_size=buffer_size,
                entirely_on_buffer="entirely" in text.casefold(),
            )
        )

    if not runs:
        raise ValueError(f"No runs were found in section {section_name!r} of {note_path}.")
    return runs


def mixed_buffer_chain(
    note_path: Path,
    runs_dir: Path,
    section_name: str,
    continuation_buffer_size: int,
) -> list[ListedRun]:
    runs = parse_runs(note_path, runs_dir, section_name)
    chain = [
        run
        for run in runs
        if (
            not run.entirely_on_buffer
            and (
                run.buffer_size is None
                or run.buffer_size == continuation_buffer_size
            )
        )
    ]
    if not any(run.buffer_size == continuation_buffer_size for run in chain):
        raise ValueError(
            f"Section {section_name!r} does not contain a continuation annotated "
            f"with buffer {continuation_buffer_size:,}."
        )
    return chain


def resolve_checkpoints(
    note_path: Path,
    runs_dir: Path,
    section_name: str,
    agent: str,
    steps: tuple[int, ...],
    continuation_buffer_size: int,
) -> list[CheckpointSpec]:
    chain = mixed_buffer_chain(
        note_path, runs_dir, section_name, continuation_buffer_size
    )
    specs: list[CheckpointSpec] = []

    for step in steps:
        matches = [
            (run, run.run_dir / f"dqn_step_{step}.pt")
            for run in chain
            if (run.run_dir / f"dqn_step_{step}.pt").exists()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one checkpoint for {agent} at {step:,} steps in the "
                f"selected mixed-buffer chain, but found {len(matches)}."
            )
        run, checkpoint_path = matches[0]
        specs.append(
            CheckpointSpec(
                agent=agent,
                training_step=step,
                run_id=run.run_id,
                checkpoint_path=checkpoint_path,
            )
        )
    return specs


def make_evaluation_env_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        env_id=args.env_id,
        capture_video=False,
        render_mode=None,
        full_action_space=args.full_action_space,
        noop_max=args.noop_max,
        frame_skip=args.frame_skip,
        screen_size=args.screen_size,
        terminal_on_life_loss=args.terminal_on_life_loss,
        frame_stack=args.frame_stack,
        seed=args.seed,
        pong_reward_shaping=False,
        clip_rewards=False,
    )


def evaluate_checkpoint(
    spec: CheckpointSpec,
    args: argparse.Namespace,
    device: object,
) -> EvaluationResult:
    import torch

    from .train_dqn import as_observation_array, load_model, make_env

    env_args = make_evaluation_env_args(args)
    env = make_env(env_args, args.output_dir, eval_mode=True)
    model = load_model(spec.checkpoint_path, env, device)
    rng = random.Random(args.seed)
    returns: list[float] = []

    try:
        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset(seed=args.seed + episode)
            done = False
            total_reward = 0.0

            while not done:
                if rng.random() < args.eval_epsilon:
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
    finally:
        env.close()

    return EvaluationResult(spec=spec, returns=tuple(returns))


def write_results(
    results: list[EvaluationResult],
    output_dir: Path,
    continuation_buffer_size: int,
    eval_epsilon: float,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.csv"
    episodes_path = output_dir / "evaluation_episodes.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "agent",
                "training_step",
                "run_id",
                "checkpoint",
                "continuation_buffer_size",
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
                    "run_id": result.spec.run_id,
                    "checkpoint": result.spec.checkpoint_path,
                    "continuation_buffer_size": continuation_buffer_size,
                    "episodes": len(result.returns),
                    "eval_epsilon": eval_epsilon,
                    "mean_score": result.mean_return,
                    "std_score": result.std_return,
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


def plot_results(results: list[EvaluationResult], output_path: Path, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        agent: sorted(
            (result for result in results if result.spec.agent == agent),
            key=lambda result: result.spec.training_step,
        )
        for agent in ("Baseline", "Heuristic shaped-reward")
    }
    colors = {"Baseline": "#3975a8", "Heuristic shaped-reward": "#c4493d"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for agent, agent_results in series.items():
        steps = np.array([result.spec.training_step for result in agent_results])
        means = np.array([result.mean_return for result in agent_results])
        stds = np.array([result.std_return for result in agent_results])
        ax.errorbar(
            steps,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.2,
            capsize=4,
            color=colors[agent],
            label=f"{agent} (mean +/- std)",
        )

    ax.set_title("Pong Evaluation: Mixed Replay-Buffer Training Path", fontsize=14)
    ax.set_xlabel("Training checkpoint step")
    ax.set_ylabel("Original Pong episode score")
    ax.set_ylim(-22, 22)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate baseline and heuristic shaped-reward Pong checkpoints "
            "from the mixed replay-buffer training path."
        )
    )
    parser.add_argument("--note", type=Path, default=PROJECT_ROOT / "note.txt")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluations" / "pong_mixed_replay_buffer",
    )
    parser.add_argument(
        "--continuation-buffer-size",
        type=int,
        default=20_000,
        help="Annotated final continuation to select from note.txt (default: 20000).",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.0,
        help="Random-action probability during evaluation (default: greedy policy).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--env-id", default="ALE/Pong-v5")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--screen-size", type=int, default=84)
    parser.add_argument("--noop-max", type=int, default=30)
    parser.add_argument("--full-action-space", action="store_true")
    parser.add_argument("--terminal-on-life-loss", action="store_true")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also export evaluation_comparison.png (requires matplotlib).",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved checkpoint paths without loading Atari or running episodes.",
    )
    args = parser.parse_args()
    if args.continuation_buffer_size <= 0:
        parser.error("--continuation-buffer-size must be greater than zero.")
    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero.")
    if not 0.0 <= args.eval_epsilon <= 1.0:
        parser.error("--eval-epsilon must be between zero and one.")
    if any(step <= 0 for step in args.steps):
        parser.error("--steps values must be greater than zero.")
    args.steps = tuple(dict.fromkeys(args.steps))
    return args


def main() -> None:
    args = parse_args()
    specs = resolve_checkpoints(
        args.note,
        args.runs_dir,
        "baseline",
        "Baseline",
        args.steps,
        args.continuation_buffer_size,
    )
    specs.extend(
        resolve_checkpoints(
            args.note,
            args.runs_dir,
            "enhanced reward/penalty",
            "Heuristic shaped-reward",
            args.steps,
            args.continuation_buffer_size,
        )
    )

    print(
        "Selected mixed replay-buffer continuation ending with buffer "
        f"{args.continuation_buffer_size:,}."
    )
    for spec in specs:
        print(
            f"{spec.agent}: step={spec.training_step:,} run={spec.run_id} "
            f"checkpoint={spec.checkpoint_path}"
        )
    if args.dry_run:
        return

    from .train_dqn import pick_device

    device = pick_device(args.device, args.gpu_id)
    results: list[EvaluationResult] = []
    for spec in specs:
        result = evaluate_checkpoint(spec, args, device)
        results.append(result)
        print(
            f"{spec.agent}: step={spec.training_step:,} "
            f"mean_score={result.mean_return:.2f} std={result.std_return:.2f}"
        )

    summary_path, episodes_path = write_results(
        results,
        args.output_dir,
        args.continuation_buffer_size,
        args.eval_epsilon,
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved episode returns: {episodes_path}")
    if args.plot:
        plot_path = args.output_dir / "evaluation_comparison.png"
        plot_results(results, plot_path, args.show)
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
