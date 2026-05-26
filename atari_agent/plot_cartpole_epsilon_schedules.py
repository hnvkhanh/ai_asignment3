from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEDULE_ORDER = ["fast_decay", "slow_decay", "fixed_0.1", "fixed_0.2"]


@dataclass(frozen=True)
class TrainingRun:
    schedule: str
    label: str
    seed: int
    training_log: Path
    evaluation_mean: float


def latest_experiment(runs_dir: Path) -> Path:
    candidates = sorted(
        (
            directory
            for directory in runs_dir.glob("CartPole_epsilon_*")
            if (directory / "epsilon_schedule_summary.csv").exists()
        ),
        key=lambda directory: directory.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No CartPole epsilon experiment found in {runs_dir}.")
    return candidates[0]


def load_runs(experiment_dir: Path) -> list[TrainingRun]:
    summary_path = experiment_dir / "epsilon_schedule_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    runs: list[TrainingRun] = []
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            training_log = Path(row["training_log"])
            if not training_log.exists():
                raise FileNotFoundError(f"Training log not found: {training_log}")
            runs.append(
                TrainingRun(
                    schedule=row["schedule"],
                    label=row["label"],
                    seed=int(row["seed"]),
                    training_log=training_log,
                    evaluation_mean=float(row["evaluation_mean"]),
                )
            )
    if not runs:
        raise ValueError(f"No experiment results found in {summary_path}.")
    return runs


def load_training_log(log_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with log_path.open(newline="", encoding="utf-8") as log_file:
        rows = list(csv.DictReader(log_file))
    return (
        np.array([int(row["episode"]) for row in rows], dtype=np.int64),
        np.array([float(row["return"]) for row in rows], dtype=np.float32),
        np.array([float(row["epsilon"]) for row in rows], dtype=np.float32),
    )


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    actual_window = min(window, len(values))
    weights = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(values, weights, mode="valid")


def plot_results(
    runs: list[TrainingRun],
    output_path: Path,
    window: int,
    show: bool,
) -> dict[str, float]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "fast_decay": "#3975a8",
        "slow_decay": "#c4493d",
        "fixed_0.1": "#27875c",
        "fixed_0.2": "#e07a36",
    }
    present = {run.schedule for run in runs}
    schedules = [name for name in SCHEDULE_ORDER if name in present]
    schedules.extend(sorted(present.difference(schedules)))
    grouped = {name: [run for run in runs if run.schedule == name] for name in schedules}
    evaluation_averages: dict[str, float] = {}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(14, 9), dpi=150)
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    learning_axis = fig.add_subplot(grid[0, :])
    epsilon_axis = fig.add_subplot(grid[1, 0])
    evaluation_axis = fig.add_subplot(grid[1, 1])

    for schedule in schedules:
        schedule_runs = grouped[schedule]
        loaded = [load_training_log(run.training_log) for run in schedule_runs]
        shortest = min(len(returns) for _, returns, _ in loaded)
        actual_window = min(window, shortest)
        smoothed = np.stack(
            [moving_average(returns[:shortest], actual_window) for _, returns, _ in loaded]
        )
        curve_episodes = loaded[0][0][actual_window - 1 : shortest]
        mean_curve = np.mean(smoothed, axis=0)
        std_curve = np.std(smoothed, axis=0)
        label = schedule_runs[0].label
        color = colors.get(schedule)
        learning_axis.plot(
            curve_episodes,
            mean_curve,
            color=color,
            linewidth=2.3,
            label=label,
        )
        learning_axis.fill_between(
            curve_episodes,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=color,
            alpha=0.14,
        )
        epsilon_axis.plot(
            loaded[0][0],
            loaded[0][2],
            color=color,
            linewidth=2.1,
            label=label,
        )
        evaluation_averages[label] = float(
            np.mean([run.evaluation_mean for run in schedule_runs])
        )

    learning_axis.set_title(
        f"Training Return ({window}-Episode Moving Average, Shading = Seed Std)"
    )
    learning_axis.set_xlabel("Training episode")
    learning_axis.set_ylabel("Episode return")
    learning_axis.set_ylim(bottom=0)
    learning_axis.legend(loc="upper left")

    epsilon_axis.set_title("Epsilon-Greedy Schedule")
    epsilon_axis.set_xlabel("Training episode")
    epsilon_axis.set_ylabel("Epsilon")
    epsilon_axis.set_ylim(-0.02, 1.04)
    epsilon_axis.legend(fontsize=8)

    labels = [grouped[schedule][0].label for schedule in schedules]
    positions = np.arange(len(labels))
    means = [
        np.mean([run.evaluation_mean for run in grouped[schedule]])
        for schedule in schedules
    ]
    stds = [
        np.std([run.evaluation_mean for run in grouped[schedule]])
        for schedule in schedules
    ]
    evaluation_axis.bar(
        positions,
        means,
        yerr=stds,
        capsize=5,
        color=[colors.get(schedule, "#777777") for schedule in schedules],
        alpha=0.9,
    )
    evaluation_axis.set_title("Final Greedy Evaluation")
    evaluation_axis.set_ylabel("Mean episode return")
    evaluation_axis.set_xticks(positions, labels, rotation=18, ha="right")
    evaluation_axis.set_ylim(bottom=0)

    fig.suptitle("CartPole DQN: Epsilon-Greedy Schedule Comparison", fontsize=17)
    fig.text(
        0.5,
        0.01,
        "Bars show the mean final evaluation performance across training seeds; "
        "error bars show variation between seeds.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return evaluation_averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CartPole DQN epsilon-greedy schedule experiment results."
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--runs-dir", type=Path, default=PROJECT_ROOT / "runs_cartpole"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "visualizations"
        / "cartpole_epsilon_schedule_comparison.png",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir or latest_experiment(args.runs_dir)
    runs = load_runs(experiment_dir)
    averages = plot_results(runs, args.output, args.window, args.show)
    print(f"Experiment: {experiment_dir}")
    for label, average in averages.items():
        print(f"{label}: mean final greedy return {average:.2f}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
