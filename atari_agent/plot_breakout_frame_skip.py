from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class FrameSkipResult:
    frame_skip: int
    is_baseline: bool
    final_average_score: float
    best_score: float
    stability_std: float
    agent_steps_per_second: float
    atari_frames_per_second: float
    steps: np.ndarray
    atari_frames: np.ndarray
    returns: np.ndarray


def latest_experiment(runs_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in runs_dir.glob("Breakout_frame_skip_*")
            if (path / "frame_skip_summary.csv").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No Breakout frame-skip experiment containing frame_skip_summary.csv "
            f"was found in {runs_dir}."
        )
    return candidates[0]


def load_results(experiment_dir: Path) -> list[FrameSkipResult]:
    summary_path = experiment_dir / "frame_skip_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    results: list[FrameSkipResult] = []
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        for row in csv.DictReader(summary_file):
            frame_skip = int(row["frame_skip"])
            log_path = experiment_dir / f"frame_skip_{frame_skip}" / "training_log.csv"
            if not log_path.exists():
                raise FileNotFoundError(f"Training log not found: {log_path}")

            with log_path.open(newline="", encoding="utf-8") as log_file:
                episodes = list(csv.DictReader(log_file))
            if not episodes:
                raise ValueError(f"Training log contains no episodes: {log_path}")

            results.append(
                FrameSkipResult(
                    frame_skip=frame_skip,
                    is_baseline=row["is_baseline"].lower() == "true",
                    final_average_score=float(row["final_average_score"]),
                    best_score=float(row["best_score"]),
                    stability_std=float(row["training_stability_std"]),
                    agent_steps_per_second=float(row["agent_steps_per_second"]),
                    atari_frames_per_second=float(row["atari_frames_per_second"]),
                    steps=np.array([int(item["step"]) for item in episodes]),
                    atari_frames=np.array(
                        [int(item["atari_frames"]) for item in episodes]
                    ),
                    returns=np.array(
                        [float(item["return"]) for item in episodes],
                        dtype=np.float32,
                    ),
                )
            )

    if not results:
        raise ValueError(f"No frame-skip results were found in {summary_path}.")
    return sorted(results, key=lambda result: result.frame_skip)


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    actual_window = min(window, len(values))
    weights = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(values, weights, mode="valid"), actual_window


def plot_results(
    results: list[FrameSkipResult],
    output_path: Path,
    window: int,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {2: "#3975a8", 4: "#c4493d", 8: "#27875c"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle("Breakout DQN: Frame-Skip Comparison", fontsize=17, y=0.98)

    for result in results:
        average, actual_window = moving_average(result.returns, window)
        label = f"skip {result.frame_skip}"
        if result.is_baseline:
            label += " (baseline)"
        color = colors.get(result.frame_skip)
        axes[0, 0].plot(
            result.steps[actual_window - 1 :],
            average,
            label=label,
            color=color,
            linewidth=2,
        )
        axes[0, 1].plot(
            result.atari_frames[actual_window - 1 :],
            average,
            label=label,
            color=color,
            linewidth=2,
        )

    axes[0, 0].set_title(f"Learning Curve by Agent Steps ({window}-Episode Average)")
    axes[0, 0].set_xlabel("Agent training step")
    axes[0, 0].set_ylabel("Episode score")
    axes[0, 0].legend()

    axes[0, 1].set_title(f"Learning Curve by Atari Frames ({window}-Episode Average)")
    axes[0, 1].set_xlabel("Estimated Atari frames")
    axes[0, 1].set_ylabel("Episode score")
    axes[0, 1].legend()

    labels = [
        f"{result.frame_skip}\n(baseline)" if result.is_baseline else str(result.frame_skip)
        for result in results
    ]
    positions = np.arange(len(results))
    bar_colors = [colors.get(result.frame_skip, "#555555") for result in results]
    final_averages = [result.final_average_score for result in results]
    best_scores = [result.best_score for result in results]
    stability = [result.stability_std for result in results]
    axes[1, 0].bar(
        positions - 0.18,
        final_averages,
        width=0.36,
        yerr=stability,
        capsize=5,
        color=bar_colors,
        alpha=0.9,
        label="Final average score +/- stability std",
    )
    axes[1, 0].bar(
        positions + 0.18,
        best_scores,
        width=0.36,
        color=bar_colors,
        alpha=0.38,
        label="Best score",
    )
    axes[1, 0].set_title("Final Performance and Stability")
    axes[1, 0].set_xlabel("Frame skip")
    axes[1, 0].set_ylabel("Episode score")
    axes[1, 0].set_xticks(positions, labels)
    axes[1, 0].legend()

    agent_speed = [result.agent_steps_per_second for result in results]
    frame_speed = [result.atari_frames_per_second for result in results]
    axes[1, 1].bar(
        positions - 0.18,
        agent_speed,
        width=0.36,
        color="#4878a8",
        label="Agent steps / second",
    )
    axes[1, 1].bar(
        positions + 0.18,
        frame_speed,
        width=0.36,
        color="#e07a36",
        label="Atari frames / second",
    )
    axes[1, 1].set_title("Training Throughput")
    axes[1, 1].set_xlabel("Frame skip")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.28)
        ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.02,
        "Error bars show the final-return standard deviation from frame_skip_summary.csv; "
        "learning curves use raw training returns.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export comparison plots for the Breakout frame-skip experiment."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        help="Experiment directory containing frame_skip_summary.csv and frame_skip_* logs.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_breakout",
        help="Directory searched for the most recent experiment if --experiment-dir is omitted.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir or latest_experiment(args.runs_dir)
    output_path = args.output or experiment_dir / "frame_skip_comparison_plot.png"
    results = load_results(experiment_dir)
    plot_results(results, output_path, args.window, args.show)

    print(f"Loaded frame skips: {', '.join(str(result.frame_skip) for result in results)}")
    for result in results:
        baseline = " baseline" if result.is_baseline else ""
        print(
            f"frame_skip={result.frame_skip}{baseline}: "
            f"final_average={result.final_average_score:.2f}, "
            f"best={result.best_score:.2f}, "
            f"stability_std={result.stability_std:.2f}"
        )
    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
