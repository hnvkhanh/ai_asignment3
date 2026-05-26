from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class BreakoutSeries:
    label: str
    log_path: Path
    steps: np.ndarray
    raw_returns: np.ndarray
    shaped_returns: np.ndarray | None = None


def latest_log(parent_dir: Path, pattern: str, suffix: Path) -> Path:
    candidates = sorted(
        (
            directory / suffix
            for directory in parent_dir.glob(pattern)
            if (directory / suffix).exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No training log matching {pattern}/{suffix} was found in {parent_dir}."
        )
    return candidates[0]


def load_series(
    log_path: Path,
    label: str,
    raw_return_column: str,
    shaped_return_column: str | None = None,
) -> BreakoutSeries:
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found: {log_path}")

    required_columns = {"step", raw_return_column}
    if shaped_return_column is not None:
        required_columns.add(shaped_return_column)

    with log_path.open(newline="", encoding="utf-8") as log_file:
        reader = csv.DictReader(log_file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or []))
            raise ValueError(
                f"{log_path} is missing required columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"Training log contains no episode results: {log_path}")

    shaped_returns = (
        np.array([float(row[shaped_return_column]) for row in rows], dtype=np.float32)
        if shaped_return_column is not None
        else None
    )
    return BreakoutSeries(
        label=label,
        log_path=log_path,
        steps=np.array([int(row["step"]) for row in rows], dtype=np.int64),
        raw_returns=np.array(
            [float(row[raw_return_column]) for row in rows],
            dtype=np.float32,
        ),
        shaped_returns=shaped_returns,
    )


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    actual_window = min(window, len(values))
    weights = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(values, weights, mode="valid"), actual_window


def plot_comparison(
    baseline: BreakoutSeries,
    shaped: BreakoutSeries,
    output_path: Path,
    window: int,
    show: bool,
) -> dict[str, float]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#3975a8", "shaped": "#c4493d"}
    baseline_average, baseline_window = moving_average(baseline.raw_returns, window)
    shaped_raw_average, shaped_window = moving_average(shaped.raw_returns, window)
    if shaped.shaped_returns is None:
        raise ValueError("The shaped-reward series must contain shaped episode returns.")
    shaped_objective_average, objective_window = moving_average(
        shaped.shaped_returns, window
    )
    final_raw_scores = {
        baseline.label: float(np.mean(baseline.raw_returns[-baseline_window:])),
        shaped.label: float(np.mean(shaped.raw_returns[-shaped_window:])),
    }
    final_shaped_return = float(np.mean(shaped.shaped_returns[-objective_window:]))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 9),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].scatter(
        baseline.steps,
        baseline.raw_returns,
        color=colors["baseline"],
        s=12,
        alpha=0.1,
        edgecolors="none",
    )
    axes[0].plot(
        baseline.steps[baseline_window - 1 :],
        baseline_average,
        color=colors["baseline"],
        linewidth=2.4,
        label=(
            f"{baseline.label} "
            f"(final avg {final_raw_scores[baseline.label]:.2f})"
        ),
    )
    axes[0].scatter(
        shaped.steps,
        shaped.raw_returns,
        color=colors["shaped"],
        s=12,
        alpha=0.1,
        edgecolors="none",
    )
    axes[0].plot(
        shaped.steps[shaped_window - 1 :],
        shaped_raw_average,
        color=colors["shaped"],
        linewidth=2.4,
        label=f"{shaped.label} (final avg {final_raw_scores[shaped.label]:.2f})",
    )
    axes[0].set_title(
        f"Comparable Game Performance ({window}-Episode Moving Average)",
        fontsize=12,
    )
    axes[0].set_ylabel("Raw Breakout score")
    axes[0].legend(loc="upper left")

    axes[1].scatter(
        shaped.steps,
        shaped.shaped_returns,
        color="#e07a36",
        s=12,
        alpha=0.13,
        edgecolors="none",
    )
    axes[1].plot(
        shaped.steps[objective_window - 1 :],
        shaped_objective_average,
        color="#e07a36",
        linewidth=2.3,
        label=f"Heuristic training return (final avg {final_shaped_return:.2f})",
    )
    axes[1].set_title("Heuristic Training Objective (Separate Reward Scale)", fontsize=12)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Shaped return")
    axes[1].legend(loc="upper left")

    max_step = max(int(baseline.steps[-1]), int(shaped.steps[-1]))
    axes[1].set_xlim(0, max_step * 1.02)
    fig.suptitle(
        "Breakout DQN: Baseline vs Heuristic Shaped-Reward Training",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Top panel compares raw Breakout score at frame skip 4; bottom panel "
        "shows only the shaped agent's optimization reward.",
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
    final_raw_scores["Heuristic training return"] = final_shaped_return
    return final_raw_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the Breakout frame-skip-4 baseline against a heuristic "
            "shaped-reward run using comparable raw episode scores."
        )
    )
    parser.add_argument(
        "--baseline-experiment",
        type=Path,
        help="Directory containing frame_skip_4/training_log.csv.",
    )
    parser.add_argument(
        "--baseline-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_breakout",
        help="Directory searched for the latest baseline experiment.",
    )
    parser.add_argument(
        "--shaped-run",
        type=Path,
        help="Directory containing a shaped-reward training_log.csv.",
    )
    parser.add_argument(
        "--shaped-runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs_breakout_shaped",
        help="Directory searched for the latest shaped-reward run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "visualizations"
            / "breakout_baseline_vs_shaped_reward.png"
        ),
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    baseline_log = (
        args.baseline_experiment / "frame_skip_4" / "training_log.csv"
        if args.baseline_experiment is not None
        else latest_log(
            args.baseline_runs_dir,
            "Breakout_frame_skip_*",
            Path("frame_skip_4") / "training_log.csv",
        )
    )
    shaped_log = (
        args.shaped_run / "training_log.csv"
        if args.shaped_run is not None
        else latest_log(
            args.shaped_runs_dir,
            "Breakout-v5_shaped_*",
            Path("training_log.csv"),
        )
    )

    baseline = load_series(baseline_log, "Baseline", "return")
    shaped = load_series(
        shaped_log,
        "Heuristic shaped-reward",
        "raw_return",
        "shaped_return",
    )
    final_scores = plot_comparison(
        baseline,
        shaped,
        args.output,
        args.window,
        args.show,
    )

    print(f"Baseline log: {baseline.log_path}")
    print(f"Shaped-reward log: {shaped.log_path}")
    print(f"Loaded baseline episodes: {len(baseline.raw_returns)}")
    print(f"Loaded shaped-reward episodes: {len(shaped.raw_returns)}")
    for label, score in final_scores.items():
        print(f"{label}: last {args.window}-episode average {score:.2f}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
