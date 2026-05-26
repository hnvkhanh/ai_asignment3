from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PATTERN = re.compile(r"(?P<run_id>\d{8}_\d{6})\s*\((?P<steps>[\d_]+)")
BUFFER_PATTERN = re.compile(r"buffer\s+(?P<buffer_size>[\d_]+)", re.IGNORECASE)


@dataclass(frozen=True)
class BaselineRun:
    run_id: str
    planned_steps: int
    log_path: Path
    buffer_size: int | None
    entirely_on_buffer: bool


def parse_baseline_runs(
    note_path: Path,
    runs_dir: Path,
    selected_buffer_size: int,
    prefer_entire_run: bool = True,
) -> list[BaselineRun]:
    in_baseline_section = False
    entries: list[BaselineRun] = []

    for line in note_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.lower() == "baseline":
            in_baseline_section = True
            continue
        if not in_baseline_section:
            continue
        if not text:
            continue

        match = RUN_PATTERN.search(text)
        if match is None:
            if entries:
                break
            continue

        run_id = match.group("run_id")
        buffer_match = BUFFER_PATTERN.search(text)
        buffer_size = (
            int(buffer_match.group("buffer_size").replace("_", ""))
            if buffer_match is not None
            else None
        )
        log_path = runs_dir / f"Pong-v5_{run_id}" / "training_log.csv"
        entries.append(
            BaselineRun(
                run_id=run_id,
                planned_steps=int(match.group("steps").replace("_", "")),
                log_path=log_path,
                buffer_size=buffer_size,
                entirely_on_buffer="entirely" in text.lower(),
            )
        )

    complete_runs = [
        run
        for run in entries
        if run.buffer_size == selected_buffer_size and run.entirely_on_buffer
    ]
    if complete_runs and prefer_entire_run:
        runs = [complete_runs[-1]]
    else:
        # Unannotated logs precede the split into buffer-size continuations.
        runs = [
            run
            for run in entries
            if (
                (run.buffer_size is None or run.buffer_size == selected_buffer_size)
                and not run.entirely_on_buffer
            )
        ]

    if not runs:
        raise ValueError(f"No Pong baseline runs were found in {note_path}.")
    for run in runs:
        if not run.log_path.exists():
            raise FileNotFoundError(
                f"Baseline run {run.run_id} is listed in {note_path}, "
                f"but its log was not found at {run.log_path}."
            )
    return runs


def load_episode_returns(runs: list[BaselineRun]) -> tuple[np.ndarray, np.ndarray]:
    return_by_step: dict[int, float] = {}
    for run in runs:
        with run.log_path.open(newline="", encoding="utf-8") as log_file:
            for row in csv.DictReader(log_file):
                return_by_step[int(float(row["step"]))] = float(row["return"])

    steps = np.array(sorted(return_by_step), dtype=np.int64)
    returns = np.array([return_by_step[step] for step in steps], dtype=np.float32)
    if steps.size == 0:
        raise ValueError("The selected baseline logs contain no episode records.")
    return steps, returns


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    actual_window = min(window, len(values))
    weights = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(values, weights, mode="valid"), actual_window


def plot_curve(
    runs: list[BaselineRun],
    steps: np.ndarray,
    returns: np.ndarray,
    output_path: Path,
    window: int,
    buffer_size: int,
    show: bool,
) -> float:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    averages, actual_window = moving_average(returns, window)
    average_steps = steps[actual_window - 1 :]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.scatter(
        steps,
        returns,
        color="#3975a8",
        alpha=0.25,
        s=18,
        edgecolors="none",
        label="Episode return",
    )
    ax.plot(
        average_steps,
        averages,
        color="#c4493d",
        linewidth=2.3,
        label=f"{actual_window}-episode moving average",
    )

    for index, run in enumerate(runs[:-1]):
        ax.axvline(
            run.planned_steps,
            color="#555555",
            alpha=0.26,
            linestyle="--",
            linewidth=1.0,
            label="Resume checkpoint" if index == 0 else None,
        )

    final_steps = max(runs[-1].planned_steps, int(steps[-1]))
    ax.set_xlim(0, final_steps * 1.02)
    ax.set_ylim(-22, 22)
    ax.set_title(
        f"Pong Baseline DQN Learning Curve (Replay Buffer {buffer_size:,})",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode score")
    ax.text(
        0.01,
        0.98,
        f"{len(runs)} resumed logs from note.txt | final training budget: "
        f"{runs[-1].planned_steps:,} steps | buffer: {buffer_size:,}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return float(np.mean(returns[-actual_window:]))


def plot_comparison(
    buffer_sizes: list[int],
    note_path: Path,
    runs_dir: Path,
    output_path: Path,
    window: int,
    show: bool,
) -> dict[str, float]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#3975a8", "#c4493d", "#27875c", "#e07a36"]
    final_averages: dict[str, float] = {}
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)

    max_step = 0
    for index, buffer_size in enumerate(buffer_sizes):
        runs = parse_baseline_runs(note_path, runs_dir, buffer_size)
        steps, returns = load_episode_returns(runs)
        averages, actual_window = moving_average(returns, window)
        final_average = float(np.mean(returns[-actual_window:]))
        series_name = f"Replay buffer {buffer_size:,}"
        final_averages[series_name] = final_average
        max_step = max(max_step, int(steps[-1]), runs[-1].planned_steps)
        origin = "full run" if runs[0].entirely_on_buffer else "resumed run"

        ax.plot(
            steps[actual_window - 1 :],
            averages,
            color=colors[index % len(colors)],
            linewidth=2.4,
            label=(
                f"Buffer {buffer_size:,} ({origin}, "
                f"final avg {final_average:.2f})"
            ),
        )

    switch_from, switch_to = 10_000, 20_000
    if switch_from in buffer_sizes and switch_to in buffer_sizes:
        switched_runs = parse_baseline_runs(
            note_path,
            runs_dir,
            switch_to,
            prefer_entire_run=False,
        )
        if any(run.buffer_size == switch_to for run in switched_runs):
            steps, returns = load_episode_returns(switched_runs)
            averages, actual_window = moving_average(returns, window)
            final_average = float(np.mean(returns[-actual_window:]))
            prefix_steps = [
                run.planned_steps
                for run in switched_runs
                if run.buffer_size is None
            ]
            switch_step = max(prefix_steps)
            series_name = (
                f"Replay buffer {switch_from:,} -> {switch_to:,} "
                f"at {switch_step:,} steps"
            )
            final_averages[series_name] = final_average
            max_step = max(max_step, int(steps[-1]), switched_runs[-1].planned_steps)
            ax.plot(
                steps[actual_window - 1 :],
                averages,
                color=colors[2],
                linewidth=2.4,
                label=(
                    f"Buffer {switch_from:,} -> {switch_to:,} at "
                    f"{switch_step / 1_000_000:.1f}M "
                    f"(final avg {final_average:.2f})"
                ),
            )
            ax.axvline(
                switch_step,
                color="#555555",
                linestyle="--",
                linewidth=1.1,
                alpha=0.45,
                label=f"Buffer switch at {switch_step / 1_000_000:.1f}M",
            )

    ax.set_xlim(0, max_step * 1.02)
    ax.set_ylim(-22, 22)
    ax.set_title("Pong Baseline DQN: Replay Buffer Comparison", fontsize=15, pad=12)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Episode score")
    ax.text(
        0.01,
        0.98,
        f"{window}-episode moving averages from baseline logs in note.txt",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#444444",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return final_averages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the Pong baseline learning curve from runs listed in note.txt."
    )
    parser.add_argument("--note", type=Path, default=PROJECT_ROOT / "note.txt")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "pong_baseline_learning_curve.png",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10_000,
        help="Select the replay-buffer continuation annotated in note.txt (default: 10000).",
    )
    parser.add_argument(
        "--compare-buffers",
        type=int,
        nargs="+",
        metavar="SIZE",
        help="Plot multiple annotated replay-buffer baselines on the same chart.",
    )
    parser.add_argument(
        "--comparison-output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "visualizations"
            / "pong_baseline_buffer_comparison.png"
        ),
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero.")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be greater than zero.")
    if args.compare_buffers is not None:
        if len(set(args.compare_buffers)) < 2:
            parser.error("--compare-buffers requires at least two different buffer sizes.")
        if any(size <= 0 for size in args.compare_buffers):
            parser.error("--compare-buffers sizes must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    if args.compare_buffers is not None:
        final_averages = plot_comparison(
            list(dict.fromkeys(args.compare_buffers)),
            args.note,
            args.runs_dir,
            args.comparison_output,
            args.window,
            args.show,
        )
        for series_name, final_average in final_averages.items():
            print(
                f"{series_name}: "
                f"last {args.window}-episode average score {final_average:.2f}"
            )
        print(f"Saved plot: {args.comparison_output}")
        return

    runs = parse_baseline_runs(args.note, args.runs_dir, args.buffer_size)
    steps, returns = load_episode_returns(runs)
    final_average = plot_curve(
        runs,
        steps,
        returns,
        args.output,
        args.window,
        args.buffer_size,
        args.show,
    )
    print(f"Loaded {len(returns)} episodes from {len(runs)} baseline log files.")
    print(f"Selected replay buffer continuation: {args.buffer_size:,}")
    print(f"Last {min(args.window, len(returns))}-episode average score: {final_average:.2f}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
