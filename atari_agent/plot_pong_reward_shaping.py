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
class TrainingRun:
    run_id: str
    planned_steps: int
    log_path: Path
    buffer_size: int | None
    entirely_on_buffer: bool


@dataclass(frozen=True)
class TrainingSeries:
    title: str
    metric_label: str
    runs: list[TrainingRun]
    steps: np.ndarray
    returns: np.ndarray


def parse_section_runs(
    note_path: Path,
    runs_dir: Path,
    section_name: str,
    selected_buffer_size: int,
) -> list[TrainingRun]:
    in_section = False
    entries: list[TrainingRun] = []

    for line in note_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.casefold() == section_name.casefold():
            in_section = True
            continue
        if not in_section or not text:
            continue

        match = RUN_PATTERN.search(text)
        if match is None:
            if entries:
                break
            continue

        buffer_match = BUFFER_PATTERN.search(text)
        buffer_size = (
            int(buffer_match.group("buffer_size").replace("_", ""))
            if buffer_match is not None
            else None
        )
        run_id = match.group("run_id")
        entries.append(
            TrainingRun(
                run_id=run_id,
                planned_steps=int(match.group("steps").replace("_", "")),
                log_path=runs_dir / f"Pong-v5_{run_id}" / "training_log.csv",
                buffer_size=buffer_size,
                entirely_on_buffer="entirely" in text.casefold(),
            )
        )

    if not entries:
        raise ValueError(f"No runs were found in the {section_name!r} section of {note_path}.")

    runs = [
        run
        for run in entries
        if (
            (run.buffer_size is None or run.buffer_size == selected_buffer_size)
            and not run.entirely_on_buffer
        )
    ]
    if not runs:
        raise ValueError(
            f"No {section_name!r} series using replay buffer "
            f"{selected_buffer_size:,} was found in {note_path}."
        )

    for run in runs:
        if not run.log_path.exists():
            raise FileNotFoundError(
                f"Run {run.run_id} is listed in {note_path}, but its log was not "
                f"found at {run.log_path}."
            )
    return runs


def load_episode_returns(runs: list[TrainingRun]) -> tuple[np.ndarray, np.ndarray]:
    return_by_step: dict[int, float] = {}
    for run in runs:
        with run.log_path.open(newline="", encoding="utf-8") as log_file:
            reader = csv.DictReader(log_file)
            if reader.fieldnames is None or not {"step", "return"}.issubset(
                reader.fieldnames
            ):
                raise ValueError(
                    f"Training log must contain step and return columns: {run.log_path}"
                )
            for row in reader:
                return_by_step[int(float(row["step"]))] = float(row["return"])

    steps = np.array(sorted(return_by_step), dtype=np.int64)
    returns = np.array([return_by_step[step] for step in steps], dtype=np.float32)
    if steps.size == 0:
        raise ValueError("The selected logs contain no episode records.")
    return steps, returns


def load_series(
    note_path: Path,
    runs_dir: Path,
    section_name: str,
    title: str,
    metric_label: str,
    selected_buffer_size: int,
) -> TrainingSeries:
    runs = parse_section_runs(note_path, runs_dir, section_name, selected_buffer_size)
    steps, returns = load_episode_returns(runs)
    return TrainingSeries(title, metric_label, runs, steps, returns)


def moving_average(values: np.ndarray, window: int) -> tuple[np.ndarray, int]:
    actual_window = min(window, len(values))
    weights = np.ones(actual_window, dtype=np.float32) / actual_window
    return np.convolve(values, weights, mode="valid"), actual_window


def plot_comparison(
    baseline: TrainingSeries,
    shaped: TrainingSeries,
    output_path: Path,
    window: int,
    buffer_size: int,
    show: bool,
) -> dict[str, float]:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=150, sharex=True)
    colors = ["#3975a8", "#c4493d"]
    summaries: dict[str, float] = {}
    max_step = 0

    for ax, series, color in zip(axes, (baseline, shaped), colors):
        average, actual_window = moving_average(series.returns, window)
        final_average = float(np.mean(series.returns[-actual_window:]))
        summaries[series.title] = final_average
        max_step = max(max_step, int(series.steps[-1]), series.runs[-1].planned_steps)

        ax.scatter(
            series.steps,
            series.returns,
            color=color,
            alpha=0.18,
            s=14,
            edgecolors="none",
            label="Episode return",
        )
        ax.plot(
            series.steps[actual_window - 1 :],
            average,
            color=color,
            linewidth=2.4,
            label=f"{actual_window}-episode moving average (final {final_average:.2f})",
        )
        for index, run in enumerate(series.runs[:-1]):
            ax.axvline(
                run.planned_steps,
                color="#555555",
                alpha=0.2,
                linestyle="--",
                linewidth=0.9,
                label="Resume checkpoint" if index == 0 else None,
            )

        ax.set_title(series.title, fontsize=12)
        ax.set_ylabel(series.metric_label)
        ax.legend(loc="upper left")

    axes[-1].set_xlabel("Training step")
    axes[-1].set_xlim(0, max_step * 1.02)
    fig.suptitle(
        "Pong DQN: Baseline vs Heuristic Shaped-Reward Training",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Aligned runs from note.txt; the baseline panel reports Pong score while the "
        "shaped-reward panel reports the training reward, so their y-values are not "
        f"directly comparable. Selected continuation buffer: {buffer_size:,}.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export aligned Pong learning curves for baseline and heuristic "
            "shaped-reward runs listed in note.txt."
        )
    )
    parser.add_argument("--note", type=Path, default=PROJECT_ROOT / "note.txt")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "visualizations"
            / "pong_baseline_vs_shaped_reward.png"
        ),
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10_000,
        help="Select the matching replay-buffer continuation (default: 10000).",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.window <= 0:
        parser.error("--window must be greater than zero.")
    if args.buffer_size <= 0:
        parser.error("--buffer-size must be greater than zero.")
    return args


def main() -> None:
    args = parse_args()
    baseline = load_series(
        args.note,
        args.runs_dir,
        "baseline",
        "Baseline agent (original reward)",
        "Episode score",
        args.buffer_size,
    )
    shaped = load_series(
        args.note,
        args.runs_dir,
        "enhanced reward/penalty",
        "Agent trained with heuristic shaped reward",
        "Shaped episode return",
        args.buffer_size,
    )
    summaries = plot_comparison(
        baseline,
        shaped,
        args.output,
        args.window,
        args.buffer_size,
        args.show,
    )

    print(
        f"Loaded baseline episodes: {len(baseline.returns)} from "
        f"{len(baseline.runs)} logs."
    )
    print(
        f"Loaded shaped-reward episodes: {len(shaped.returns)} from "
        f"{len(shaped.runs)} logs."
    )
    for name, final_average in summaries.items():
        print(f"{name}: last {args.window}-episode average return {final_average:.2f}")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
