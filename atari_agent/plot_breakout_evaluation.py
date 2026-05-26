from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "evaluations" / "breakout_checkpoint_comparison"


@dataclass(frozen=True)
class EvaluationSummary:
    agent: str
    training_step: int
    frame_skip: int
    episodes: int
    mean_score: float
    std_score: float


def load_summaries(summary_path: Path) -> list[EvaluationSummary]:
    if not summary_path.exists():
        raise FileNotFoundError(f"Evaluation summary not found: {summary_path}")

    required_columns = {
        "agent",
        "training_step",
        "frame_skip",
        "episodes",
        "mean_score",
        "std_score",
    }
    summaries: list[EvaluationSummary] = []
    with summary_path.open(newline="", encoding="utf-8") as summary_file:
        reader = csv.DictReader(summary_file)
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            missing = sorted(required_columns.difference(reader.fieldnames or []))
            raise ValueError(
                f"Evaluation summary is missing required columns: {', '.join(missing)}"
            )
        for row in reader:
            summaries.append(
                EvaluationSummary(
                    agent=row["agent"],
                    training_step=int(row["training_step"]),
                    frame_skip=int(row["frame_skip"]),
                    episodes=int(row["episodes"]),
                    mean_score=float(row["mean_score"]),
                    std_score=float(row["std_score"]),
                )
            )

    if not summaries:
        raise ValueError(f"Evaluation summary contains no results: {summary_path}")
    return summaries


def step_label(step: int) -> str:
    return f"{step / 1_000:g}K"


def plot_summaries(
    summaries: list[EvaluationSummary],
    output_path: Path,
    show: bool,
) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agents = ["Baseline", "Heuristic shaped-reward"]
    colors = {"Baseline": "#3975a8", "Heuristic shaped-reward": "#c4493d"}
    steps = sorted({summary.training_step for summary in summaries})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    for agent in agents:
        series = sorted(
            (summary for summary in summaries if summary.agent == agent),
            key=lambda summary: summary.training_step,
        )
        if not series:
            continue
        ax.errorbar(
            [summary.training_step for summary in series],
            [summary.mean_score for summary in series],
            yerr=[summary.std_score for summary in series],
            color=colors[agent],
            marker="o",
            markersize=6,
            linewidth=2.3,
            capsize=5,
            label=f"{agent} (mean +/- std)",
        )

    episode_counts = {summary.episodes for summary in summaries}
    frame_skips = {summary.frame_skip for summary in summaries}
    episodes_text = str(episode_counts.pop()) if len(episode_counts) == 1 else "varied"
    frame_skip_text = str(frame_skips.pop()) if len(frame_skips) == 1 else "varied"

    ax.set_title("Breakout Evaluation: Baseline vs Heuristic Shaped Reward", fontsize=15)
    ax.set_xlabel("Training checkpoint")
    ax.set_ylabel("Original Breakout episode score")
    ax.set_xticks(steps, [step_label(step) for step in steps])
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.text(
        0.5,
        0.02,
        f"Error bars show standard deviation over {episodes_text} episodes; "
        f"both policies evaluated with frame skip {frame_skip_text} and raw game rewards.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot exported Breakout checkpoint evaluation results from CSV."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "evaluation_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "evaluation_comparison.png",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.summary)
    plot_summaries(summaries, args.output, args.show)

    for summary in summaries:
        print(
            f"{summary.agent}: step={summary.training_step:,} "
            f"mean_score={summary.mean_score:.2f} std={summary.std_score:.2f}"
        )
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
