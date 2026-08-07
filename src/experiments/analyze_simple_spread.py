from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


DEFAULT_ALGORITHMS = (
    "maddpg",
    "independent_ddpg",
)


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file does not exist: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return list(
            csv.DictReader(file)
        )


def sample_standard_deviation(
    values: Sequence[float],
) -> float:
    """Return sample standard deviation or zero."""

    if len(values) < 2:
        return 0.0

    return float(
        stdev(values)
    )


def create_comparison_summary(
    results_dir: Path,
    output_path: Path,
    algorithms: Sequence[str],
) -> None:
    """Create one statistical summary row per algorithm."""

    fieldnames = [
        "algorithm",
        "num_seeds",
        "mean_training_return",
        "std_training_return",
        "mean_evaluation_return",
        "std_evaluation_return",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for algorithm in algorithms:
            summary_path = (
                results_dir
                / algorithm
                / "summary.csv"
            )

            rows = read_csv_rows(
                summary_path
            )

            if not rows:
                raise ValueError(
                    f"No result rows found in {summary_path}."
                )

            training_returns = [
                float(
                    row["mean_training_return"]
                )
                for row in rows
            ]

            evaluation_returns = [
                float(
                    row["mean_evaluation_return"]
                )
                for row in rows
                if row["mean_evaluation_return"]
            ]

            if not evaluation_returns:
                raise ValueError(
                    "No evaluation returns found for "
                    f"{algorithm}."
                )

            writer.writerow(
                {
                    "algorithm": algorithm,
                    "num_seeds": len(rows),
                    "mean_training_return": fmean(
                        training_returns
                    ),
                    "std_training_return": (
                        sample_standard_deviation(
                            training_returns
                        )
                    ),
                    "mean_evaluation_return": fmean(
                        evaluation_returns
                    ),
                    "std_evaluation_return": (
                        sample_standard_deviation(
                            evaluation_returns
                        )
                    ),
                }
            )


def summarize_curve(
    source_path: Path,
    value_field: str,
) -> list[tuple[int, float, float, int]]:
    """Aggregate a curve across random seeds."""

    rows = read_csv_rows(
        source_path
    )

    grouped_values: dict[
        int,
        list[float],
    ] = defaultdict(list)

    for row in rows:
        episode = int(
            row["episode"]
        )

        grouped_values[episode].append(
            float(
                row[value_field]
            )
        )

    if not grouped_values:
        raise ValueError(
            f"No curve rows found in {source_path}."
        )

    return [
        (
            episode,
            float(
                fmean(values)
            ),
            sample_standard_deviation(
                values
            ),
            len(values),
        )
        for episode, values in sorted(
            grouped_values.items()
        )
    ]


def create_curve_summary(
    results_dir: Path,
    output_path: Path,
    algorithms: Sequence[str],
    source_filename: str,
    value_field: str,
) -> None:
    """Write mean and standard deviation curves."""

    fieldnames = [
        "algorithm",
        "episode",
        "mean_return",
        "std_return",
        "num_seeds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for algorithm in algorithms:
            source_path = (
                results_dir
                / algorithm
                / source_filename
            )

            curve = summarize_curve(
                source_path=source_path,
                value_field=value_field,
            )

            for (
                episode,
                mean_return,
                std_return,
                num_seeds,
            ) in curve:
                writer.writerow(
                    {
                        "algorithm": algorithm,
                        "episode": episode,
                        "mean_return": mean_return,
                        "std_return": std_return,
                        "num_seeds": num_seeds,
                    }
                )


def plot_curve_summary(
    summary_path: Path,
    output_path: Path,
    title: str,
) -> None:
    """Create a mean curve with one-standard-deviation bands."""

    rows = read_csv_rows(
        summary_path
    )

    grouped_rows: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        grouped_rows[
            row["algorithm"]
        ].append(row)

    figure, axes = plt.subplots(
        figsize=(8, 5)
    )

    for algorithm, algorithm_rows in (
        grouped_rows.items()
    ):
        ordered_rows = sorted(
            algorithm_rows,
            key=lambda row: int(
                row["episode"]
            ),
        )

        episodes = [
            int(row["episode"])
            for row in ordered_rows
        ]

        means = [
            float(row["mean_return"])
            for row in ordered_rows
        ]

        standard_deviations = [
            float(row["std_return"])
            for row in ordered_rows
        ]

        lower_bounds = [
            mean - deviation
            for mean, deviation in zip(
                means,
                standard_deviations,
                strict=True,
            )
        ]

        upper_bounds = [
            mean + deviation
            for mean, deviation in zip(
                means,
                standard_deviations,
                strict=True,
            )
        ]

        line = axes.plot(
            episodes,
            means,
            label=algorithm,
        )[0]

        axes.fill_between(
            episodes,
            lower_bounds,
            upper_bounds,
            color=line.get_color(),
            alpha=0.2,
        )

    axes.set_title(title)
    axes.set_xlabel(
        "Training episode"
    )
    axes.set_ylabel(
        "Mean agent return"
    )
    axes.grid(
        True,
        alpha=0.3,
    )
    axes.legend()
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
    )

    plt.close(figure)


def analyze_simple_spread_results(
    results_dir: str | Path,
    output_dir: str | Path,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
) -> tuple[Path, ...]:
    """Generate statistical summaries and comparison plots."""

    source_directory = Path(
        results_dir
    )

    analysis_directory = Path(
        output_dir
    )

    analysis_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_path = (
        analysis_directory
        / "comparison_summary.csv"
    )

    training_summary_path = (
        analysis_directory
        / "training_curve_summary.csv"
    )

    evaluation_summary_path = (
        analysis_directory
        / "evaluation_curve_summary.csv"
    )

    training_plot_path = (
        analysis_directory
        / "training_curves.png"
    )

    evaluation_plot_path = (
        analysis_directory
        / "evaluation_curves.png"
    )

    create_comparison_summary(
        results_dir=source_directory,
        output_path=comparison_path,
        algorithms=algorithms,
    )

    create_curve_summary(
        results_dir=source_directory,
        output_path=training_summary_path,
        algorithms=algorithms,
        source_filename="training_curves.csv",
        value_field="training_return",
    )

    create_curve_summary(
        results_dir=source_directory,
        output_path=evaluation_summary_path,
        algorithms=algorithms,
        source_filename="evaluation_curves.csv",
        value_field="evaluation_return",
    )

    plot_curve_summary(
        summary_path=training_summary_path,
        output_path=training_plot_path,
        title=(
            "Simple Spread Training Returns"
        ),
    )

    plot_curve_summary(
        summary_path=evaluation_summary_path,
        output_path=evaluation_plot_path,
        title=(
            "Simple Spread Deterministic Evaluation Returns"
        ),
    )

    return (
        comparison_path,
        training_summary_path,
        evaluation_summary_path,
        training_plot_path,
        evaluation_plot_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and plot matched "
            "Simple Spread experiments."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing the MADDPG and "
            "independent-DDPG result directories."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summaries and plots.",
    )

    arguments = parser.parse_args()

    output_paths = (
        analyze_simple_spread_results(
            results_dir=arguments.results_dir,
            output_dir=arguments.output_dir,
        )
    )

    print("=== Analysis complete ===")

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
