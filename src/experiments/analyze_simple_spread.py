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


def create_final_checkpoint_by_seed(
    results_dir: Path,
    output_path: Path,
    algorithms: Sequence[str],
) -> None:
    """Write final deterministic evaluation per seed."""

    fieldnames = [
        "algorithm",
        "seed",
        "final_evaluation_episode",
        "final_evaluation_return",
        "runtime_seconds",
        "runtime_minutes",
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

            for row in rows:
                if not row[
                    "final_evaluation_return"
                ]:
                    raise ValueError(
                        "Missing final evaluation "
                        f"return for {algorithm}, "
                        f"seed {row['seed']}."
                    )

                writer.writerow(
                    {
                        "algorithm": algorithm,
                        "seed": int(
                            row["seed"]
                        ),
                        "final_evaluation_episode": int(
                            row[
                                "final_evaluation_episode"
                            ]
                        ),
                        "final_evaluation_return": float(
                            row[
                                "final_evaluation_return"
                            ]
                        ),
                        "runtime_seconds": float(
                            row["runtime_seconds"]
                        ),
                        "runtime_minutes": float(
                            row["runtime_minutes"]
                        ),
                    }
                )


def create_final_checkpoint_summary(
    results_dir: Path,
    output_path: Path,
    algorithms: Sequence[str],
) -> None:
    """Summarize final performance and runtime."""

    fieldnames = [
        "algorithm",
        "num_seeds",
        "final_mean_return",
        "final_std_return",
        "mean_runtime_seconds",
        "std_runtime_seconds",
        "min_runtime_seconds",
        "max_runtime_seconds",
        "total_runtime_seconds",
        "mean_runtime_minutes",
        "total_runtime_minutes",
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
            rows = read_csv_rows(
                results_dir
                / algorithm
                / "summary.csv"
            )

            if not rows:
                raise ValueError(
                    f"No results found for {algorithm}."
                )

            final_returns = [
                float(
                    row[
                        "final_evaluation_return"
                    ]
                )
                for row in rows
                if row[
                    "final_evaluation_return"
                ]
            ]

            runtimes = [
                float(
                    row["runtime_seconds"]
                )
                for row in rows
            ]

            if len(final_returns) != len(rows):
                raise ValueError(
                    "Every seed must have a final "
                    f"evaluation return for {algorithm}."
                )

            mean_runtime = fmean(
                runtimes
            )

            total_runtime = sum(
                runtimes
            )

            writer.writerow(
                {
                    "algorithm": algorithm,
                    "num_seeds": len(rows),
                    "final_mean_return": fmean(
                        final_returns
                    ),
                    "final_std_return": (
                        sample_standard_deviation(
                            final_returns
                        )
                    ),
                    "mean_runtime_seconds": (
                        mean_runtime
                    ),
                    "std_runtime_seconds": (
                        sample_standard_deviation(
                            runtimes
                        )
                    ),
                    "min_runtime_seconds": min(
                        runtimes
                    ),
                    "max_runtime_seconds": max(
                        runtimes
                    ),
                    "total_runtime_seconds": (
                        total_runtime
                    ),
                    "mean_runtime_minutes": (
                        mean_runtime / 60.0
                    ),
                    "total_runtime_minutes": (
                        total_runtime / 60.0
                    ),
                }
            )


def create_paired_final_comparison(
    results_dir: Path,
    output_path: Path,
) -> None:
    """Compare final returns on matched seeds."""

    maddpg_rows = read_csv_rows(
        results_dir
        / "maddpg"
        / "summary.csv"
    )

    independent_rows = read_csv_rows(
        results_dir
        / "independent_ddpg"
        / "summary.csv"
    )

    maddpg_by_seed = {
        int(row["seed"]): row
        for row in maddpg_rows
    }

    independent_by_seed = {
        int(row["seed"]): row
        for row in independent_rows
    }

    maddpg_seeds = set(
        maddpg_by_seed
    )

    independent_seeds = set(
        independent_by_seed
    )

    if maddpg_seeds != independent_seeds:
        raise ValueError(
            "MADDPG and Independent DDPG must use "
            "the same matched seed set."
        )

    fieldnames = [
        "seed",
        "maddpg_final_return",
        "independent_ddpg_final_return",
        "maddpg_minus_independent_ddpg",
        "winner",
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

        for seed in sorted(
            maddpg_seeds
        ):
            maddpg_return = float(
                maddpg_by_seed[seed][
                    "final_evaluation_return"
                ]
            )

            independent_return = float(
                independent_by_seed[seed][
                    "final_evaluation_return"
                ]
            )

            difference = (
                maddpg_return
                - independent_return
            )

            if difference > 0.0:
                winner = "maddpg"
            elif difference < 0.0:
                winner = "independent_ddpg"
            else:
                winner = "tie"

            writer.writerow(
                {
                    "seed": seed,
                    "maddpg_final_return": (
                        maddpg_return
                    ),
                    "independent_ddpg_final_return": (
                        independent_return
                    ),
                    "maddpg_minus_independent_ddpg": (
                        difference
                    ),
                    "winner": winner,
                }
            )


def create_paired_final_summary(
    results_dir: Path,
    paired_path: Path,
    output_path: Path,
) -> None:
    """Summarize matched-seed final comparison."""

    rows = read_csv_rows(
        paired_path
    )

    if not rows:
        raise ValueError(
            "No paired comparison rows found."
        )

    differences = [
        float(
            row[
                "maddpg_minus_independent_ddpg"
            ]
        )
        for row in rows
    ]

    maddpg_wins = sum(
        row["winner"] == "maddpg"
        for row in rows
    )

    independent_wins = sum(
        row["winner"]
        == "independent_ddpg"
        for row in rows
    )

    ties = sum(
        row["winner"] == "tie"
        for row in rows
    )

    total_experiment_runtime = 0.0

    for algorithm in DEFAULT_ALGORITHMS:
        algorithm_rows = read_csv_rows(
            results_dir
            / algorithm
            / "summary.csv"
        )

        total_experiment_runtime += sum(
            float(
                row["runtime_seconds"]
            )
            for row in algorithm_rows
        )

    fieldnames = [
        "num_pairs",
        "maddpg_wins",
        "independent_ddpg_wins",
        "ties",
        "mean_paired_difference",
        "std_paired_difference",
        "total_experiment_seed_runtime_seconds",
        "total_experiment_seed_runtime_minutes",
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

        writer.writerow(
            {
                "num_pairs": len(rows),
                "maddpg_wins": maddpg_wins,
                "independent_ddpg_wins": (
                    independent_wins
                ),
                "ties": ties,
                "mean_paired_difference": fmean(
                    differences
                ),
                "std_paired_difference": (
                    sample_standard_deviation(
                        differences
                    )
                ),
                "total_experiment_seed_runtime_seconds": (
                    total_experiment_runtime
                ),
                "total_experiment_seed_runtime_minutes": (
                    total_experiment_runtime
                    / 60.0
                ),
            }
        )


def plot_curve_summary(
    summary_path: Path,
    output_path: Path,
    title: str,
) -> None:
    """Create mean curve with standard-deviation bands."""

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

    axes.set_title(
        title
    )
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

    plt.close(
        figure
    )


def plot_final_checkpoint_by_seed(
    source_path: Path,
    output_path: Path,
) -> None:
    """Plot final deterministic return for every seed."""

    rows = read_csv_rows(
        source_path
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
                row["seed"]
            ),
        )

        seeds = [
            int(row["seed"])
            for row in ordered_rows
        ]

        returns = [
            float(
                row[
                    "final_evaluation_return"
                ]
            )
            for row in ordered_rows
        ]

        axes.plot(
            seeds,
            returns,
            marker="o",
            label=algorithm,
        )

    axes.set_title(
        "Simple Spread Final Deterministic Evaluation"
    )
    axes.set_xlabel(
        "Matched seed"
    )
    axes.set_ylabel(
        "Final evaluation return"
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

    plt.close(
        figure
    )


def analyze_simple_spread_results(
    results_dir: str | Path,
    output_dir: str | Path,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
) -> tuple[Path, ...]:
    """Generate summaries and comparison plots."""

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

    final_by_seed_path = (
        analysis_directory
        / "final_checkpoint_by_seed.csv"
    )

    final_summary_path = (
        analysis_directory
        / "final_checkpoint_summary.csv"
    )

    paired_path = (
        analysis_directory
        / "paired_final_comparison.csv"
    )

    paired_summary_path = (
        analysis_directory
        / "paired_final_summary.csv"
    )

    training_plot_path = (
        analysis_directory
        / "training_curves.png"
    )

    evaluation_plot_path = (
        analysis_directory
        / "evaluation_curves.png"
    )

    final_plot_path = (
        analysis_directory
        / "final_checkpoint_by_seed.png"
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

    create_final_checkpoint_by_seed(
        results_dir=source_directory,
        output_path=final_by_seed_path,
        algorithms=algorithms,
    )

    create_final_checkpoint_summary(
        results_dir=source_directory,
        output_path=final_summary_path,
        algorithms=algorithms,
    )

    create_paired_final_comparison(
        results_dir=source_directory,
        output_path=paired_path,
    )

    create_paired_final_summary(
        results_dir=source_directory,
        paired_path=paired_path,
        output_path=paired_summary_path,
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
            "Simple Spread Deterministic "
            "Evaluation Returns"
        ),
    )

    plot_final_checkpoint_by_seed(
        source_path=final_by_seed_path,
        output_path=final_plot_path,
    )

    return (
        comparison_path,
        training_summary_path,
        evaluation_summary_path,
        final_by_seed_path,
        final_summary_path,
        paired_path,
        paired_summary_path,
        training_plot_path,
        evaluation_plot_path,
        final_plot_path,
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
            "Directory containing MADDPG and "
            "Independent-DDPG results."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory for summaries and plots."
        ),
    )

    arguments = parser.parse_args()

    output_paths = (
        analyze_simple_spread_results(
            results_dir=arguments.results_dir,
            output_dir=arguments.output_dir,
        )
    )

    print(
        "=== Analysis complete ==="
    )

    for output_path in output_paths:
        print(
            output_path
        )


if __name__ == "__main__":
    main()