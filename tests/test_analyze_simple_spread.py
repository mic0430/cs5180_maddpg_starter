from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.experiments.analyze_simple_spread import (
    analyze_simple_spread_results,
    sample_standard_deviation,
)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def create_fake_algorithm_results(
    results_dir: Path,
    algorithm: str,
    evaluation_offset: float,
) -> None:
    algorithm_dir = (
        results_dir / algorithm
    )

    write_csv(
        algorithm_dir / "summary.csv",
        [
            "algorithm",
            "seed",
            "mean_training_return",
            "mean_evaluation_return",
            "final_evaluation_episode",
            "final_evaluation_return",
            "runtime_seconds",
            "runtime_minutes",
        ],
        [
            {
                "algorithm": algorithm,
                "seed": 1,
                "mean_training_return": -5.0,
                "mean_evaluation_return": (
                    -4.0
                    + evaluation_offset
                ),
                "final_evaluation_episode": 2,
                "final_evaluation_return": (
                    -2.0
                    + evaluation_offset
                ),
                "runtime_seconds": 60.0,
                "runtime_minutes": 1.0,
            },
            {
                "algorithm": algorithm,
                "seed": 2,
                "mean_training_return": -3.0,
                "mean_evaluation_return": (
                    -2.0
                    + evaluation_offset
                ),
                "final_evaluation_episode": 2,
                "final_evaluation_return": (
                    -1.0
                    + evaluation_offset
                ),
                "runtime_seconds": 120.0,
                "runtime_minutes": 2.0,
            },
        ],
    )

    training_rows: list[
        dict[str, object]
    ] = []

    evaluation_rows: list[
        dict[str, object]
    ] = []

    for seed in (1, 2):
        training_rows.extend(
            [
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": 1,
                    "training_return": (
                        -5.0 + seed
                    ),
                },
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": 2,
                    "training_return": (
                        -4.0 + seed
                    ),
                },
            ]
        )

        evaluation_rows.extend(
            [
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": 1,
                    "evaluation_return": (
                        -4.0
                        + seed
                        + evaluation_offset
                    ),
                },
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": 2,
                    "evaluation_return": (
                        -3.0
                        + seed
                        + evaluation_offset
                    ),
                },
            ]
        )

    write_csv(
        algorithm_dir
        / "training_curves.csv",
        [
            "algorithm",
            "seed",
            "episode",
            "training_return",
        ],
        training_rows,
    )

    write_csv(
        algorithm_dir
        / "evaluation_curves.csv",
        [
            "algorithm",
            "seed",
            "episode",
            "evaluation_return",
        ],
        evaluation_rows,
    )


def test_sample_standard_deviation_single_value() -> None:
    assert sample_standard_deviation(
        [3.0]
    ) == 0.0


def test_analysis_writes_summaries_and_plots(
    tmp_path: Path,
) -> None:
    results_dir = (
        tmp_path / "results"
    )

    create_fake_algorithm_results(
        results_dir=results_dir,
        algorithm="maddpg",
        evaluation_offset=0.0,
    )

    create_fake_algorithm_results(
        results_dir=results_dir,
        algorithm="independent_ddpg",
        evaluation_offset=-1.0,
    )

    output_dir = (
        tmp_path / "analysis"
    )

    output_paths = (
        analyze_simple_spread_results(
            results_dir=results_dir,
            output_dir=output_dir,
        )
    )

    assert len(
        output_paths
    ) == 10

    for output_path in output_paths:
        assert output_path.exists()
        assert (
            output_path.stat().st_size
            > 0
        )

    comparison_path = (
        output_dir
        / "comparison_summary.csv"
    )

    with comparison_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        comparison_rows = list(
            csv.DictReader(file)
        )

    rows_by_algorithm = {
        row["algorithm"]: row
        for row in comparison_rows
    }

    assert float(
        rows_by_algorithm[
            "maddpg"
        ]["mean_evaluation_return"]
    ) == pytest.approx(
        -3.0
    )

    assert float(
        rows_by_algorithm[
            "independent_ddpg"
        ]["mean_evaluation_return"]
    ) == pytest.approx(
        -4.0
    )

    final_summary_path = (
        output_dir
        / "final_checkpoint_summary.csv"
    )

    with final_summary_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        final_rows = list(
            csv.DictReader(file)
        )

    final_by_algorithm = {
        row["algorithm"]: row
        for row in final_rows
    }

    assert float(
        final_by_algorithm[
            "maddpg"
        ]["final_mean_return"]
    ) == pytest.approx(
        -1.5
    )

    assert float(
        final_by_algorithm[
            "independent_ddpg"
        ]["final_mean_return"]
    ) == pytest.approx(
        -2.5
    )

    assert float(
        final_by_algorithm[
            "maddpg"
        ]["mean_runtime_seconds"]
    ) == pytest.approx(
        90.0
    )

    assert float(
        final_by_algorithm[
            "maddpg"
        ]["total_runtime_seconds"]
    ) == pytest.approx(
        180.0
    )

    paired_path = (
        output_dir
        / "paired_final_comparison.csv"
    )

    with paired_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        paired_rows = list(
            csv.DictReader(file)
        )

    assert len(
        paired_rows
    ) == 2

    assert all(
        row["winner"] == "maddpg"
        for row in paired_rows
    )

    assert all(
        float(
            row[
                "maddpg_minus_independent_ddpg"
            ]
        )
        == pytest.approx(1.0)
        for row in paired_rows
    )

    paired_summary_path = (
        output_dir
        / "paired_final_summary.csv"
    )

    with paired_summary_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        paired_summary_rows = list(
            csv.DictReader(file)
        )

    assert len(
        paired_summary_rows
    ) == 1

    paired_summary = (
        paired_summary_rows[0]
    )

    assert (
        paired_summary["maddpg_wins"]
        == "2"
    )

    assert (
        paired_summary[
            "independent_ddpg_wins"
        ]
        == "0"
    )

    assert float(
        paired_summary[
            "mean_paired_difference"
        ]
    ) == pytest.approx(
        1.0
    )

    assert float(
        paired_summary[
            "total_experiment_seed_runtime_seconds"
        ]
    ) == pytest.approx(
        360.0
    )