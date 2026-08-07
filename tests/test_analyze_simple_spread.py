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
        writer.writerows(rows)


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
        ],
        [
            {
                "algorithm": algorithm,
                "seed": 1,
                "mean_training_return": -5.0,
                "mean_evaluation_return": (
                    -4.0 + evaluation_offset
                ),
            },
            {
                "algorithm": algorithm,
                "seed": 2,
                "mean_training_return": -3.0,
                "mean_evaluation_return": (
                    -2.0 + evaluation_offset
                ),
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
                    "training_return": -5.0 + seed,
                },
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": 2,
                    "training_return": -4.0 + seed,
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
        algorithm_dir / "training_curves.csv",
        [
            "algorithm",
            "seed",
            "episode",
            "training_return",
        ],
        training_rows,
    )

    write_csv(
        algorithm_dir / "evaluation_curves.csv",
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

    assert len(output_paths) == 5

    for output_path in output_paths:
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    comparison_path = (
        output_dir
        / "comparison_summary.csv"
    )

    with comparison_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 2

    rows_by_algorithm = {
        row["algorithm"]: row
        for row in rows
    }

    assert float(
        rows_by_algorithm[
            "maddpg"
        ]["mean_evaluation_return"]
    ) == pytest.approx(-3.0)

    assert float(
        rows_by_algorithm[
            "independent_ddpg"
        ]["mean_evaluation_return"]
    ) == pytest.approx(-4.0)
