from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.experiments.run_simple_spread_seeds import (
    run_seed_sweep,
    validate_seeds,
)
from src.experiments.train_simple_spread import (
    TrainingConfig,
)


def test_validate_seeds_rejects_duplicates() -> None:
    with pytest.raises(
        ValueError,
        match="Seeds must be unique",
    ):
        validate_seeds([1, 1])


def test_seed_sweep_writes_expected_outputs(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        algorithm="maddpg",
        seed=1,
        device="cpu",
        num_agents=2,
        max_cycles=2,
        local_ratio=0.5,
        episodes=2,
        batch_size=2,
        replay_capacity=32,
        learning_starts=2,
        update_every=1,
        hidden_sizes=(8, 8),
        exploration_noise_std=0.2,
        exploration_noise_final=0.1,
        exploration_decay_episodes=2,
        evaluation_interval=1,
        evaluation_episodes=1,
        log_dir=(
            tmp_path / "unused-logs"
        ),
    )

    output_dir = (
        tmp_path / "results"
    )

    results = run_seed_sweep(
        base_config=config,
        seeds=[3],
        output_dir=output_dir,
    )

    assert len(results) == 1

    algorithm_dir = (
        output_dir / "maddpg"
    )

    summary_path = (
        algorithm_dir / "summary.csv"
    )

    training_path = (
        algorithm_dir
        / "training_curves.csv"
    )

    evaluation_path = (
        algorithm_dir
        / "evaluation_curves.csv"
    )

    checkpoint_path = (
        algorithm_dir
        / "seed_3"
        / "checkpoint.pt"
    )

    assert summary_path.exists()
    assert training_path.exists()
    assert evaluation_path.exists()
    assert checkpoint_path.exists()

    with summary_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        summary_rows = list(
            csv.DictReader(file)
        )

    with training_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        training_rows = list(
            csv.DictReader(file)
        )

    with evaluation_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        evaluation_rows = list(
            csv.DictReader(file)
        )

    assert len(summary_rows) == 1

    summary_row = summary_rows[0]

    assert summary_row["seed"] == "3"

    assert (
        summary_row[
            "final_evaluation_episode"
        ]
        == "2"
    )

    assert (
        summary_row[
            "final_evaluation_return"
        ]
        != ""
    )

    assert (
        summary_row[
            "mean_evaluation_return"
        ]
        != ""
    )

    runtime_seconds = float(
        summary_row[
            "runtime_seconds"
        ]
    )

    runtime_minutes = float(
        summary_row[
            "runtime_minutes"
        ]
    )

    assert runtime_seconds > 0.0

    assert runtime_minutes == pytest.approx(
        runtime_seconds / 60.0
    )

    assert len(training_rows) == 2
    assert len(evaluation_rows) == 2

    assert (
        evaluation_rows[-1]["episode"]
        == "2"
    )

    assert (
        float(
            summary_row[
                "final_evaluation_return"
            ]
        )
        == pytest.approx(
            float(
                evaluation_rows[-1][
                    "evaluation_return"
                ]
            )
        )
    )