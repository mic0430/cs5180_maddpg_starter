from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.experiments.train_simple_spread import (
    TrainingConfig,
    TrainingResult,
    train_simple_spread,
)


def validate_seeds(
    seeds: Sequence[int],
) -> tuple[int, ...]:
    """Validate and normalize experiment seeds."""

    normalized_seeds = tuple(
        int(seed)
        for seed in seeds
    )

    if not normalized_seeds:
        raise ValueError(
            "At least one seed is required."
        )

    if any(
        seed < 0
        for seed in normalized_seeds
    ):
        raise ValueError(
            "Seeds must be non-negative."
        )

    if len(set(normalized_seeds)) != len(
        normalized_seeds
    ):
        raise ValueError(
            "Seeds must be unique."
        )

    return normalized_seeds


def write_summary_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
) -> None:
    """Write one summary row per seed."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "seed",
                "episodes_completed",
                "environment_steps",
                "update_count",
                "mean_training_return",
                "mean_evaluation_return",
                "evaluation_runs",
                "checkpoint_path",
            ],
        )

        writer.writeheader()

        for seed, result in zip(
            seeds,
            results,
            strict=True,
        ):
            writer.writerow(
                {
                    "algorithm": result.algorithm,
                    "seed": seed,
                    "episodes_completed": (
                        result.episodes_completed
                    ),
                    "environment_steps": (
                        result.environment_steps
                    ),
                    "update_count": result.update_count,
                    "mean_training_return": (
                        result.mean_episode_return
                    ),
                    "mean_evaluation_return": (
                        result.mean_evaluation_return
                        if result.mean_evaluation_return
                        is not None
                        else ""
                    ),
                    "evaluation_runs": (
                        result.evaluation_runs
                    ),
                    "checkpoint_path": (
                        str(result.checkpoint_path)
                        if result.checkpoint_path
                        is not None
                        else ""
                    ),
                }
            )


def write_training_curves_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
) -> None:
    """Write every per-episode training return."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "seed",
                "episode",
                "training_return",
            ],
        )

        writer.writeheader()

        for seed, result in zip(
            seeds,
            results,
            strict=True,
        ):
            for episode, episode_return in enumerate(
                result.training_episode_returns,
                start=1,
            ):
                writer.writerow(
                    {
                        "algorithm": result.algorithm,
                        "seed": seed,
                        "episode": episode,
                        "training_return": (
                            episode_return
                        ),
                    }
                )


def write_evaluation_curves_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
) -> None:
    """Write deterministic evaluation returns."""

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "seed",
                "episode",
                "evaluation_return",
            ],
        )

        writer.writeheader()

        for seed, result in zip(
            seeds,
            results,
            strict=True,
        ):
            for episode, evaluation_return in zip(
                result.evaluation_episode_indices,
                result.evaluation_returns,
                strict=True,
            ):
                writer.writerow(
                    {
                        "algorithm": result.algorithm,
                        "seed": seed,
                        "episode": episode,
                        "evaluation_return": (
                            evaluation_return
                        ),
                    }
                )


def run_seed_sweep(
    base_config: TrainingConfig,
    seeds: Sequence[int],
    output_dir: str | Path,
) -> tuple[TrainingResult, ...]:
    """Run matched training for multiple seeds."""

    normalized_seeds = validate_seeds(seeds)

    algorithm_dir = (
        Path(output_dir)
        / base_config.algorithm
    )

    algorithm_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[TrainingResult] = []

    for seed in normalized_seeds:
        seed_dir = (
            algorithm_dir
            / f"seed_{seed}"
        )

        seed_config = replace(
            base_config,
            seed=seed,
            log_dir=seed_dir / "tensorboard",
            checkpoint_path=(
                seed_dir / "checkpoint.pt"
            ),
        )

        print(
            f"=== Running "
            f"{base_config.algorithm} "
            f"seed {seed} ==="
        )

        result = train_simple_spread(
            seed_config
        )

        results.append(result)

        print(
            f"Seed {seed} complete: "
            f"train={result.mean_episode_return:.4f}, "
            f"evaluation="
            f"{result.mean_evaluation_return}"
        )

    completed_results = tuple(results)

    write_summary_csv(
        algorithm_dir / "summary.csv",
        normalized_seeds,
        completed_results,
    )

    write_training_curves_csv(
        algorithm_dir / "training_curves.csv",
        normalized_seeds,
        completed_results,
    )

    write_evaluation_curves_csv(
        algorithm_dir / "evaluation_curves.csv",
        normalized_seeds,
        completed_results,
    )

    return completed_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched Simple Spread training "
            "over multiple random seeds."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Base YAML training configuration.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="One or more unique non-negative seeds.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for CSVs, logs, and checkpoints.",
    )

    arguments = parser.parse_args()

    config = TrainingConfig.from_yaml(
        arguments.config
    )

    results = run_seed_sweep(
        base_config=config,
        seeds=arguments.seeds,
        output_dir=arguments.output_dir,
    )

    algorithm_dir = (
        arguments.output_dir
        / config.algorithm
    )

    print("=== Seed sweep complete ===")
    print(f"Algorithm: {config.algorithm}")
    print(f"Seeds completed: {len(results)}")
    print(
        f"Summary: "
        f"{algorithm_dir / 'summary.csv'}"
    )
    print(
        f"Training curves: "
        f"{algorithm_dir / 'training_curves.csv'}"
    )
    print(
        f"Evaluation curves: "
        f"{algorithm_dir / 'evaluation_curves.csv'}"
    )


if __name__ == "__main__":
    main()
