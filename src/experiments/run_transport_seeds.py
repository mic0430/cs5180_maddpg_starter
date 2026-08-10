from __future__ import annotations

import argparse
import csv
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.experiments.train_transport import (
    EvaluationResult,
    TrainingConfig,
    TrainingResult,
    train_transport,
)


def validate_seeds(
    seeds: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(int(seed) for seed in seeds)

    if not normalized:
        raise ValueError(
            "At least one seed is required."
        )

    if any(seed < 0 for seed in normalized):
        raise ValueError(
            "Seeds must be non-negative."
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "Seeds must be unique."
        )

    return normalized


def write_summary_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
    runtimes: tuple[float, ...],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "algorithm",
        "seed",
        "episodes_completed",
        "environment_steps",
        "update_count",
        "mean_training_return",
        "training_seconds",
        "wall_runtime_seconds",
        "final_eval_return",
        "final_success_rate",
        "final_mean_steps",
        "final_mean_distance",
        "final_mean_collisions",
        "final_mean_force_disagreement",
        "final_mean_control_effort",
        "final_mean_max_abs_orientation",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for seed, result, runtime in zip(
            seeds,
            results,
            runtimes,
            strict=True,
        ):
            evaluation = result.final_evaluation

            row = {
                "algorithm": result.algorithm,
                "seed": seed,
                "episodes_completed": (
                    result.episodes_completed
                ),
                "environment_steps": (
                    result.environment_steps
                ),
                "update_count": (
                    result.update_count
                ),
                "mean_training_return": (
                    result.mean_episode_return
                ),
                "training_seconds": (
                    result.training_seconds
                ),
                "wall_runtime_seconds": runtime,
                "final_eval_return": "",
                "final_success_rate": "",
                "final_mean_steps": "",
                "final_mean_distance": "",
                "final_mean_collisions": "",
                "final_mean_force_disagreement": "",
                "final_mean_control_effort": "",
                "final_mean_max_abs_orientation": "",
            }

            if evaluation is not None:
                row.update(
                    {
                        "final_eval_return": (
                            evaluation.mean_return
                        ),
                        "final_success_rate": (
                            evaluation.success_rate
                        ),
                        "final_mean_steps": (
                            evaluation.mean_steps
                        ),
                        "final_mean_distance": (
                            evaluation.mean_final_distance
                        ),
                        "final_mean_collisions": (
                            evaluation.mean_collisions
                        ),
                        "final_mean_force_disagreement": (
                            evaluation.mean_force_disagreement
                        ),
                        "final_mean_control_effort": (
                            evaluation.mean_control_effort
                        ),
                        "final_mean_max_abs_orientation": (
                            evaluation.mean_max_abs_orientation
                        ),
                    }
                )

            writer.writerow(row)


def write_training_curves_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
) -> None:
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


def evaluation_row(
    algorithm: str,
    seed: int,
    episode: int,
    evaluation: EvaluationResult,
) -> dict[str, object]:
    return {
        "algorithm": algorithm,
        "seed": seed,
        "episode": episode,
        "mean_return": evaluation.mean_return,
        "success_rate": evaluation.success_rate,
        "mean_steps": evaluation.mean_steps,
        "mean_final_distance": (
            evaluation.mean_final_distance
        ),
        "mean_collisions": (
            evaluation.mean_collisions
        ),
        "mean_force_disagreement": (
            evaluation.mean_force_disagreement
        ),
        "mean_control_effort": (
            evaluation.mean_control_effort
        ),
        "mean_max_abs_orientation": (
            evaluation.mean_max_abs_orientation
        ),
    }


def write_evaluation_curves_csv(
    path: Path,
    seeds: tuple[int, ...],
    results: tuple[TrainingResult, ...],
) -> None:
    fields = [
        "algorithm",
        "seed",
        "episode",
        "mean_return",
        "success_rate",
        "mean_steps",
        "mean_final_distance",
        "mean_collisions",
        "mean_force_disagreement",
        "mean_control_effort",
        "mean_max_abs_orientation",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for seed, result in zip(
            seeds,
            results,
            strict=True,
        ):
            if (
                len(result.evaluation_episode_indices)
                != len(result.evaluation_results)
            ):
                raise RuntimeError(
                    "Evaluation episode/result "
                    "history lengths do not match."
                )

            for episode, evaluation in zip(
                result.evaluation_episode_indices,
                result.evaluation_results,
                strict=True,
            ):
                writer.writerow(
                    evaluation_row(
                        algorithm=result.algorithm,
                        seed=seed,
                        episode=episode,
                        evaluation=evaluation,
                    )
                )


def run_seed_sweep(
    base_config: TrainingConfig,
    seeds: Sequence[int],
    output_dir: str | Path,
) -> tuple[TrainingResult, ...]:
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
    runtimes: list[float] = []

    sweep_start = time.perf_counter()

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

        print()
        print(
            f"=== Running "
            f"{base_config.algorithm} "
            f"seed {seed} ==="
        )

        run_start = time.perf_counter()

        result = train_transport(
            seed_config
        )

        runtime = (
            time.perf_counter()
            - run_start
        )

        results.append(result)
        runtimes.append(runtime)

        evaluation = result.final_evaluation

        if evaluation is None:
            print(
                f"Seed {seed} complete: "
                f"train_return="
                f"{result.mean_episode_return:.4f}, "
                f"runtime="
                f"{runtime / 60.0:.2f} min"
            )
        else:
            print(
                f"Seed {seed} complete: "
                f"success="
                f"{evaluation.success_rate:.3f}, "
                f"return="
                f"{evaluation.mean_return:.4f}, "
                f"steps="
                f"{evaluation.mean_steps:.2f}, "
                f"runtime="
                f"{runtime / 60.0:.2f} min"
            )

    completed_results = tuple(results)
    completed_runtimes = tuple(runtimes)

    write_summary_csv(
        algorithm_dir / "summary.csv",
        normalized_seeds,
        completed_results,
        completed_runtimes,
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

    sweep_runtime = (
        time.perf_counter()
        - sweep_start
    )

    print()
    print("=== Seed sweep complete ===")
    print(
        f"Algorithm: {base_config.algorithm}"
    )
    print(
        f"Seeds completed: "
        f"{len(completed_results)}"
    )
    mean_runtime_seconds = (
        sum(completed_runtimes)
        / len(completed_runtimes)
    )

    print(
        f"Average runtime per seed: "
        f"{mean_runtime_seconds / 60.0:.2f} min"
    )
    print(
        f"Sum of individual seed runtimes: "
        f"{sum(completed_runtimes) / 60.0:.2f} min"
    )
    print(
        f"Seed sweep wall-clock runtime: "
        f"{sweep_runtime / 60.0:.2f} min"
    )
    print(
        f"Summary: "
        f"{algorithm_dir / 'summary.csv'}"
    )

    return completed_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run cooperative-transport training "
            "over multiple matched random seeds."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    config = TrainingConfig.from_yaml(
        args.config
    )

    run_seed_sweep(
        base_config=config,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()