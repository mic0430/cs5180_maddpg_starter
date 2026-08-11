from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from src.algorithms.independent_ddpg import IndependentDDPG
from src.algorithms.maddpg import MADDPG
from src.controllers.force_feedback import (
    ForceFeedbackController,
    ForceFeedbackGains,
)
from src.envs.transport_env import CooperativeTransportEnv
from src.experiments.train_transport import (
    _info_float,
    load_environment_config,
    reward_vector,
)


DEFAULT_TRAINING_SEEDS = [
    3,
    7,
    11,
    19,
    23,
    29,
    31,
    37,
    41,
    47,
]

# Fresh common test conditions.
# These are intentionally different from the seeds used during
# training, developmental controller tuning, and the existing
# training-time evaluation seed blocks.
DEFAULT_TEST_SEEDS = list(range(200001, 200021))


METRIC_NAMES = [
    "return",
    "steps",
    "final_distance",
    "collisions",
    "force_disagreement",
    "control_effort",
    "max_abs_orientation",
]


def evaluate_policy(
    policy: Any,
    *,
    algorithm: str,
    environment_config: Path,
    test_seeds: list[int],
    training_seed: int | None,
) -> list[dict[str, Any]]:
    env_config = load_environment_config(environment_config)
    env = CooperativeTransportEnv(env_config)

    rows: list[dict[str, Any]] = []

    for test_seed in test_seeds:
        observations, info = env.reset(seed=test_seed)

        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False

        initial_orientation = _info_float(
            info,
            "payload_orientation",
        )

        if np.isfinite(initial_orientation):
            max_abs_orientation = abs(initial_orientation)
        else:
            max_abs_orientation = 0.0

        while not (terminated or truncated):
            if algorithm == "handcrafted":
                actions = policy.select_actions(observations)
            else:
                actions = policy.select_actions(
                    observations,
                    explore=False,
                )

            (
                observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(actions)

            rewards = reward_vector(
                rewards,
                env.num_agents,
            )

            episode_return += float(np.mean(rewards))
            steps += 1

            orientation = _info_float(
                info,
                "payload_orientation",
            )

            if np.isfinite(orientation):
                max_abs_orientation = max(
                    max_abs_orientation,
                    abs(orientation),
                )

        rows.append(
            {
                "algorithm": algorithm,
                "training_seed": (
                    ""
                    if training_seed is None
                    else training_seed
                ),
                "test_seed": test_seed,
                "success": int(
                    bool(info.get("success", False))
                ),
                "return": episode_return,
                "steps": steps,
                "final_distance": _info_float(
                    info,
                    "payload_distance_to_goal",
                ),
                "collisions": _info_float(
                    info,
                    "collision_count",
                ),
                "force_disagreement": _info_float(
                    info,
                    "mean_force_disagreement",
                ),
                "control_effort": _info_float(
                    info,
                    "total_control_effort",
                ),
                "max_abs_orientation": (
                    max_abs_orientation
                ),
            }
        )

    return rows


def make_learner(
    algorithm: str,
    environment_config: Path,
    checkpoint: Path,
    device: str,
) -> MADDPG | IndependentDDPG:
    env_config = load_environment_config(environment_config)
    env = CooperativeTransportEnv(env_config)

    common_arguments = {
        "num_agents": env.num_agents,
        "observation_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "hidden_sizes": (64, 64),
        "device": device,
    }

    if algorithm == "maddpg":
        learner = MADDPG(**common_arguments)
    elif algorithm == "independent_ddpg":
        learner = IndependentDDPG(**common_arguments)
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    learner.load(
        checkpoint,
        load_optimizers=False,
    )

    return learner


def safe_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)

    if np.all(np.isnan(array)):
        return float("nan")

    return float(np.nanmean(array))


def sample_std(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]

    if len(array) < 2:
        return float("nan")

    return float(np.std(array, ddof=1))


def summarize_policy(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = rows[0]

    summary: dict[str, Any] = {
        "algorithm": first["algorithm"],
        "training_seed": first["training_seed"],
        "test_episode_count": len(rows),
        "success_rate": safe_mean(
            [float(row["success"]) for row in rows]
        ),
    }

    for metric in METRIC_NAMES:
        summary[f"mean_{metric}"] = safe_mean(
            [float(row[metric]) for row in rows]
        )

    return summary


def summarize_algorithm(
    policy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = policy_rows[0]

    output: dict[str, Any] = {
        "algorithm": first["algorithm"],
        "policy_count": len(policy_rows),
        "test_seeds_per_policy": int(
            policy_rows[0]["test_episode_count"]
        ),
        "mean_success_rate": safe_mean(
            [
                float(row["success_rate"])
                for row in policy_rows
            ]
        ),
        "std_success_rate_across_policies": sample_std(
            [
                float(row["success_rate"])
                for row in policy_rows
            ]
        ),
    }

    for metric in METRIC_NAMES:
        field = f"mean_{metric}"

        values = [
            float(row[field])
            for row in policy_rows
        ]

        output[field] = safe_mean(values)
        output[
            f"std_{metric}_across_policies"
        ] = sample_std(values)

    return output


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows available for {path}"
        )

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
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained MADDPG and Independent "
            "DDPG checkpoints plus the handcrafted "
            "force-feedback controller on identical "
            "fixed transport test seeds."
        )
    )

    parser.add_argument(
        "--environment-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--maddpg-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--idddpg-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAINING_SEEDS,
    )

    parser.add_argument(
        "--test-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TEST_SEEDS,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_episode_rows: list[dict[str, Any]] = []
    policy_summaries: list[dict[str, Any]] = []

    algorithm_roots = {
        "maddpg": args.maddpg_root,
        "independent_ddpg": args.idddpg_root,
    }

    for algorithm, root in algorithm_roots.items():
        for training_seed in args.training_seeds:
            checkpoint = (
                root
                / f"seed_{training_seed}"
                / "checkpoint.pt"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {checkpoint}"
                )

            print(
                f"Evaluating {algorithm} "
                f"training seed {training_seed}..."
            )

            learner = make_learner(
                algorithm=algorithm,
                environment_config=(
                    args.environment_config
                ),
                checkpoint=checkpoint,
                device=args.device,
            )

            rows = evaluate_policy(
                learner,
                algorithm=algorithm,
                environment_config=(
                    args.environment_config
                ),
                test_seeds=args.test_seeds,
                training_seed=training_seed,
            )

            all_episode_rows.extend(rows)
            policy_summaries.append(
                summarize_policy(rows)
            )

    print("Evaluating handcrafted controller...")

    handcrafted = ForceFeedbackController(
        ForceFeedbackGains()
    )

    handcrafted_rows = evaluate_policy(
        handcrafted,
        algorithm="handcrafted",
        environment_config=args.environment_config,
        test_seeds=args.test_seeds,
        training_seed=None,
    )

    all_episode_rows.extend(handcrafted_rows)
    policy_summaries.append(
        summarize_policy(handcrafted_rows)
    )

    aggregate_rows: list[dict[str, Any]] = []

    for algorithm in [
        "maddpg",
        "independent_ddpg",
        "handcrafted",
    ]:
        matching = [
            row
            for row in policy_summaries
            if row["algorithm"] == algorithm
        ]

        aggregate_rows.append(
            summarize_algorithm(matching)
        )

    write_csv(
        args.output_dir / "episode_results.csv",
        all_episode_rows,
    )

    write_csv(
        args.output_dir / "policy_summary.csv",
        policy_summaries,
    )

    write_csv(
        args.output_dir / "aggregate_summary.csv",
        aggregate_rows,
    )

    print()
    print("=== Common evaluation complete ===")

    for row in aggregate_rows:
        print(
            f"{row['algorithm']}: "
            f"success="
            f"{row['mean_success_rate']:.3f}, "
            f"return="
            f"{row['mean_return']:.3f}, "
            f"steps="
            f"{row['mean_steps']:.2f}"
        )

    print(
        f"Results: {args.output_dir}"
    )


if __name__ == "__main__":
    main()