from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.controllers.force_feedback import (
    ForceFeedbackController,
    ForceFeedbackGains,
)
from src.envs.transport_env import CooperativeTransportEnv
from src.experiments.run_transport_common_eval import make_learner
from src.experiments.train_transport import (
    _info_float,
    load_environment_config,
    reward_vector,
)
from src.robustness import (
    ActionDeadzone,
    ObservationNoise,
    aggregate_results,
    control_effort,
    force_imbalance,
    payload_stability,
    plot_robustness_curves,
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

DEFAULT_OBSERVATION_NOISE_LEVELS = [
    0.00,
    0.02,
    0.05,
    0.10,
    0.20,
]

DEFAULT_ACTION_DEADZONE_LEVELS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
]


def apply_observation_noise(
    observations: np.ndarray,
    perturbation: ObservationNoise,
) -> np.ndarray:
    """Apply observation noise independently to each agent observation."""
    observations = np.asarray(observations, dtype=np.float32)

    noisy_observations = [
        perturbation.apply(agent_observation)
        for agent_observation in observations
    ]

    return np.asarray(
        noisy_observations,
        dtype=np.float32,
    )


def select_policy_actions(
    policy: Any,
    observations: np.ndarray,
    algorithm: str,
) -> np.ndarray:
    """Select deterministic actions from one of the supported controllers."""
    if algorithm == "handcrafted":
        actions = policy.select_actions(observations)
    else:
        actions = policy.select_actions(
            observations,
            explore=False,
        )

    return np.asarray(actions, dtype=np.float32)


def evaluate_disturbance_condition(
    policy: Any,
    *,
    algorithm: str,
    training_seed: int | None,
    environment_config: Path,
    disturbance_type: str,
    disturbance_level: float,
    disturbance_level_index: int,
    test_seeds: list[int],
) -> list[dict[str, Any]]:
    """Evaluate one policy under one fixed robustness condition."""
    env_config = load_environment_config(environment_config)
    env = CooperativeTransportEnv(env_config)

    rows: list[dict[str, Any]] = []

    for test_seed in test_seeds:
        observations, info = env.reset(seed=test_seed)

        episode_return = 0.0
        steps = 0
        terminated = False
        truncated = False

        orientations: list[float] = []
        angular_velocities: list[float] = []
        executed_actions: list[np.ndarray] = []
        force_imbalances: list[float] = []

        initial_orientation = _info_float(
            info,
            "payload_orientation",
        )
        initial_angular_velocity = _info_float(
            info,
            "payload_angular_velocity",
        )

        if (
            np.isfinite(initial_orientation)
            and np.isfinite(initial_angular_velocity)
        ):
            orientations.append(initial_orientation)
            angular_velocities.append(initial_angular_velocity)

        max_abs_orientation = (
            abs(initial_orientation)
            if np.isfinite(initial_orientation)
            else 0.0
        )

        observation_noise: ObservationNoise | None = None
        action_deadzone: ActionDeadzone | None = None

        if disturbance_type == "observation_noise":
            # The same test seed + disturbance level produces the same
            # observation-noise stream for every evaluated policy.
            noise_seed = (
                test_seed
                + 1_000_000 * disturbance_level_index
            )

            observation_noise = ObservationNoise(
                magnitude=disturbance_level,
                seed=noise_seed,
            )

        elif disturbance_type == "action_deadzone":
            action_deadzone = ActionDeadzone(
                threshold=disturbance_level,
            )

        else:
            raise ValueError(
                f"Unsupported disturbance type: {disturbance_type}"
            )

        while not (terminated or truncated):
            policy_observations = observations

            if observation_noise is not None:
                policy_observations = apply_observation_noise(
                    observations,
                    observation_noise,
                )

            actions = select_policy_actions(
                policy,
                policy_observations,
                algorithm,
            )

            if action_deadzone is not None:
                actions = np.asarray(
                    action_deadzone.apply(actions),
                    dtype=np.float32,
                )

            executed_actions.append(
                np.asarray(actions, dtype=np.float64).copy()
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
            angular_velocity = _info_float(
                info,
                "payload_angular_velocity",
            )

            if np.isfinite(orientation):
                max_abs_orientation = max(
                    max_abs_orientation,
                    abs(orientation),
                )

            if (
                np.isfinite(orientation)
                and np.isfinite(angular_velocity)
            ):
                orientations.append(orientation)
                angular_velocities.append(angular_velocity)

            coupling_forces = info.get("coupling_forces")

            if coupling_forces is not None:
                coupling_forces = np.asarray(
                    coupling_forces,
                    dtype=np.float64,
                )

                if (
                    coupling_forces.ndim == 2
                    and np.isfinite(coupling_forces).all()
                ):
                    force_imbalances.append(
                        force_imbalance(coupling_forces)
                    )

        if not orientations:
            raise RuntimeError(
                "Transport environment did not provide finite "
                "payload_orientation/payload_angular_velocity values."
            )

        if not executed_actions:
            raise RuntimeError(
                "Robustness episode completed without any actions."
            )

        episode_control_effort = control_effort(
            np.asarray(executed_actions)
        )

        episode_force_imbalance = (
            float(np.mean(force_imbalances))
            if force_imbalances
            else _info_float(
                info,
                "mean_force_disagreement",
            )
        )

        if not np.isfinite(episode_force_imbalance):
            raise RuntimeError(
                "Unable to calculate force imbalance for robustness episode."
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
                "disturbance_type": disturbance_type,
                "disturbance_level": disturbance_level,
                "success": int(
                    bool(info.get("success", False))
                ),
                "episode_return": episode_return,
                "final_goal_distance": _info_float(
                    info,
                    "payload_distance_to_goal",
                ),
                "episode_length": steps,
                "payload_stability": payload_stability(
                    orientations,
                    angular_velocities,
                ),
                "control_effort": episode_control_effort,
                "force_imbalance": episode_force_imbalance,
                "collisions": _info_float(
                    info,
                    "collision_count",
                ),
                "force_disagreement": _info_float(
                    info,
                    "mean_force_disagreement",
                ),
                "max_abs_orientation": max_abs_orientation,
            }
        )

    return rows


def evaluate_policy_robustness(
    policy: Any,
    *,
    algorithm: str,
    training_seed: int | None,
    environment_config: Path,
    observation_noise_levels: list[float],
    action_deadzone_levels: list[float],
    test_seeds: list[int],
) -> list[dict[str, Any]]:
    """Evaluate one policy over all frozen robustness conditions."""
    rows: list[dict[str, Any]] = []

    for index, level in enumerate(observation_noise_levels):
        print(
            f"  observation_noise={level:.3f}",
            flush=True,
        )

        rows.extend(
            evaluate_disturbance_condition(
                policy,
                algorithm=algorithm,
                training_seed=training_seed,
                environment_config=environment_config,
                disturbance_type="observation_noise",
                disturbance_level=level,
                disturbance_level_index=index,
                test_seeds=test_seeds,
            )
        )

    for index, level in enumerate(action_deadzone_levels):
        print(
            f"  action_deadzone={level:.3f}",
            flush=True,
        )

        rows.extend(
            evaluate_disturbance_condition(
                policy,
                algorithm=algorithm,
                training_seed=training_seed,
                environment_config=environment_config,
                disturbance_type="action_deadzone",
                disturbance_level=level,
                disturbance_level_index=index,
                test_seeds=test_seeds,
            )
        )

    return rows


def build_policy_summary(
    episode_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each trained policy separately."""
    return aggregate_results(
        episode_results,
        group_fields=(
            "algorithm",
            "training_seed",
            "disturbance_type",
            "disturbance_level",
        ),
    )


def build_aggregate_summary(
    policy_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate policy-level robustness results across training seeds."""
    metric_columns = [
        "success_rate",
        "mean_episode_return",
        "mean_final_goal_distance",
        "mean_episode_length",
        "mean_payload_stability",
        "mean_control_effort",
        "mean_force_imbalance",
    ]

    rows: list[dict[str, Any]] = []

    grouped = policy_summary.groupby(
        [
            "algorithm",
            "disturbance_type",
            "disturbance_level",
        ],
        dropna=False,
    )

    for (
        algorithm,
        disturbance_type,
        disturbance_level,
    ), group in grouped:
        row: dict[str, Any] = {
            "algorithm": algorithm,
            "disturbance_type": disturbance_type,
            "disturbance_level": disturbance_level,
            "policy_count": len(group),
        }

        for metric in metric_columns:
            values = pd.to_numeric(
                group[metric],
                errors="raise",
            ).to_numpy(dtype=np.float64)

            row[metric] = float(np.mean(values))

            row[f"std_{metric}_across_policies"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else float("nan")
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        [
            "disturbance_type",
            "disturbance_level",
            "algorithm",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cooperative-transport controllers under "
            "observation noise and action deadzones."
        )
    )

    parser.add_argument(
        "--environment-config",
        type=Path,
        default=Path("configs/cooperative_transport.yaml"),
    )

    parser.add_argument(
        "--maddpg-root",
        type=Path,
        default=Path(
            "results/experiment2a_nominal_official/maddpg"
        ),
    )

    parser.add_argument(
        "--idddpg-root",
        type=Path,
        default=Path(
            "results/experiment2a_nominal_official/"
            "independent_ddpg"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/experiment3_robustness_official"
        ),
    )

    parser.add_argument(
        "--training-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_TRAINING_SEEDS,
    )

    parser.add_argument(
        "--test-seed-start",
        type=int,
        default=300001,
    )

    parser.add_argument(
        "--test-seed-count",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--observation-noise-levels",
        type=float,
        nargs="+",
        default=DEFAULT_OBSERVATION_NOISE_LEVELS,
    )

    parser.add_argument(
        "--action-deadzone-levels",
        type=float,
        nargs="+",
        default=DEFAULT_ACTION_DEADZONE_LEVELS,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_seeds = list(
        range(
            args.test_seed_start,
            args.test_seed_start + args.test_seed_count,
        )
    )

    print("Experiment 3 robustness evaluation")
    print(f"Environment: {args.environment_config}")
    print(f"Training seeds: {args.training_seeds}")
    print(
        "Test seeds: "
        f"{test_seeds[0]}-{test_seeds[-1]}"
    )
    print(
        "Observation noise levels: "
        f"{args.observation_noise_levels}"
    )
    print(
        "Action deadzone levels: "
        f"{args.action_deadzone_levels}"
    )
    print()

    all_rows: list[dict[str, Any]] = []

    start_time = time.perf_counter()

    for algorithm, root in (
        ("maddpg", args.maddpg_root),
        ("independent_ddpg", args.idddpg_root),
    ):
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
                f"{algorithm} seed={training_seed}",
                flush=True,
            )

            learner = make_learner(
                algorithm,
                args.environment_config,
                checkpoint,
                args.device,
            )

            all_rows.extend(
                evaluate_policy_robustness(
                    learner,
                    algorithm=algorithm,
                    training_seed=training_seed,
                    environment_config=args.environment_config,
                    observation_noise_levels=(
                        args.observation_noise_levels
                    ),
                    action_deadzone_levels=(
                        args.action_deadzone_levels
                    ),
                    test_seeds=test_seeds,
                )
            )

    print(
        "handcrafted controller",
        flush=True,
    )

    handcrafted = ForceFeedbackController(
        ForceFeedbackGains()
    )

    all_rows.extend(
        evaluate_policy_robustness(
            handcrafted,
            algorithm="handcrafted",
            training_seed=None,
            environment_config=args.environment_config,
            observation_noise_levels=(
                args.observation_noise_levels
            ),
            action_deadzone_levels=(
                args.action_deadzone_levels
            ),
            test_seeds=test_seeds,
        )
    )

    elapsed_seconds = time.perf_counter() - start_time

    episode_results = pd.DataFrame(all_rows)

    expected_episode_count = (
        (
            len(args.training_seeds) * 2
            + 1
        )
        * len(test_seeds)
        * (
            len(args.observation_noise_levels)
            + len(args.action_deadzone_levels)
        )
    )

    if len(episode_results) != expected_episode_count:
        raise RuntimeError(
            "Unexpected robustness episode count: "
            f"expected {expected_episode_count}, "
            f"got {len(episode_results)}"
        )

    episode_summary = aggregate_results(
        episode_results
    )

    policy_summary = build_policy_summary(
        episode_results
    )

    aggregate_summary = build_aggregate_summary(
        policy_summary
    )

    episode_results.to_csv(
        args.output_dir / "episode_results.csv",
        index=False,
    )

    episode_summary.to_csv(
        args.output_dir / "episode_level_summary.csv",
        index=False,
    )

    policy_summary.to_csv(
        args.output_dir / "policy_summary.csv",
        index=False,
    )

    aggregate_summary.to_csv(
        args.output_dir / "aggregate_summary.csv",
        index=False,
    )

    plot_files = plot_robustness_curves(
        episode_results,
        args.output_dir,
    )

    timing = pd.DataFrame(
        [
            {
                "episode_count": len(episode_results),
                "elapsed_seconds": elapsed_seconds,
                "elapsed_minutes": elapsed_seconds / 60.0,
            }
        ]
    )

    timing.to_csv(
        args.output_dir / "timing_summary.csv",
        index=False,
    )

    print()
    print(
        f"Completed {len(episode_results)} "
        "robustness episodes."
    )
    print(
        f"Elapsed: {elapsed_seconds / 60.0:.2f} min"
    )
    print()

    print("Aggregate summary:")
    print(
        aggregate_summary[
            [
                "algorithm",
                "disturbance_type",
                "disturbance_level",
                "success_rate",
                "mean_episode_return",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved:")
    print(
        f"  {args.output_dir / 'episode_results.csv'}"
    )
    print(
        f"  {args.output_dir / 'episode_level_summary.csv'}"
    )
    print(
        f"  {args.output_dir / 'policy_summary.csv'}"
    )
    print(
        f"  {args.output_dir / 'aggregate_summary.csv'}"
    )
    print(
        f"  {args.output_dir / 'timing_summary.csv'}"
    )

    for plot_file in plot_files:
        print(f"  {plot_file}")


if __name__ == "__main__":
    main()