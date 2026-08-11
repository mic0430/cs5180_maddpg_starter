"""Robustness tools for the cooperative transport experiments."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


class ObservationNoise:
    """Add Gaussian noise to an observation."""

    def __init__(self, magnitude=0.0, seed=None):
        if not np.isfinite(magnitude) or magnitude < 0:
            raise ValueError("magnitude must be finite and non-negative")

        self.magnitude = magnitude
        self.rng = np.random.default_rng(seed)

    def apply(self, observation):
        observation = np.asarray(observation, dtype=float)

        if observation.ndim != 1:
            raise ValueError("observation must be a flat array")

        if not np.isfinite(observation).all():
            raise ValueError("observation must be finite")

        if self.magnitude == 0:
            return observation.copy()

        noise = self.rng.normal(
            0,
            self.magnitude,
            observation.shape,
        )
        return observation + noise


class ActionDeadzone:
    """Set small action values to zero."""

    def __init__(self, threshold=0.0):
        if not np.isfinite(threshold) or threshold < 0 or threshold > 1:
            raise ValueError("threshold must be finite and between 0 and 1")

        self.threshold = threshold

    def apply(self, action):
        action = np.asarray(action, dtype=float)

        if not np.isfinite(action).all():
            raise ValueError("action must be finite")

        action = np.clip(action, -1, 1)
        action[np.abs(action) < self.threshold] = 0
        return action


def check_values(values, name):
    """Convert values to a finite, non-empty NumPy array."""
    values = np.asarray(values, dtype=float)

    if values.size == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")

    return values


def success_rate(successes):
    successes = check_values(successes, "successes")

    if not np.isin(successes, [0, 1]).all():
        raise ValueError("successes must contain True/False or 0/1")

    return float(np.mean(successes))


def final_goal_distance(distances):
    distances = check_values(distances, "distances")
    return float(distances.flatten()[-1])


def delivery_time(episode_lengths):
    episode_lengths = check_values(
        episode_lengths,
        "episode lengths",
    )
    return float(np.mean(episode_lengths))


def payload_stability(orientations, angular_velocities):
    orientations = check_values(orientations, "orientations")
    angular_velocities = check_values(
        angular_velocities,
        "angular velocities",
    )

    if orientations.shape != angular_velocities.shape:
        raise ValueError("orientation and velocity shapes must match")

    stability = np.sqrt(
        np.mean(orientations**2 + angular_velocities**2)
    )
    return float(stability)


def control_effort(actions):
    actions = check_values(actions, "actions")
    return float(np.sum(actions**2))


def force_imbalance(coupling_forces):
    forces = check_values(coupling_forces, "coupling forces")

    # Allow either one step or a full episode.
    if forces.ndim == 2:
        forces = np.expand_dims(forces, axis=0)

    if forces.ndim != 3:
        raise ValueError("forces must have shape (steps, agents, dimensions)")

    average_force = np.mean(forces, axis=1, keepdims=True)
    differences = np.linalg.norm(
        forces - average_force,
        axis=-1,
    )
    return float(np.mean(differences))


DEFAULT_METRICS = [
    "episode_return",
    "final_goal_distance",
    "episode_length",
    "payload_stability",
    "control_effort",
    "force_imbalance",
]


def aggregate_results(
    records,
    group_fields=(
        "algorithm",
        "disturbance_type",
        "disturbance_level",
    ),
    metric_fields=None,
):
    """Calculate summary statistics for episode records."""
    results = pd.DataFrame(records).copy()

    if results.empty:
        raise ValueError("records cannot be empty")

    for field in group_fields:
        if field not in results.columns:
            raise ValueError(f"missing group field: {field}")

    if "success" not in results.columns:
        raise ValueError("records must include success")

    results["success"] = pd.to_numeric(results["success"])

    if not results["success"].isin([0, 1]).all():
        raise ValueError("success must contain True/False or 0/1")

    if metric_fields is None:
        metric_fields = [
            field
            for field in DEFAULT_METRICS
            if field in results.columns
        ]

    for field in metric_fields:
        if field not in results.columns:
            raise ValueError(f"missing metric field: {field}")

        results[field] = pd.to_numeric(results[field])

        if not np.isfinite(results[field]).all():
            raise ValueError(f"{field} must be finite")

    grouped = results.groupby(list(group_fields), dropna=False)
    summary = grouped.size().reset_index(name="num_episodes")
    summary["success_rate"] = grouped["success"].mean().values

    for field in metric_fields:
        summary[f"mean_{field}"] = grouped[field].mean().values
        summary[f"std_{field}"] = (
            grouped[field].std().fillna(0).values
        )

    return summary


def plot_robustness_curves(records, output_dir):
    """Make the four plots needed for the robustness experiment."""
    results = pd.DataFrame(records).copy()

    # Raw episode results need to be summarized first.
    if "success_rate" not in results or "mean_episode_return" not in results:
        results = aggregate_results(results)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "observation_noise",
            "success_rate",
            "Success rate",
            "success_rate_vs_observation_noise.png",
        ),
        (
            "observation_noise",
            "mean_episode_return",
            "Mean episode return",
            "return_vs_observation_noise.png",
        ),
        (
            "action_deadzone",
            "success_rate",
            "Success rate",
            "success_rate_vs_action_deadzone.png",
        ),
        (
            "action_deadzone",
            "mean_episode_return",
            "Mean episode return",
            "return_vs_action_deadzone.png",
        ),
    ]

    output_files = []

    for disturbance, value, label, filename in plots:
        rows = results[results["disturbance_type"] == disturbance]

        if rows.empty:
            raise ValueError(f"no results found for {disturbance}")

        figure, axis = plt.subplots(figsize=(7, 4.5))

        for algorithm, algorithm_rows in rows.groupby("algorithm"):
            algorithm_rows = algorithm_rows.sort_values(
                "disturbance_level"
            )
            axis.plot(
                algorithm_rows["disturbance_level"],
                algorithm_rows[value],
                marker="o",
                label=algorithm,
            )

        axis.set_xlabel("Disturbance level")
        axis.set_ylabel(label)
        axis.set_title(f"{label} vs {disturbance.replace('_', ' ')}")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()

        output_file = output_dir / filename
        figure.savefig(output_file, dpi=150)
        plt.close(figure)
        output_files.append(output_file)

    return output_files
