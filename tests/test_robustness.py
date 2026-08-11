from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    ActionDeadzone,
    ObservationNoise,
    aggregate_results,
    control_effort,
    delivery_time,
    final_goal_distance,
    force_imbalance,
    payload_stability,
    plot_robustness_curves,
    success_rate,
)


def test_zero_observation_noise_is_identity_copy() -> None:
    observation = np.array([1.0, -2.0, 3.0])
    result = ObservationNoise().apply(observation)
    np.testing.assert_array_equal(result, observation)
    assert result is not observation


def test_seeded_observation_noise_is_reproducible_and_finite() -> None:
    observation = np.arange(8, dtype=float)
    first = ObservationNoise(0.2, seed=11).apply(observation)
    second = ObservationNoise(0.2, seed=11).apply(observation)
    np.testing.assert_array_equal(first, second)
    assert first.shape == observation.shape
    assert np.isfinite(first).all()


def test_observation_noise_rejects_nonflat_observation() -> None:
    with pytest.raises(ValueError, match="flat"):
        ObservationNoise(0.1).apply(np.zeros((2, 2)))


def test_perturbations_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="magnitude must be finite"):
        ObservationNoise(np.nan)

    with pytest.raises(ValueError, match="observation must be finite"):
        ObservationNoise().apply(np.array([0.0, np.nan]))

    with pytest.raises(ValueError, match="threshold must be finite"):
        ActionDeadzone(np.nan)

    with pytest.raises(ValueError, match="action must be finite"):
        ActionDeadzone().apply(np.array([0.0, np.inf]))


def test_zero_deadzone_clips_but_otherwise_preserves_action() -> None:
    action = np.array([-1.5, -0.2, 0.0, 0.8, 1.5])
    result = ActionDeadzone(0.0).apply(action)
    np.testing.assert_array_equal(result, [-1.0, -0.2, 0.0, 0.8, 1.0])


def test_deadzone_is_componentwise_and_keeps_threshold_boundary() -> None:
    action = np.array([[-0.1, -0.09], [0.09, 0.1], [2.0, -2.0]])
    result = ActionDeadzone(0.1).apply(action)
    np.testing.assert_allclose(result, [[-0.1, 0.0], [0.0, 0.1], [1.0, -1.0]])
    assert result.shape == action.shape
    assert np.max(np.abs(result)) <= 1.0


def test_metrics_match_known_synthetic_values() -> None:
    assert success_rate([True, False, True, True]) == pytest.approx(0.75)
    assert final_goal_distance([4.0, 2.0, 0.5]) == pytest.approx(0.5)
    assert delivery_time([10, 20]) == pytest.approx(15.0)
    assert payload_stability([3.0, 0.0], [4.0, 0.0]) == pytest.approx(np.sqrt(12.5))
    assert control_effort([[1.0, 0.0], [0.0, -2.0]]) == pytest.approx(5.0)
    forces = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    assert force_imbalance(forces) == pytest.approx(1.0)


def test_metrics_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="episode lengths must be finite"):
        delivery_time([10, np.nan])


def synthetic_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for algorithm in ("maddpg", "controller"):
        for disturbance_type in ("observation_noise", "action_deadzone"):
            for level in (0.0, 0.2):
                for episode, success in enumerate((True, False)):
                    rows.append(
                        {
                            "algorithm": algorithm,
                            "seed": episode,
                            "disturbance_type": disturbance_type,
                            "disturbance_level": level,
                            "episode": episode,
                            "success": success,
                            "episode_return": 2.0 + 2.0 * episode,
                            "final_goal_distance": 0.5 + episode,
                            "episode_length": 10 + episode,
                            "payload_stability": 0.1 + episode,
                            "control_effort": 3.0 + episode,
                            "force_imbalance": 0.2 + episode,
                        }
                    )
    return rows


def test_aggregation_computes_mean_std_count_and_success_rate() -> None:
    summary = aggregate_results(pd.DataFrame(synthetic_records()))
    assert len(summary) == 8
    row = summary.iloc[0]
    assert row["num_episodes"] == 2
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["mean_episode_return"] == pytest.approx(3.0)
    assert row["std_episode_return"] == pytest.approx(np.sqrt(2.0))


def test_aggregation_supports_custom_seed_grouping() -> None:
    summary = aggregate_results(
        synthetic_records(),
        group_fields=("algorithm", "seed", "disturbance_type", "disturbance_level"),
    )
    assert "seed" in summary
    assert (summary["num_episodes"] == 1).all()
    assert (summary.filter(like="std_") == 0.0).all(axis=None)


def test_plotting_produces_four_nonempty_files(tmp_path: Path) -> None:
    outputs = plot_robustness_curves(synthetic_records(), tmp_path)
    assert len(outputs) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
