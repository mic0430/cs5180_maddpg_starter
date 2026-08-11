from pathlib import Path

import numpy as np
import pytest
import yaml

from src.envs.transport_env import CooperativeTransportEnv

from src.controllers.force_feedback import (
    ForceFeedbackController,
    ForceFeedbackGains,
)


def test_goal_direction_produces_forward_action():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[6:8] = [1.0, 0.0]

    action = controller.select_action(observation)

    assert action.shape == (2,)
    assert action[0] > 0.0
    assert action[1] == pytest.approx(0.0)


def test_attachment_error_produces_corrective_action():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[2:4] = [0.0, 1.0]

    action = controller.select_action(observation)

    assert action[0] == pytest.approx(0.0)
    assert action[1] > 0.0


def test_local_coupling_force_changes_action():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[8:10] = [0.5, -0.25]

    action = controller.select_action(observation)

    assert action[0] > 0.0
    assert action[1] < 0.0


def test_agent_velocity_is_damped():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[0:2] = [1.0, 0.0]

    action = controller.select_action(observation)

    assert action[0] < 0.0
    assert action[1] == pytest.approx(0.0)


def test_actions_are_bounded():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[6:8] = [100.0, -100.0]

    action = controller.select_action(observation)

    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


def test_select_actions_preserves_agent_dimension():
    controller = ForceFeedbackController()

    observations = np.zeros((2, 12), dtype=np.float64)
    observations[:, 6] = 1.0

    actions = controller.select_actions(observations)

    assert actions.shape == (2, 2)
    assert np.all(actions[:, 0] > 0.0)


def test_invalid_observation_shape_is_rejected():
    controller = ForceFeedbackController()

    with pytest.raises(ValueError):
        controller.select_action(
            np.zeros(11, dtype=np.float64)
        )


def test_nonfinite_observation_is_rejected():
    controller = ForceFeedbackController()

    observation = np.zeros(12, dtype=np.float64)
    observation[0] = np.nan

    with pytest.raises(ValueError):
        controller.select_action(observation)


def test_invalid_action_limit_is_rejected():
    with pytest.raises(ValueError):
        ForceFeedbackController(
            ForceFeedbackGains(
                action_limit=0.0
            )
        )



def test_controller_completes_narrow_passage_smoke():
    config_path = Path(
        "configs/cooperative_transport_narrow_passage.yaml"
    )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    env = CooperativeTransportEnv(config)
    controller = ForceFeedbackController()

    observations, info = env.reset(seed=3)

    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated):
        actions = controller.select_actions(observations)

        assert actions.shape == (
            env.num_agents,
            env.action_dim,
        )
        assert np.all(np.isfinite(actions))
        assert np.all(actions >= env.action_low)
        assert np.all(actions <= env.action_high)

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(actions)

        assert np.all(np.isfinite(observations))
        assert np.all(np.isfinite(rewards))

        steps += 1

        assert steps <= env.max_steps

    assert info["success"] is True
    assert terminated is True
    assert truncated is False
    assert info["collision_count"] == 0
