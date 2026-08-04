from __future__ import annotations

import numpy as np
import pytest

from src.envs.simple_spread_v3 import SimpleSpreadWrapper


NUM_AGENTS = 3
OBSERVATION_DIM = 18


@pytest.fixture
def env_2d() -> SimpleSpreadWrapper:
    """Create a wrapper whose actor outputs 2D force actions."""
    env = SimpleSpreadWrapper(
        num_agents=NUM_AGENTS,
        max_cycles=5,
        local_ratio=0.5,
        use_2d_actions=True,
        render_mode=None,
    )

    yield env

    env.close()


@pytest.fixture
def env_5d() -> SimpleSpreadWrapper:
    """Create a wrapper whose actor outputs MPE2's 5D actions."""
    env = SimpleSpreadWrapper(
        num_agents=NUM_AGENTS,
        max_cycles=5,
        local_ratio=0.5,
        use_2d_actions=False,
        render_mode=None,
    )

    yield env

    env.close()


def test_initial_dimensions(
    env_2d: SimpleSpreadWrapper,
) -> None:
    """The wrapper should expose the expected environment dimensions."""
    assert env_2d.num_agents == NUM_AGENTS
    assert env_2d.observation_dim == OBSERVATION_DIM
    assert env_2d.environment_action_dim == 5
    assert env_2d.action_dim == 2

    assert env_2d.global_observation_dim == (
        NUM_AGENTS * OBSERVATION_DIM
    )

    assert env_2d.joint_action_dim == (
        NUM_AGENTS * 2
    )

    assert env_2d.agent_names == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]


def test_reset_returns_correct_shape_and_dtype(
    env_2d: SimpleSpreadWrapper,
) -> None:
    observations, infos = env_2d.reset(seed=42)

    assert observations.shape == (
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    assert observations.dtype == np.float32
    assert observations.flags.c_contiguous

    assert isinstance(infos, dict)


def test_reset_is_reproducible_with_same_seed(
    env_2d: SimpleSpreadWrapper,
) -> None:
    observations_1, _ = env_2d.reset(seed=42)
    observations_2, _ = env_2d.reset(seed=42)

    np.testing.assert_allclose(
        observations_1,
        observations_2,
    )


def test_step_with_zero_2d_actions(
    env_2d: SimpleSpreadWrapper,
) -> None:
    env_2d.reset(seed=42)

    actions = np.zeros(
        (NUM_AGENTS, 2),
        dtype=np.float32,
    )

    (
        next_observations,
        rewards,
        terminations,
        truncations,
        infos,
    ) = env_2d.step(actions)

    assert next_observations.shape == (
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    assert rewards.shape == (NUM_AGENTS,)
    assert terminations.shape == (NUM_AGENTS,)
    assert truncations.shape == (NUM_AGENTS,)

    assert next_observations.dtype == np.float32
    assert rewards.dtype == np.float32
    assert terminations.dtype == np.bool_
    assert truncations.dtype == np.bool_

    assert next_observations.flags.c_contiguous
    assert rewards.flags.c_contiguous
    assert terminations.flags.c_contiguous
    assert truncations.flags.c_contiguous

    assert isinstance(infos, dict)


def test_convert_2d_actions(
    env_2d: SimpleSpreadWrapper,
) -> None:
    actions = np.asarray(
        [
            [-1.0, 1.0],
            [0.5, -0.25],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    converted = env_2d._convert_2d_actions(actions)

    expected = np.asarray(
        [
            # no-op, left, right, down, up
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.5, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert converted.shape == (NUM_AGENTS, 5)
    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous

    np.testing.assert_allclose(
        converted,
        expected,
    )


def test_convert_2d_actions_clips_out_of_range_values(
    env_2d: SimpleSpreadWrapper,
) -> None:
    actions = np.asarray(
        [
            [-2.0, 3.0],
            [4.0, -5.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    converted = env_2d._convert_2d_actions(actions)

    expected = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        converted,
        expected,
    )


def test_scale_tanh_actions_to_environment_bounds(
    env_5d: SimpleSpreadWrapper,
) -> None:
    actions = np.asarray(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [1.0, 0.5, 0.0, -0.5, -1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    scaled = env_5d._scale_actor_actions(actions)

    expected = np.asarray(
        [
            [0.0, 0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    assert scaled.shape == (NUM_AGENTS, 5)
    assert scaled.dtype == np.float32
    assert scaled.flags.c_contiguous

    np.testing.assert_allclose(
        scaled,
        expected,
        atol=1e-6,
    )


def test_step_with_5d_actor_actions(
    env_5d: SimpleSpreadWrapper,
) -> None:
    env_5d.reset(seed=42)

    actions = np.zeros(
        (NUM_AGENTS, 5),
        dtype=np.float32,
    )

    (
        next_observations,
        rewards,
        terminations,
        truncations,
        infos,
    ) = env_5d.step(actions)

    assert next_observations.shape == (
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    assert rewards.shape == (NUM_AGENTS,)
    assert terminations.shape == (NUM_AGENTS,)
    assert truncations.shape == (NUM_AGENTS,)
    assert isinstance(infos, dict)


def test_invalid_action_shape_raises_error(
    env_2d: SimpleSpreadWrapper,
) -> None:
    env_2d.reset(seed=42)

    invalid_actions = np.zeros(
        (NUM_AGENTS, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Expected actions with shape",
    ):
        env_2d.step(invalid_actions)


def test_nan_actions_raise_error(
    env_2d: SimpleSpreadWrapper,
) -> None:
    env_2d.reset(seed=42)

    actions = np.zeros(
        (NUM_AGENTS, 2),
        dtype=np.float32,
    )

    actions[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        env_2d.step(actions)


def test_infinite_actions_raise_error(
    env_2d: SimpleSpreadWrapper,
) -> None:
    env_2d.reset(seed=42)

    actions = np.zeros(
        (NUM_AGENTS, 2),
        dtype=np.float32,
    )

    actions[1, 1] = np.inf

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        env_2d.step(actions)


def test_global_state(
    env_2d: SimpleSpreadWrapper,
) -> None:
    observations, _ = env_2d.reset(seed=42)

    global_state = env_2d.global_state(
        observations
    )

    assert global_state.shape == (
        NUM_AGENTS * OBSERVATION_DIM,
    )

    assert global_state.dtype == np.float32
    assert global_state.flags.c_contiguous

    np.testing.assert_allclose(
        global_state,
        observations.reshape(-1),
    )


def test_global_state_rejects_wrong_shape(
    env_2d: SimpleSpreadWrapper,
) -> None:
    invalid_observations = np.zeros(
        (NUM_AGENTS, 10),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Expected observations with shape",
    ):
        env_2d.global_state(
            invalid_observations
        )


def test_episode_reaches_truncation(
    env_2d: SimpleSpreadWrapper,
) -> None:
    env_2d.reset(seed=42)

    actions = np.zeros(
        (NUM_AGENTS, 2),
        dtype=np.float32,
    )

    episode_finished = False

    for _ in range(5):
        (
            _,
            _,
            terminations,
            truncations,
            _,
        ) = env_2d.step(actions)

        done = np.logical_or(
            terminations,
            truncations,
        )

        if done.all():
            episode_finished = True
            break

    assert episode_finished
    assert truncations.all()


def test_multiple_random_steps_do_not_crash(
    env_2d: SimpleSpreadWrapper,
) -> None:
    rng = np.random.default_rng(42)

    observations, _ = env_2d.reset(seed=42)

    assert observations.shape == (
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    for _ in range(5):
        actions = rng.uniform(
            low=-1.0,
            high=1.0,
            size=(NUM_AGENTS, 2),
        ).astype(np.float32)

        (
            observations,
            rewards,
            terminations,
            truncations,
            _,
        ) = env_2d.step(actions)

        assert observations.shape == (
            NUM_AGENTS,
            OBSERVATION_DIM,
        )

        assert rewards.shape == (NUM_AGENTS,)

        done = np.logical_or(
            terminations,
            truncations,
        )

        if done.all():
            break