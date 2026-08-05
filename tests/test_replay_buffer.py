from __future__ import annotations

import numpy as np
import pytest

from src.common.replay_buffer import MultiAgentReplayBuffer, ReplayBatch


CAPACITY = 5
NUM_AGENTS = 3
OBSERVATION_DIM = 18
ACTION_DIM = 2


@pytest.fixture
def buffer() -> MultiAgentReplayBuffer:
    """Create an empty replay buffer for each test."""
    return MultiAgentReplayBuffer(
        capacity=CAPACITY,
        num_agents=NUM_AGENTS,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        seed=42,
    )


def make_transition(
    value: float = 0.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Create one valid multi-agent transition.

    Using a unique value makes it easy to identify whether transitions
    were stored or overwritten correctly.
    """
    observations = np.full(
        (NUM_AGENTS, OBSERVATION_DIM),
        fill_value=value,
        dtype=np.float32,
    )

    actions = np.full(
        (NUM_AGENTS, ACTION_DIM),
        fill_value=value,
        dtype=np.float32,
    )

    rewards = np.full(
        (NUM_AGENTS,),
        fill_value=value,
        dtype=np.float32,
    )

    next_observations = np.full(
        (NUM_AGENTS, OBSERVATION_DIM),
        fill_value=value + 1.0,
        dtype=np.float32,
    )

    terminations = np.zeros(
        (NUM_AGENTS,),
        dtype=np.bool_,
    )

    truncations = np.zeros(
        (NUM_AGENTS,),
        dtype=np.bool_,
    )

    return (
        observations,
        actions,
        rewards,
        next_observations,
        terminations,
        truncations,
    )


def add_transition(
    buffer: MultiAgentReplayBuffer,
    value: float,
) -> None:
    """Create and add one transition to the buffer."""
    (
        observations,
        actions,
        rewards,
        next_observations,
        terminations,
        truncations,
    ) = make_transition(value)

    buffer.add(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminations=terminations,
        truncations=truncations,
    )


def test_initial_state(
    buffer: MultiAgentReplayBuffer,
) -> None:
    assert len(buffer) == 0
    assert buffer.position == 0
    assert not buffer.is_full

    assert buffer.capacity == CAPACITY
    assert buffer.num_agents == NUM_AGENTS
    assert buffer.observation_dim == OBSERVATION_DIM
    assert buffer.action_dim == ACTION_DIM


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("capacity", 0),
        ("num_agents", 0),
        ("observation_dim", 0),
        ("action_dim", 0),
    ],
)
def test_invalid_initialization_raises_error(
    keyword: str,
    value: int,
) -> None:
    arguments = {
        "capacity": CAPACITY,
        "num_agents": NUM_AGENTS,
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
    }

    arguments[keyword] = value

    with pytest.raises(ValueError):
        MultiAgentReplayBuffer(**arguments)


def test_add_one_transition(
    buffer: MultiAgentReplayBuffer,
) -> None:
    transition = make_transition(value=2.0)

    buffer.add(
        observations=transition[0],
        actions=transition[1],
        rewards=transition[2],
        next_observations=transition[3],
        terminations=transition[4],
        truncations=transition[5],
    )

    assert len(buffer) == 1
    assert buffer.position == 1
    assert not buffer.is_full

    np.testing.assert_allclose(
        buffer._observations[0],
        transition[0],
    )

    np.testing.assert_allclose(
        buffer._actions[0],
        transition[1],
    )

    np.testing.assert_allclose(
        buffer._rewards[0],
        transition[2],
    )

    np.testing.assert_allclose(
        buffer._next_observations[0],
        transition[3],
    )

    np.testing.assert_array_equal(
        buffer._terminations[0],
        transition[4],
    )

    np.testing.assert_array_equal(
        buffer._truncations[0],
        transition[5],
    )


def test_add_converts_input_dtypes(
    buffer: MultiAgentReplayBuffer,
) -> None:
    observations = np.zeros(
        (NUM_AGENTS, OBSERVATION_DIM),
        dtype=np.float64,
    )

    actions = np.zeros(
        (NUM_AGENTS, ACTION_DIM),
        dtype=np.float64,
    )

    rewards = np.zeros(
        (NUM_AGENTS,),
        dtype=np.int64,
    )

    next_observations = np.ones(
        (NUM_AGENTS, OBSERVATION_DIM),
        dtype=np.float64,
    )

    terminations = np.zeros(
        (NUM_AGENTS,),
        dtype=np.int64,
    )

    truncations = np.ones(
        (NUM_AGENTS,),
        dtype=np.int64,
    )

    buffer.add(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminations=terminations,
        truncations=truncations,
    )

    assert buffer._observations.dtype == np.float32
    assert buffer._actions.dtype == np.float32
    assert buffer._rewards.dtype == np.float32
    assert buffer._next_observations.dtype == np.float32
    assert buffer._terminations.dtype == np.bool_
    assert buffer._truncations.dtype == np.bool_


def test_buffer_becomes_full(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    assert len(buffer) == CAPACITY
    assert buffer.is_full
    assert buffer.position == 0


def test_circular_buffer_overwrites_oldest_data(
    buffer: MultiAgentReplayBuffer,
) -> None:
    # Store values 0, 1, 2, 3, 4.
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    # These overwrite slots 0 and 1.
    add_transition(buffer, 5.0)
    add_transition(buffer, 6.0)

    assert len(buffer) == CAPACITY
    assert buffer.is_full
    assert buffer.position == 2

    stored_values = set(
        float(value)
        for value in buffer._observations[
            :,
            0,
            0,
        ]
    )

    assert stored_values == {
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    }


def test_sample_returns_correct_shapes(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    batch_size = 3
    batch = buffer.sample(batch_size)

    assert isinstance(batch, ReplayBatch)

    assert batch.observations.shape == (
        batch_size,
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    assert batch.actions.shape == (
        batch_size,
        NUM_AGENTS,
        ACTION_DIM,
    )

    assert batch.rewards.shape == (
        batch_size,
        NUM_AGENTS,
    )

    assert batch.next_observations.shape == (
        batch_size,
        NUM_AGENTS,
        OBSERVATION_DIM,
    )

    assert batch.terminations.shape == (
        batch_size,
        NUM_AGENTS,
    )

    assert batch.truncations.shape == (
        batch_size,
        NUM_AGENTS,
    )


def test_sample_returns_correct_dtypes(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    batch = buffer.sample(batch_size=3)

    assert batch.observations.dtype == np.float32
    assert batch.actions.dtype == np.float32
    assert batch.rewards.dtype == np.float32
    assert batch.next_observations.dtype == np.float32
    assert batch.terminations.dtype == np.bool_
    assert batch.truncations.dtype == np.bool_


def test_sample_returns_contiguous_arrays(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    batch = buffer.sample(batch_size=3)

    assert batch.observations.flags.c_contiguous
    assert batch.actions.flags.c_contiguous
    assert batch.rewards.flags.c_contiguous
    assert batch.next_observations.flags.c_contiguous
    assert batch.terminations.flags.c_contiguous
    assert batch.truncations.flags.c_contiguous


def test_sample_without_replacement_has_unique_transitions(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    batch = buffer.sample(
        batch_size=CAPACITY,
        replace=False,
    )

    sampled_values = batch.observations[
        :,
        0,
        0,
    ]

    assert len(np.unique(sampled_values)) == CAPACITY


def test_sample_with_replacement_allows_large_batch(
    buffer: MultiAgentReplayBuffer,
) -> None:
    add_transition(buffer, 1.0)
    add_transition(buffer, 2.0)

    batch = buffer.sample(
        batch_size=10,
        replace=True,
    )

    assert batch.observations.shape == (
        10,
        NUM_AGENTS,
        OBSERVATION_DIM,
    )


def test_sample_from_empty_buffer_raises_error(
    buffer: MultiAgentReplayBuffer,
) -> None:
    with pytest.raises(
        ValueError,
        match="empty replay buffer",
    ):
        buffer.sample(batch_size=1)


def test_sample_too_many_without_replacement_raises_error(
    buffer: MultiAgentReplayBuffer,
) -> None:
    add_transition(buffer, 1.0)
    add_transition(buffer, 2.0)

    with pytest.raises(
        ValueError,
        match="without replacement",
    ):
        buffer.sample(
            batch_size=3,
            replace=False,
        )


@pytest.mark.parametrize(
    "batch_size",
    [0, -1],
)
def test_invalid_batch_size_raises_error(
    buffer: MultiAgentReplayBuffer,
    batch_size: int,
) -> None:
    add_transition(buffer, 1.0)

    with pytest.raises(
        ValueError,
        match="batch_size must be at least 1",
    ):
        buffer.sample(batch_size=batch_size)


@pytest.mark.parametrize(
    ("field_name", "wrong_shape"),
    [
        (
            "observations",
            (NUM_AGENTS, OBSERVATION_DIM - 1),
        ),
        (
            "actions",
            (NUM_AGENTS, ACTION_DIM + 1),
        ),
        (
            "rewards",
            (NUM_AGENTS, 1),
        ),
        (
            "next_observations",
            (NUM_AGENTS - 1, OBSERVATION_DIM),
        ),
        (
            "terminations",
            (NUM_AGENTS, 1),
        ),
        (
            "truncations",
            (NUM_AGENTS + 1,),
        ),
    ],
)
def test_wrong_transition_shape_raises_error(
    buffer: MultiAgentReplayBuffer,
    field_name: str,
    wrong_shape: tuple[int, ...],
) -> None:
    (
        observations,
        actions,
        rewards,
        next_observations,
        terminations,
        truncations,
    ) = make_transition(1.0)

    values = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminations": terminations,
        "truncations": truncations,
    }

    dtype = (
        np.bool_
        if field_name in {
            "terminations",
            "truncations",
        }
        else np.float32
    )

    values[field_name] = np.zeros(
        wrong_shape,
        dtype=dtype,
    )

    with pytest.raises(
        ValueError,
        match=field_name,
    ):
        buffer.add(**values)


@pytest.mark.parametrize(
    "field_name",
    [
        "observations",
        "actions",
        "rewards",
        "next_observations",
    ],
)
def test_non_finite_float_data_raises_error(
    buffer: MultiAgentReplayBuffer,
    field_name: str,
) -> None:
    (
        observations,
        actions,
        rewards,
        next_observations,
        terminations,
        truncations,
    ) = make_transition(1.0)

    values = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminations": terminations,
        "truncations": truncations,
    }

    invalid_value = values[field_name].copy()
    invalid_value.flat[0] = np.nan
    values[field_name] = invalid_value

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        buffer.add(**values)


def test_batch_dones_combines_termination_and_truncation() -> None:
    batch = ReplayBatch(
        observations=np.zeros(
            (2, NUM_AGENTS, OBSERVATION_DIM),
            dtype=np.float32,
        ),
        actions=np.zeros(
            (2, NUM_AGENTS, ACTION_DIM),
            dtype=np.float32,
        ),
        rewards=np.zeros(
            (2, NUM_AGENTS),
            dtype=np.float32,
        ),
        next_observations=np.zeros(
            (2, NUM_AGENTS, OBSERVATION_DIM),
            dtype=np.float32,
        ),
        terminations=np.asarray(
            [
                [True, False, False],
                [False, False, False],
            ],
            dtype=np.bool_,
        ),
        truncations=np.asarray(
            [
                [False, True, False],
                [False, False, True],
            ],
            dtype=np.bool_,
        ),
    )

    expected = np.asarray(
        [
            [True, True, False],
            [False, False, True],
        ],
        dtype=np.bool_,
    )

    np.testing.assert_array_equal(
        batch.dones,
        expected,
    )


def test_clear_resets_buffer_state(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    assert len(buffer) == CAPACITY
    assert buffer.is_full

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.position == 0
    assert not buffer.is_full

    with pytest.raises(ValueError):
        buffer.sample(batch_size=1)


def test_reseed_makes_sampling_reproducible(
    buffer: MultiAgentReplayBuffer,
) -> None:
    for value in range(CAPACITY):
        add_transition(buffer, float(value))

    buffer.reseed(123)
    first_batch = buffer.sample(
        batch_size=3,
        replace=False,
    )

    buffer.reseed(123)
    second_batch = buffer.sample(
        batch_size=3,
        replace=False,
    )

    np.testing.assert_array_equal(
        first_batch.observations,
        second_batch.observations,
    )

    np.testing.assert_array_equal(
        first_batch.actions,
        second_batch.actions,
    )