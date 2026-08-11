
from __future__ import annotations

import pytest
import torch

from src.algorithms.networks import Actor, Critic


OBSERVATION_DIM = 18
ACTION_DIM = 2
NUM_AGENTS = 3
BATCH_SIZE = 8


def test_actor_batch_shape_and_bounds() -> None:
    actor = Actor(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(64, 64),
    )

    observations = torch.randn(
        BATCH_SIZE,
        OBSERVATION_DIM,
    )

    actions = actor(observations)

    assert actions.shape == (
        BATCH_SIZE,
        ACTION_DIM,
    )
    assert torch.all(actions <= 1.0)
    assert torch.all(actions >= -1.0)


def test_actor_accepts_single_observation() -> None:
    actor = Actor(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )

    observation = torch.randn(
        OBSERVATION_DIM
    )

    action = actor(observation)

    assert action.shape == (ACTION_DIM,)


def test_centralized_critic_batch_shape() -> None:
    joint_observation_dim = (
        NUM_AGENTS * OBSERVATION_DIM
    )
    joint_action_dim = (
        NUM_AGENTS * ACTION_DIM
    )

    critic = Critic(
        observation_dim=joint_observation_dim,
        action_dim=joint_action_dim,
        hidden_sizes=(64, 64),
    )

    observations = torch.randn(
        BATCH_SIZE,
        joint_observation_dim,
    )
    actions = torch.randn(
        BATCH_SIZE,
        joint_action_dim,
    )

    q_values = critic(
        observations,
        actions,
    )

    assert q_values.shape == (
        BATCH_SIZE,
        1,
    )


def test_critic_accepts_single_transition() -> None:
    critic = Critic(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )

    observation = torch.randn(
        OBSERVATION_DIM
    )
    action = torch.randn(
        ACTION_DIM
    )

    q_value = critic(
        observation,
        action,
    )

    assert q_value.shape == (1, 1)


def test_networks_run_explicitly_on_cpu() -> None:
    actor = Actor(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(32, 16),
    ).to("cpu")

    critic = Critic(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(32, 16),
    ).to("cpu")

    observations = torch.randn(
        BATCH_SIZE,
        OBSERVATION_DIM,
        device="cpu",
    )
    actions = actor(observations)
    q_values = critic(
        observations,
        actions,
    )

    assert actions.device.type == "cpu"
    assert q_values.device.type == "cpu"
    assert q_values.shape == (
        BATCH_SIZE,
        1,
    )


@pytest.mark.parametrize(
    (
        "observation_dim",
        "action_dim",
    ),
    [
        (0, 2),
        (18, 0),
        (-1, 2),
        (18, -1),
    ],
)
def test_actor_rejects_invalid_dimensions(
    observation_dim: int,
    action_dim: int,
) -> None:
    with pytest.raises(ValueError):
        Actor(
            observation_dim=observation_dim,
            action_dim=action_dim,
        )


def test_actor_rejects_wrong_input_shape() -> None:
    actor = Actor(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )

    invalid_observations = torch.randn(
        BATCH_SIZE,
        10,
    )

    with pytest.raises(
        ValueError,
        match="expected observation dimension",
    ):
        actor(invalid_observations)


def test_critic_rejects_mismatched_batch_sizes() -> None:
    critic = Critic(
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
    )

    observations = torch.randn(
        8,
        OBSERVATION_DIM,
    )
    actions = torch.randn(
        7,
        ACTION_DIM,
    )

    with pytest.raises(
        ValueError,
        match="batch sizes must match",
    ):
        critic(
            observations,
            actions,
        )
