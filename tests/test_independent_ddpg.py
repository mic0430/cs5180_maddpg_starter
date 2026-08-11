
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.algorithms.independent_ddpg import IndependentDDPG
from src.common.replay_buffer import ReplayBatch


NUM_AGENTS = 3
OBSERVATION_DIM = 18
ACTION_DIM = 2
BATCH_SIZE = 16


def make_batch(
    seed: int = 42,
) -> ReplayBatch:
    rng = np.random.default_rng(seed)

    observations = rng.normal(
        size=(
            BATCH_SIZE,
            NUM_AGENTS,
            OBSERVATION_DIM,
        )
    ).astype(np.float32)

    actions = rng.uniform(
        low=-1.0,
        high=1.0,
        size=(
            BATCH_SIZE,
            NUM_AGENTS,
            ACTION_DIM,
        ),
    ).astype(np.float32)

    rewards = rng.normal(
        size=(
            BATCH_SIZE,
            NUM_AGENTS,
        )
    ).astype(np.float32)

    next_observations = rng.normal(
        size=(
            BATCH_SIZE,
            NUM_AGENTS,
            OBSERVATION_DIM,
        )
    ).astype(np.float32)

    terminations = np.zeros(
        (
            BATCH_SIZE,
            NUM_AGENTS,
        ),
        dtype=np.bool_,
    )
    truncations = np.zeros_like(
        terminations
    )

    return ReplayBatch(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminations=terminations,
        truncations=truncations,
    )


def parameter_vector(
    module: nn.Module,
) -> torch.Tensor:
    return torch.cat(
        [
            parameter.detach().cpu().reshape(-1)
            for parameter in module.parameters()
        ]
    )


@pytest.fixture
def independent_ddpg() -> IndependentDDPG:
    torch.manual_seed(42)

    return IndependentDDPG(
        num_agents=NUM_AGENTS,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(32, 32),
        actor_lr=1e-2,
        critic_lr=1e-2,
        gamma=0.95,
        tau=0.05,
        exploration_noise_std=0.2,
        device="cpu",
    )


def test_critics_use_only_local_dimensions(
    independent_ddpg: IndependentDDPG,
) -> None:
    for critic in independent_ddpg.critics:
        assert (
            critic.observation_dim
            == OBSERVATION_DIM
        )
        assert critic.action_dim == ACTION_DIM

        assert (
            critic.observation_dim
            != NUM_AGENTS * OBSERVATION_DIM
        )


def test_initial_targets_match_online_networks(
    independent_ddpg: IndependentDDPG,
) -> None:
    for online, target in zip(
        independent_ddpg.actors,
        independent_ddpg.target_actors,
    ):
        torch.testing.assert_close(
            parameter_vector(online),
            parameter_vector(target),
        )

        for online_parameter, target_parameter in zip(
            online.parameters(),
            target.parameters(),
        ):
            assert (
                online_parameter.data_ptr()
                != target_parameter.data_ptr()
            )


def test_select_actions_shape_and_bounds(
    independent_ddpg: IndependentDDPG,
) -> None:
    observations = np.zeros(
        (
            NUM_AGENTS,
            OBSERVATION_DIM,
        ),
        dtype=np.float32,
    )

    first = independent_ddpg.select_actions(
        observations,
        explore=False,
    )
    second = independent_ddpg.select_actions(
        observations,
        explore=False,
    )

    assert first.shape == (
        NUM_AGENTS,
        ACTION_DIM,
    )
    assert first.dtype == np.float32
    assert np.all(first >= -1.0)
    assert np.all(first <= 1.0)

    np.testing.assert_array_equal(
        first,
        second,
    )


def test_update_changes_online_and_target_parameters(
    independent_ddpg: IndependentDDPG,
) -> None:
    batch = make_batch()

    online_before = parameter_vector(
        independent_ddpg.actors[0]
    )
    target_before = parameter_vector(
        independent_ddpg.target_actors[0]
    )

    statistics = independent_ddpg.update(batch)

    online_after = parameter_vector(
        independent_ddpg.actors[0]
    )
    target_after = parameter_vector(
        independent_ddpg.target_actors[0]
    )

    online_change = torch.linalg.vector_norm(
        online_after - online_before
    )
    target_change = torch.linalg.vector_norm(
        target_after - target_before
    )

    assert online_change.item() > 0.0
    assert target_change.item() > 0.0
    assert target_change.item() < online_change.item()

    assert set(statistics) == {
        "actor_loss",
        "critic_loss",
        "mean_q",
        "mean_target_q",
    }

    assert all(
        np.isfinite(value)
        for value in statistics.values()
    )


def test_terminations_disable_bootstrapping(
    independent_ddpg: IndependentDDPG,
) -> None:
    batch = make_batch()

    terminated_batch = ReplayBatch(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        next_observations=batch.next_observations,
        terminations=np.ones_like(
            batch.terminations
        ),
        truncations=batch.truncations,
    )

    observations = torch.as_tensor(
        terminated_batch.observations,
        dtype=torch.float32,
    )
    rewards = torch.as_tensor(
        terminated_batch.rewards,
        dtype=torch.float32,
    )
    terminations = torch.as_tensor(
        terminated_batch.terminations,
        dtype=torch.bool,
    )

    agent_index = 0

    with torch.no_grad():
        next_actions = (
            independent_ddpg.target_actors[
                agent_index
            ](
                observations[
                    :,
                    agent_index,
                    :,
                ]
            )
        )
        next_q = (
            independent_ddpg.target_critics[
                agent_index
            ](
                observations[
                    :,
                    agent_index,
                    :,
                ],
                next_actions,
            )
        )

        target_q = (
            rewards[
                :,
                agent_index,
            ].unsqueeze(-1)
            + independent_ddpg.gamma
            * (
                1.0
                - terminations[
                    :,
                    agent_index,
                ].to(torch.float32).unsqueeze(-1)
            )
            * next_q
        )

    expected = rewards[
        :,
        agent_index,
    ].unsqueeze(-1)

    torch.testing.assert_close(
        target_q,
        expected,
    )


def test_save_and_load_restores_parameters(
    independent_ddpg: IndependentDDPG,
    tmp_path: Path,
) -> None:
    independent_ddpg.update(
        make_batch()
    )

    checkpoint_path = (
        tmp_path / "independent_ddpg.pt"
    )
    independent_ddpg.save(
        checkpoint_path
    )

    restored = IndependentDDPG(
        num_agents=NUM_AGENTS,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=(32, 32),
        actor_lr=1e-2,
        critic_lr=1e-2,
        gamma=0.95,
        tau=0.05,
        device="cpu",
    )
    restored.load(checkpoint_path)

    for original, loaded in zip(
        independent_ddpg.actors,
        restored.actors,
    ):
        torch.testing.assert_close(
            parameter_vector(original),
            parameter_vector(loaded),
        )


def test_invalid_batch_shape_raises_error(
    independent_ddpg: IndependentDDPG,
) -> None:
    batch = make_batch()

    invalid = ReplayBatch(
        observations=batch.observations[
            :,
            :2,
            :,
        ],
        actions=batch.actions,
        rewards=batch.rewards,
        next_observations=(
            batch.next_observations
        ),
        terminations=batch.terminations,
        truncations=batch.truncations,
    )

    with pytest.raises(
        ValueError,
        match="observations must have shape",
    ):
        independent_ddpg.update(invalid)
