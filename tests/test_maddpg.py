
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.algorithms.maddpg import MADDPG
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
def maddpg() -> MADDPG:
    torch.manual_seed(42)

    return MADDPG(
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


def test_initial_targets_match_online_networks(
    maddpg: MADDPG,
) -> None:
    for online, target in zip(
        maddpg.actors,
        maddpg.target_actors,
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

    for online, target in zip(
        maddpg.critics,
        maddpg.target_critics,
    ):
        torch.testing.assert_close(
            parameter_vector(online),
            parameter_vector(target),
        )


def test_select_actions_shape_bounds_and_determinism(
    maddpg: MADDPG,
) -> None:
    observations = np.zeros(
        (
            NUM_AGENTS,
            OBSERVATION_DIM,
        ),
        dtype=np.float32,
    )

    first = maddpg.select_actions(
        observations,
        explore=False,
    )
    second = maddpg.select_actions(
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


def test_exploration_actions_remain_bounded(
    maddpg: MADDPG,
) -> None:
    observations = np.zeros(
        (
            NUM_AGENTS,
            OBSERVATION_DIM,
        ),
        dtype=np.float32,
    )

    actions = maddpg.select_actions(
        observations,
        explore=True,
        noise_std=100.0,
    )

    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)


def test_update_changes_online_and_slowly_updates_target(
    maddpg: MADDPG,
) -> None:
    batch = make_batch()

    online_before = parameter_vector(
        maddpg.actors[0]
    )
    target_before = parameter_vector(
        maddpg.target_actors[0]
    )

    statistics = maddpg.update(batch)

    online_after = parameter_vector(
        maddpg.actors[0]
    )
    target_after = parameter_vector(
        maddpg.target_actors[0]
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

    torch.testing.assert_close(
        target_after - target_before,
        maddpg.tau
        * (online_after - online_before),
        rtol=1e-4,
        atol=1e-6,
    )

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
    maddpg: MADDPG,
) -> None:
    batch = make_batch()

    next_observations = torch.as_tensor(
        batch.next_observations,
        dtype=torch.float32,
    )
    next_joint_observations = (
        next_observations.reshape(
            BATCH_SIZE,
            -1,
        )
    )
    rewards = torch.as_tensor(
        batch.rewards,
        dtype=torch.float32,
    )
    terminations = torch.ones(
        (
            BATCH_SIZE,
            NUM_AGENTS,
        ),
        dtype=torch.bool,
    )

    target_q = maddpg._target_q_values(
        agent_index=0,
        next_observations=next_observations,
        next_joint_observations=(
            next_joint_observations
        ),
        rewards=rewards,
        terminations=terminations,
    )

    expected = rewards[
        :,
        0,
    ].unsqueeze(-1)

    torch.testing.assert_close(
        target_q,
        expected,
    )


def test_save_and_load_restores_parameters(
    maddpg: MADDPG,
    tmp_path: Path,
) -> None:
    batch = make_batch()
    maddpg.update(batch)

    checkpoint_path = (
        tmp_path / "maddpg.pt"
    )
    maddpg.save(checkpoint_path)

    restored = MADDPG(
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
        maddpg.actors,
        restored.actors,
    ):
        torch.testing.assert_close(
            parameter_vector(original),
            parameter_vector(loaded),
        )


def test_invalid_batch_shape_raises_error(
    maddpg: MADDPG,
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
        maddpg.update(invalid)
