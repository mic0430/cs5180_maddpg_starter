from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from src.experiments.train_simple_spread import (
    TrainingConfig,
    build_learner,
    evaluate_simple_spread,
    exploration_noise_for_episode,
    train_simple_spread,
)
from src.envs.simple_spread_v3 import SimpleSpreadWrapper


@pytest.mark.parametrize(
    "algorithm",
    [
        "maddpg",
        "independent_ddpg",
    ],
)
def test_short_training_run_performs_updates_and_logs(
    algorithm: str,
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / f"logs-{algorithm}"
    checkpoint_path = tmp_path / f"{algorithm}.pt"

    config = TrainingConfig(
        algorithm=algorithm,
        seed=7,
        device="cpu",
        num_agents=2,
        max_cycles=3,
        local_ratio=0.5,
        episodes=2,
        batch_size=2,
        replay_capacity=32,
        learning_starts=2,
        update_every=1,
        gamma=0.95,
        tau=0.05,
        actor_lr=1e-3,
        critic_lr=1e-3,
        hidden_sizes=(8, 8),
        exploration_noise_std=0.1,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
    )

    result = train_simple_spread(config)

    assert result.algorithm == algorithm
    assert result.episodes_completed == 2
    assert result.environment_steps > 0
    assert result.update_count > 0
    assert result.log_dir == log_dir
    assert result.checkpoint_path == checkpoint_path

    assert checkpoint_path.exists()

    event_files = list(
        log_dir.glob("events.out.tfevents.*")
    )
    assert event_files


def test_yaml_configuration_loads(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        dedent(
            """
            algorithm: maddpg

            project:
              seed: 123
              device: cpu

            environment:
              num_agents: 2
              max_cycles: 4

            training:
              episodes: 3
              batch_size: 2
              replay_capacity: 20
              learning_starts: 2
              hidden_sizes: [16, 8]

            logging:
              log_dir: test-runs
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    config = TrainingConfig.from_yaml(
        config_path
    )

    assert config.algorithm == "maddpg"
    assert config.seed == 123
    assert config.device == "cpu"
    assert config.num_agents == 2
    assert config.max_cycles == 4
    assert config.episodes == 3
    assert config.batch_size == 2
    assert config.hidden_sizes == (16, 8)
    assert config.log_dir == Path("test-runs")

def test_exploration_noise_decays_linearly() -> None:
    config = TrainingConfig(
        exploration_noise_std=0.2,
        exploration_noise_final=0.05,
        exploration_decay_episodes=10,
    )

    assert exploration_noise_for_episode(
        config,
        0,
    ) == pytest.approx(0.2)

    assert exploration_noise_for_episode(
        config,
        5,
    ) == pytest.approx(0.125)

    assert exploration_noise_for_episode(
        config,
        10,
    ) == pytest.approx(0.05)

    assert exploration_noise_for_episode(
        config,
        20,
    ) == pytest.approx(0.05)

    with pytest.raises(
        ValueError,
        match="episode must be non-negative",
    ):
        exploration_noise_for_episode(
            config,
            -1,
        )


def test_training_runs_periodic_evaluation(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        algorithm="maddpg",
        seed=19,
        device="cpu",
        num_agents=2,
        max_cycles=3,
        local_ratio=0.5,
        episodes=2,
        batch_size=2,
        replay_capacity=32,
        learning_starts=2,
        update_every=1,
        hidden_sizes=(8, 8),
        exploration_noise_std=0.2,
        exploration_noise_final=0.1,
        exploration_decay_episodes=2,
        evaluation_interval=1,
        evaluation_episodes=1,
        log_dir=tmp_path / "evaluation-logs",
        checkpoint_path=(
            tmp_path / "evaluation-model.pt"
        ),
    )

    result = train_simple_spread(config)

    assert result.episodes_completed == 2
    assert result.evaluation_runs == 2
    assert result.mean_evaluation_return is not None

    assert len(
        result.training_episode_returns
    ) == 2

    assert result.evaluation_episode_indices == (
        1,
        2,
    )

    assert len(
        result.evaluation_returns
    ) == 2


def test_evaluation_is_repeatable_without_noise() -> None:
    config = TrainingConfig(
        algorithm="maddpg",
        seed=23,
        device="cpu",
        num_agents=2,
        max_cycles=3,
        local_ratio=0.5,
        batch_size=2,
        replay_capacity=32,
        learning_starts=2,
        hidden_sizes=(8, 8),
        evaluation_interval=1,
        evaluation_episodes=2,
    )

    env = SimpleSpreadWrapper(
        num_agents=config.num_agents,
        max_cycles=config.max_cycles,
        local_ratio=config.local_ratio,
        use_2d_actions=True,
        render_mode=None,
    )

    try:
        learner = build_learner(
            config=config,
            env=env,
        )

        first_return = evaluate_simple_spread(
            learner=learner,
            config=config,
        )

        second_return = evaluate_simple_spread(
            learner=learner,
            config=config,
        )

        assert first_return == pytest.approx(
            second_return
        )
    finally:
        env.close()

