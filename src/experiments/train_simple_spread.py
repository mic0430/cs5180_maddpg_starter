from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.algorithms.independent_ddpg import IndependentDDPG
from src.algorithms.maddpg import MADDPG
from src.common.replay_buffer import MultiAgentReplayBuffer
from src.common.seed import seed_everything
from src.envs.simple_spread_v3 import SimpleSpreadWrapper


@dataclass
class TrainingConfig:
    """Configuration for a Simple Spread training run."""

    algorithm: str = "maddpg"
    seed: int = 42
    device: str = "auto"

    num_agents: int = 3
    max_cycles: int = 25
    local_ratio: float = 0.5

    episodes: int = 100
    batch_size: int = 128
    replay_capacity: int = 100_000
    learning_starts: int = 1_024
    update_every: int = 1

    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 64)

    exploration_noise_std: float = 0.1
    exploration_noise_final: float = 0.1
    exploration_decay_episodes: int = 0

    evaluation_interval: int = 0
    evaluation_episodes: int = 0

    log_dir: Path | str = Path("runs/simple_spread")
    checkpoint_path: Path | str | None = None

    def __post_init__(self) -> None:
        self.algorithm = self.algorithm.strip().lower()
        self.log_dir = Path(self.log_dir)

        if self.checkpoint_path is not None:
            self.checkpoint_path = Path(
                self.checkpoint_path
            )

        if self.algorithm not in {
            "maddpg",
            "independent_ddpg",
        }:
            raise ValueError(
                "algorithm must be 'maddpg' or "
                "'independent_ddpg'."
            )

        for name in (
            "num_agents",
            "max_cycles",
            "episodes",
            "batch_size",
            "replay_capacity",
            "update_every",
        ):
            value = getattr(self, name)

            if value < 1:
                raise ValueError(
                    f"{name} must be at least 1."
                )

        if self.learning_starts < 0:
            raise ValueError(
                "learning_starts must be non-negative."
            )

        for name in (
            "exploration_decay_episodes",
            "evaluation_interval",
            "evaluation_episodes",
        ):
            value = getattr(self, name)

            if value < 0:
                raise ValueError(
                    f"{name} must be non-negative."
                )

        if self.exploration_noise_std < 0.0:
            raise ValueError(
                "exploration_noise_std must be "
                "non-negative."
            )

        if self.exploration_noise_final < 0.0:
            raise ValueError(
                "exploration_noise_final must be "
                "non-negative."
            )

        if (
            self.exploration_noise_final
            > self.exploration_noise_std
        ):
            raise ValueError(
                "exploration_noise_final must not exceed "
                "exploration_noise_std."
            )

        evaluation_is_partially_enabled = (
            self.evaluation_interval == 0
        ) != (
            self.evaluation_episodes == 0
        )

        if evaluation_is_partially_enabled:
            raise ValueError(
                "evaluation_interval and "
                "evaluation_episodes must either both "
                "be zero or both be positive."
            )

        if self.replay_capacity < self.batch_size:
            raise ValueError(
                "replay_capacity must be at least "
                "batch_size."
            )

        if not 0.0 <= self.local_ratio <= 1.0:
            raise ValueError(
                "local_ratio must be between 0 and 1."
            )

        self.hidden_sizes = tuple(
            int(size)
            for size in self.hidden_sizes
        )

        if not self.hidden_sizes:
            raise ValueError(
                "hidden_sizes must not be empty."
            )

        if any(
            size < 1
            for size in self.hidden_sizes
        ):
            raise ValueError(
                "All hidden sizes must be positive."
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> TrainingConfig:
        """Load a training configuration from YAML."""
        config_path = Path(path)

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data: dict[str, Any] = (
                yaml.safe_load(file) or {}
            )

        project = data.get("project", {})
        environment = data.get("environment", {})
        training = data.get("training", {})
        logging = data.get("logging", {})

        initial_noise = training.get(
            "exploration_noise_std",
            0.1,
        )

        return cls(
            algorithm=data.get(
                "algorithm",
                "maddpg",
            ),
            seed=project.get("seed", 42),
            device=project.get("device", "auto"),
            num_agents=environment.get(
                "num_agents",
                3,
            ),
            max_cycles=environment.get(
                "max_cycles",
                25,
            ),
            local_ratio=environment.get(
                "local_ratio",
                0.5,
            ),
            episodes=training.get(
                "episodes",
                100,
            ),
            batch_size=training.get(
                "batch_size",
                128,
            ),
            replay_capacity=training.get(
                "replay_capacity",
                100_000,
            ),
            learning_starts=training.get(
                "learning_starts",
                1_024,
            ),
            update_every=training.get(
                "update_every",
                1,
            ),
            gamma=training.get(
                "gamma",
                0.95,
            ),
            tau=training.get(
                "tau",
                0.01,
            ),
            actor_lr=training.get(
                "actor_lr",
                1e-3,
            ),
            critic_lr=training.get(
                "critic_lr",
                1e-3,
            ),
            hidden_sizes=tuple(
                training.get(
                    "hidden_sizes",
                    [64, 64],
                )
            ),
            exploration_noise_std=initial_noise,
            exploration_noise_final=training.get(
                "exploration_noise_final",
                initial_noise,
            ),
            exploration_decay_episodes=training.get(
                "exploration_decay_episodes",
                0,
            ),
            evaluation_interval=training.get(
                "evaluation_interval",
                0,
            ),
            evaluation_episodes=training.get(
                "evaluation_episodes",
                0,
            ),
            log_dir=logging.get(
                "log_dir",
                "runs/simple_spread",
            ),
            checkpoint_path=logging.get(
                "checkpoint_path",
            ),
        )


@dataclass(frozen=True)
class TrainingResult:
    """Summary returned by a completed training run."""

    algorithm: str
    episodes_completed: int
    environment_steps: int
    update_count: int
    mean_episode_return: float
    mean_evaluation_return: float | None
    evaluation_runs: int
    training_episode_returns: tuple[float, ...]
    evaluation_episode_indices: tuple[int, ...]
    evaluation_returns: tuple[float, ...]
    log_dir: Path
    checkpoint_path: Path | None


Learner = MADDPG | IndependentDDPG


def exploration_noise_for_episode(
    config: TrainingConfig,
    episode: int,
) -> float:
    """Return linearly decayed exploration noise."""

    if episode < 0:
        raise ValueError(
            "episode must be non-negative."
        )

    if config.exploration_decay_episodes == 0:
        return config.exploration_noise_std

    progress = min(
        episode / config.exploration_decay_episodes,
        1.0,
    )

    noise_std = (
        config.exploration_noise_std
        + progress
        * (
            config.exploration_noise_final
            - config.exploration_noise_std
        )
    )

    return float(noise_std)


def build_learner(
    config: TrainingConfig,
    env: SimpleSpreadWrapper,
) -> Learner:
    """Create the requested learning algorithm."""
    common_arguments = {
        "num_agents": env.num_agents,
        "observation_dim": env.observation_dim,
        "action_dim": env.action_dim,
        "hidden_sizes": config.hidden_sizes,
        "actor_lr": config.actor_lr,
        "critic_lr": config.critic_lr,
        "gamma": config.gamma,
        "tau": config.tau,
        "exploration_noise_std": (
            config.exploration_noise_std
        ),
        "device": config.device,
    }

    if config.algorithm == "maddpg":
        return MADDPG(**common_arguments)

    if config.algorithm == "independent_ddpg":
        return IndependentDDPG(**common_arguments)

    raise ValueError(
        f"Unsupported algorithm: {config.algorithm}"
    )


def evaluate_simple_spread(
    learner: Learner,
    config: TrainingConfig,
) -> float:
    """Evaluate the current policy without exploration noise."""

    if config.evaluation_episodes < 1:
        raise ValueError(
            "evaluation_episodes must be positive "
            "when evaluation is performed."
        )

    evaluation_env = SimpleSpreadWrapper(
        num_agents=config.num_agents,
        max_cycles=config.max_cycles,
        local_ratio=config.local_ratio,
        use_2d_actions=True,
        render_mode=None,
    )

    evaluation_returns: list[float] = []
    evaluation_seed_base = config.seed + 100_000

    try:
        for evaluation_episode in range(
            config.evaluation_episodes
        ):
            observations, _ = evaluation_env.reset(
                seed=(
                    evaluation_seed_base
                    + evaluation_episode
                )
            )

            agent_returns = np.zeros(
                evaluation_env.num_agents,
                dtype=np.float32,
            )

            for _ in range(config.max_cycles):
                actions = learner.select_actions(
                    observations,
                    explore=False,
                )

                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    _,
                ) = evaluation_env.step(actions)

                agent_returns += rewards

                done = np.logical_or(
                    terminations,
                    truncations,
                )

                if bool(np.all(done)):
                    break

            evaluation_returns.append(
                float(np.mean(agent_returns))
            )

        return float(
            np.mean(evaluation_returns)
        )
    finally:
        evaluation_env.close()


def train_simple_spread(
    config: TrainingConfig,
) -> TrainingResult:
    """Train MADDPG or independent DDPG on Simple Spread."""
    seed_everything(config.seed)

    random_generator = np.random.default_rng(
        config.seed
    )

    config.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=str(config.log_dir)
    )

    env = SimpleSpreadWrapper(
        num_agents=config.num_agents,
        max_cycles=config.max_cycles,
        local_ratio=config.local_ratio,
        use_2d_actions=True,
        render_mode=None,
    )

    learner = build_learner(
        config=config,
        env=env,
    )

    replay_buffer = MultiAgentReplayBuffer(
        capacity=config.replay_capacity,
        num_agents=env.num_agents,
        observation_dim=env.observation_dim,
        action_dim=env.action_dim,
        seed=config.seed,
    )

    global_step = 0
    update_count = 0
    episode_returns: list[float] = []
    evaluation_episode_indices: list[int] = []
    evaluation_returns: list[float] = []

    minimum_replay_size = max(
        config.batch_size,
        config.learning_starts,
    )

    evaluation_enabled = (
        config.evaluation_interval > 0
        and config.evaluation_episodes > 0
    )

    try:
        for episode in range(config.episodes):
            current_noise_std = (
                exploration_noise_for_episode(
                    config,
                    episode,
                )
            )

            observations, _ = env.reset(
                seed=config.seed + episode
            )

            agent_returns = np.zeros(
                env.num_agents,
                dtype=np.float32,
            )

            episode_length = 0

            for _ in range(config.max_cycles):
                if global_step < config.learning_starts:
                    actions = random_generator.uniform(
                        low=-1.0,
                        high=1.0,
                        size=(
                            env.num_agents,
                            env.action_dim,
                        ),
                    ).astype(np.float32)
                else:
                    actions = learner.select_actions(
                        observations,
                        explore=True,
                        noise_std=current_noise_std,
                    )

                (
                    next_observations,
                    rewards,
                    terminations,
                    truncations,
                    _,
                ) = env.step(actions)

                replay_buffer.add(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    next_observations=next_observations,
                    terminations=terminations,
                    truncations=truncations,
                )

                observations = next_observations
                agent_returns += rewards

                global_step += 1
                episode_length += 1

                if (
                    len(replay_buffer)
                    >= minimum_replay_size
                    and global_step
                    % config.update_every
                    == 0
                ):
                    batch = replay_buffer.sample(
                        batch_size=config.batch_size,
                        replace=False,
                    )

                    statistics = learner.update(batch)
                    update_count += 1

                    for name, value in (
                        statistics.items()
                    ):
                        writer.add_scalar(
                            f"losses/{name}",
                            value,
                            global_step,
                        )

                done = np.logical_or(
                    terminations,
                    truncations,
                )

                if bool(np.all(done)):
                    break

            mean_agent_return = float(
                np.mean(agent_returns)
            )
            episode_returns.append(
                mean_agent_return
            )

            writer.add_scalar(
                "train/episode_return_mean_agent",
                mean_agent_return,
                episode,
            )
            writer.add_scalar(
                "train/episode_return_sum_agents",
                float(np.sum(agent_returns)),
                episode,
            )
            writer.add_scalar(
                "train/episode_length",
                episode_length,
                episode,
            )
            writer.add_scalar(
                "train/replay_size",
                len(replay_buffer),
                episode,
            )
            writer.add_scalar(
                "train/update_count",
                update_count,
                episode,
            )
            writer.add_scalar(
                "train/exploration_noise_std",
                current_noise_std,
                episode,
            )

            should_evaluate = (
                evaluation_enabled
                and (
                    episode + 1
                )
                % config.evaluation_interval
                == 0
            )

            if should_evaluate:
                evaluation_return = (
                    evaluate_simple_spread(
                        learner=learner,
                        config=config,
                    )
                )

                evaluation_episode_indices.append(
                    episode + 1
                )
                evaluation_returns.append(
                    evaluation_return
                )

                writer.add_scalar(
                    "evaluation/episode_return_mean_agent",
                    evaluation_return,
                    episode + 1,
                )

        checkpoint_path: Path | None = None

        if config.checkpoint_path is not None:
            checkpoint_path = Path(
                config.checkpoint_path
            )
            learner.save(checkpoint_path)

        writer.flush()

        return TrainingResult(
            algorithm=config.algorithm,
            episodes_completed=config.episodes,
            environment_steps=global_step,
            update_count=update_count,
            mean_episode_return=float(
                np.mean(episode_returns)
            ),
            mean_evaluation_return=(
                float(
                    np.mean(evaluation_returns)
                )
                if evaluation_returns
                else None
            ),
            evaluation_runs=len(
                evaluation_returns
            ),
            training_episode_returns=tuple(
                episode_returns
            ),
            evaluation_episode_indices=tuple(
                evaluation_episode_indices
            ),
            evaluation_returns=tuple(
                evaluation_returns
            ),
            log_dir=config.log_dir,
            checkpoint_path=checkpoint_path,
        )
    finally:
        writer.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train MADDPG or independent DDPG "
            "on MPE2 Simple Spread."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a YAML training configuration.",
    )

    arguments = parser.parse_args()

    config = TrainingConfig.from_yaml(
        arguments.config
    )

    result = train_simple_spread(config)

    print("=== Training complete ===")
    print(f"Algorithm: {result.algorithm}")
    print(
        f"Episodes: {result.episodes_completed}"
    )
    print(
        f"Environment steps: "
        f"{result.environment_steps}"
    )
    print(f"Updates: {result.update_count}")
    print(
        f"Mean episode return: "
        f"{result.mean_episode_return:.4f}"
    )
    if result.mean_evaluation_return is not None:
        print(
            f"Evaluation runs: "
            f"{result.evaluation_runs}"
        )
        print(
            f"Mean evaluation return: "
            f"{result.mean_evaluation_return:.4f}"
        )

    print(f"TensorBoard log: {result.log_dir}")

    if result.checkpoint_path is not None:
        print(
            f"Checkpoint: "
            f"{result.checkpoint_path}"
        )


if __name__ == "__main__":
    main()